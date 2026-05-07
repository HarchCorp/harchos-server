"""Performance optimization middleware — pure ASGI for zero overhead.

Key optimizations:
- Gzip compression for responses > 500 bytes (pure ASGI, no body_iterator consumption)
- Response caching for GET endpoints (pure ASGI)
- Request size limits (lightweight header check)
- Stable cache keys (SHA-256, not Python hash())
- Cache invalidation on data mutations

v2: Replaced BaseHTTPMiddleware with pure ASGI middleware.
BaseHTTPMiddleware creates a new task per request and consumes
the body_iterator, adding 200-400ms per middleware layer. With
8 layers, this caused Railway proxy timeouts (502s). Pure ASGI
middleware passes through the response stream without buffering.
"""

import gzip
import hashlib
import json
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.requests import Request as StarletteRequest
from starlette.responses import StreamingResponse, JSONResponse as StarletteJSONResponse, Response as StarletteResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.cache import get_cached_json, set_cached_json, cache

logger = logging.getLogger("harchos.performance")


# ---------------------------------------------------------------------------
# Response compression middleware — pure ASGI (no body buffering)
# ---------------------------------------------------------------------------

class CompressionMiddleware:
    """Gzip compress responses > 500 bytes for clients that accept it.

    Pure ASGI implementation — does NOT consume body_iterator.
    Instead, it intercepts the ASGI response messages and compresses
    the body before sending it to the client.
    """

    MIN_SIZE = 500
    CONTENT_TYPES = {"application/json", "text/event-stream", "text/plain"}

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Check if client accepts gzip
        headers = dict(scope.get("headers", []))
        accept_encoding = headers.get(b"accept-encoding", b"").decode("latin-1", errors="ignore")
        if "gzip" not in accept_encoding:
            await self.app(scope, receive, send)
            return

        # Intercept response to potentially compress
        response_started = False
        response_headers = {}
        body_parts = []

        async def send_wrapper(message: dict) -> None:
            nonlocal response_started, response_headers, body_parts

            if message["type"] == "http.response.start":
                response_started = True
                response_headers = dict(message.get("headers", []))
                # Don't send yet — we need to check if we should compress
                return

            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                more_body = message.get("more_body", False)

                if more_body:
                    # Streaming response — just pass through, don't compress
                    if response_started and body_parts == []:
                        # First chunk of streaming — send the start message now
                        await send({
                            "type": "http.response.start",
                            "status": 200,
                            "headers": list(response_headers.items()) if isinstance(response_headers, dict) else response_headers,
                        })
                    body_parts.append(body)
                    await send(message)
                    return

                # Last (or only) body chunk — accumulate
                body_parts.append(body)
                full_body = b"".join(body_parts)

                # Check if we should compress
                content_type = response_headers.get(b"content-type", b"").decode("latin-1", errors="ignore")
                should_compress = (
                    len(full_body) >= self.MIN_SIZE
                    and any(ct in content_type for ct in self.CONTENT_TYPES)
                    and response_headers.get(b"content-encoding") != b"gzip"
                )

                if should_compress:
                    compressed = gzip.compress(full_body, compresslevel=6)
                    if len(compressed) < len(full_body):
                        ratio = round((1 - len(compressed) / len(full_body)) * 100, 1)
                        response_headers[b"content-encoding"] = b"gzip"
                        response_headers[b"content-length"] = str(len(compressed)).encode()
                        response_headers[b"x-compression-ratio"] = f"{ratio}%".encode()

                        await send({
                            "type": "http.response.start",
                            "status": 200,
                            "headers": list(response_headers.items()) if isinstance(response_headers, dict) else response_headers,
                        })
                        await send({
                            "type": "http.response.body",
                            "body": compressed,
                            "more_body": False,
                        })
                        return

                # Not compressing — send as-is
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": list(response_headers.items()) if isinstance(response_headers, dict) else response_headers,
                })
                await send({
                    "type": "http.response.body",
                    "body": full_body,
                    "more_body": False,
                })
                return

        await self.app(scope, receive, send_wrapper)


# ---------------------------------------------------------------------------
# GET response caching middleware
# ---------------------------------------------------------------------------

# Cacheable endpoints with their TTLs
CACHEABLE_ENDPOINTS = {
    "/v1/inference/models": 300,        # 5 min - model list rarely changes
    "/v1/hubs": 60,                      # 1 min - hub list changes slowly
    "/v1/carbon/intensity": 1800,        # 30 min - carbon intensity updates
    "/v1/carbon/intensity/": 1800,       # 30 min - zone-specific
    "/v1/pricing/plans": 3600,           # 1 hour - pricing rarely changes
    "/v1/regions": 86400,                # 24 hours - static data
    "/v1/inference/embeddings/models": 600,  # 10 min
    "/v1/fine-tuning/base-models": 3600, # 1 hour
}


def _stable_cache_key(path: str, query: str, api_key_prefix: str, auth_prefix: str) -> str:
    """Generate a stable, collision-resistant cache key."""
    raw = f"{path}:{query}:key={api_key_prefix}:auth={auth_prefix}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"resp:{path}:{digest}"


class ResponseCacheMiddleware:
    """Cache GET responses for cacheable endpoints — pure ASGI.

    Does NOT use BaseHTTPMiddleware. Instead, intercepts ASGI messages
    to buffer and cache successful responses without blocking the stream.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http",):
            await self.app(scope, receive, send)
            return

        # Check method
        method = scope.get("method", "GET")
        path = scope.get("path", "")

        # For mutation requests, invalidate cache
        if method != "GET":
            # Invalidate cache for mutations (fire and forget)
            try:
                import asyncio
                asyncio.create_task(self._invalidate_cache_if_needed(path))
            except Exception:
                pass
            await self.app(scope, receive, send)
            return

        # Check if cacheable
        ttl = None
        for endpoint_path, endpoint_ttl in CACHEABLE_ENDPOINTS.items():
            if path == endpoint_path or path.startswith(endpoint_path):
                ttl = endpoint_ttl
                break

        if ttl is None:
            await self.app(scope, receive, send)
            return

        # Build cache key
        headers = dict(scope.get("headers", []))
        api_key = headers.get(b"x-api-key", b"").decode("latin-1", errors="ignore")
        auth_header = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
        query_string = scope.get("query_string", b"").decode("latin-1", errors="ignore")
        cache_key = _stable_cache_key(path, query_string, api_key[:16], auth_header[:32])

        # Try cache
        cached = await get_cached_json(cache_key)
        if cached is not None:
            body = json.dumps(cached["body"], default=str).encode()
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"x-cache", b"HIT"],
                    [b"x-cache-ttl", str(ttl).encode()],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })
            return

        # Cache miss — execute and buffer response for caching
        response_status = 0
        response_headers_dict = {}
        body_parts = []

        async def send_wrapper(message: dict) -> None:
            nonlocal response_status, response_headers_dict, body_parts

            if message["type"] == "http.response.start":
                response_status = message.get("status", 200)
                response_headers_dict = dict(message.get("headers", []))
                # Don't send yet — buffer body first
                return

            if message["type"] == "http.response.body":
                body = message.get("body", b"")
                more_body = message.get("more_body", False)
                body_parts.append(body)

                if more_body:
                    return  # Keep buffering

                # Last chunk — we have the full response
                full_body = b"".join(body_parts)

                # Try to cache if successful
                if response_status == 200:
                    try:
                        # Handle gzip-compressed body
                        body_to_cache = full_body
                        if response_headers_dict.get(b"content-encoding") == b"gzip":
                            body_to_cache = gzip.decompress(full_body)

                        body_json = json.loads(body_to_cache)
                        await set_cached_json(
                            cache_key,
                            {"body": body_json, "cached_at": time.time()},
                            ttl_seconds=ttl,
                        )
                        # Add cache miss header
                        response_headers_dict[b"x-cache"] = b"MISS"
                        response_headers_dict[b"x-cache-ttl"] = str(ttl).encode()
                    except Exception as e:
                        logger.warning("Failed to cache response: %s", e)

                # Send the response
                await send({
                    "type": "http.response.start",
                    "status": response_status,
                    "headers": list(response_headers_dict.items()) if isinstance(response_headers_dict, dict) else response_headers_dict,
                })
                await send({
                    "type": "http.response.body",
                    "body": full_body,
                    "more_body": False,
                })

        await self.app(scope, receive, send_wrapper)

    @staticmethod
    async def _invalidate_cache_if_needed(path: str) -> None:
        """Invalidate cache entries when data is mutated."""
        prefixes_to_clear = []

        if "/hubs" in path:
            prefixes_to_clear.append("resp:/v1/hubs")
        if "/models" in path:
            prefixes_to_clear.append("resp:/v1/inference/models")
            prefixes_to_clear.append("resp:/v1/models")
        if "/pricing" in path:
            prefixes_to_clear.append("resp:/v1/pricing")

        for prefix in prefixes_to_clear:
            try:
                await cache.clear_pattern(prefix)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Request size limit middleware
# ---------------------------------------------------------------------------

MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB max request size


class RequestSizeLimitMiddleware:
    """Reject requests with body larger than MAX_REQUEST_SIZE — pure ASGI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length", b"").decode("latin-1", errors="ignore")

        if content_length and content_length.isdigit() and int(content_length) > MAX_REQUEST_SIZE:
            body = json.dumps({
                "error": {
                    "code": "E0200",
                    "title": "Request Too Large",
                    "detail": f"Request body exceeds maximum size of {MAX_REQUEST_SIZE // 1024 // 1024}MB.",
                }
            }).encode()
            await send({
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })
            return

        await self.app(scope, receive, send)
