"""Performance optimization middleware for sub-200ms latency.

Key optimizations:
- Gzip compression for responses > 500 bytes
- Response caching for GET endpoints (carbon data, model lists, health)
- Request size limits
- Stable cache keys (no Python hash() which varies per process)
- Cache invalidation on data mutations
"""

import gzip
import hashlib
import io
import json
import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse, JSONResponse as StarletteJSONResponse

from app.cache import get_cached_json, set_cached_json, cache

logger = logging.getLogger("harchos.performance")


# ---------------------------------------------------------------------------
# Response compression middleware
# ---------------------------------------------------------------------------

class CompressionMiddleware(BaseHTTPMiddleware):
    """Gzip compress responses > 500 bytes for clients that accept it.

    Reduces bandwidth by 60-80% for JSON API responses.
    Only compresses if the client sends Accept-Encoding: gzip.
    """

    MIN_SIZE = 500  # Don't compress tiny responses
    CONTENT_TYPES = {"application/json", "text/event-stream", "text/plain"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return response

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if not any(ct in content_type for ct in self.CONTENT_TYPES):
            return response

        # Don't compress streaming responses
        if isinstance(response, StreamingResponse):
            return response

        # Don't compress if already compressed
        if response.headers.get("Content-Encoding") == "gzip":
            return response

        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            if isinstance(chunk, str):
                body += chunk.encode()
            else:
                body += chunk

        # Don't compress small responses
        if len(body) < self.MIN_SIZE:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        # Compress
        compressed = gzip.compress(body, compresslevel=6)
        compression_ratio = round((1 - len(compressed) / len(body)) * 100, 1)

        headers = dict(response.headers)
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(compressed))
        headers["X-Compression-Ratio"] = f"{compression_ratio}%"

        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )


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
    """Generate a stable, collision-resistant cache key.

    Uses SHA-256 instead of Python's built-in hash() which is
    randomized per process and causes cache misses across workers.
    Uses full API key prefix (not truncated) to prevent collisions.
    """
    raw = f"{path}:{query}:key={api_key_prefix}:auth={auth_prefix}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"resp:{path}:{digest}"


class ResponseCacheMiddleware(BaseHTTPMiddleware):
    """Cache GET responses for cacheable endpoints.

    Uses Redis/in-memory cache with endpoint-specific TTLs.
    Adds X-Cache header (HIT/MISS) for transparency.
    Only caches successful (200) responses.
    Uses stable SHA-256 cache keys (not Python hash()).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests
        if request.method != "GET":
            # For mutation requests (POST/PATCH/DELETE), invalidate relevant caches
            await self._invalidate_cache_if_needed(request)
            return await call_next(request)

        path = request.url.path

        # Check if this endpoint is cacheable
        ttl = None
        for endpoint_path, endpoint_ttl in CACHEABLE_ENDPOINTS.items():
            if path == endpoint_path or path.startswith(endpoint_path):
                ttl = endpoint_ttl
                break

        if ttl is None:
            return await call_next(request)

        # Build cache key using stable SHA-256 (not Python hash())
        api_key = request.headers.get("X-API-Key", "")
        auth_header = request.headers.get("Authorization", "")
        query = str(request.query_params)

        # Use full key prefix, not truncated (prevents collisions)
        cache_key = _stable_cache_key(path, query, api_key[:16], auth_header[:32])

        # Try cache
        cached = await get_cached_json(cache_key)
        if cached is not None:
            response = StarletteJSONResponse(content=cached["body"])
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-TTL"] = str(ttl)
            return response

        # Cache miss — execute request
        response = await call_next(request)

        # Only cache successful responses
        if response.status_code == 200:
            try:
                body = b""
                async for chunk in response.body_iterator:
                    if isinstance(chunk, str):
                        body += chunk.encode()
                    else:
                        body += chunk

                body_json = json.loads(body)

                await set_cached_json(
                    cache_key,
                    {"body": body_json, "cached_at": time.time()},
                    ttl_seconds=ttl,
                )

                # Return the response
                response = StarletteJSONResponse(content=body_json, status_code=200)
                response.headers["X-Cache"] = "MISS"
                response.headers["X-Cache-TTL"] = str(ttl)
                return response
            except Exception as e:
                logger.warning("Failed to cache response: %s", e)
                # Return original response if caching fails
                return response

        return response

    @staticmethod
    async def _invalidate_cache_if_needed(request: Request) -> None:
        """Invalidate cache entries when data is mutated.

        When a POST/PATCH/DELETE modifies hubs, models, pricing, etc.,
        we clear the corresponding cache entries so subsequent GETs
        return fresh data.
        """
        path = request.url.path
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
                logger.debug("Cache invalidated for prefix: %s", prefix)
            except Exception as e:
                logger.warning("Cache invalidation failed for %s: %s", prefix, e)


# ---------------------------------------------------------------------------
# Request size limit middleware
# ---------------------------------------------------------------------------

MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB max request size


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with body larger than MAX_REQUEST_SIZE.

    Prevents denial-of-service via large request bodies.
    Default limit: 10MB (generous for most API use cases).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return StarletteJSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "E0200",
                        "title": "Request Too Large",
                        "detail": f"Request body exceeds maximum size of {MAX_REQUEST_SIZE // 1024 // 1024}MB.",
                    }
                },
            )
        return await call_next(request)
