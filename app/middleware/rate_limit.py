"""Rate limiting middleware with true sliding window.

Fixes:
- Uses true sliding window (not fixed-window counting)
- Uses full API key hash instead of first 12 chars (avoids key collision)
- Adds X-RateLimit-* response headers (like Together AI)
- Per-API-key and per-IP rate limiting
"""

import hashlib
import json
import logging
import time
from collections import defaultdict

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.cache import cache

logger = logging.getLogger("harchos.rate_limit")


class InMemoryRateLimiter:
    """True sliding window rate limiter for single-instance deployments.

    Each key stores a list of request timestamps. On each check:
    1. Remove timestamps older than the window
    2. If remaining count >= limit, reject
    3. Otherwise, add current timestamp and allow
    """

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> tuple[bool, int, int]:
        """Check if request is allowed. Returns (allowed, remaining, reset_at)."""
        now = time.time()
        cutoff = now - window_seconds

        # Remove expired entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        current_count = len(self._requests[key])
        remaining = max(0, max_requests - current_count)
        reset_at = int(now + window_seconds) if self._requests[key] else int(now + window_seconds)

        if current_count >= max_requests:
            return False, 0, reset_at

        self._requests[key].append(now)
        return True, remaining - 1, reset_at


# Global in-memory limiter instance
_memory_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.

    Uses Redis sliding window when available, falls back to in-memory.
    Limits by API key (authenticated) or IP address (unauthenticated).
    Adds standard rate limit headers to all responses.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks, docs, and metrics
        path = request.url.path
        skip_paths = (
            "/docs", "/redoc", "/openapi.json",
            "/v1/health", "/v1/monitoring/health/detailed",
            "/v1/metrics", "/",
        )
        if path in skip_paths:
            return await call_next(request)

        # Determine rate limit key
        api_key = request.headers.get("X-API-Key") or ""
        auth_header = request.headers.get("Authorization", "")

        rate_key = None
        if api_key and api_key.startswith(settings.api_key_prefix):
            # FIX: Use SHA-256 hash of the full key, not first 12 chars
            rate_key = f"rl:apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        elif auth_header and "Bearer" in auth_header:
            token = auth_header.replace("Bearer ", "").strip()
            if token.startswith(settings.api_key_prefix):
                rate_key = f"rl:apikey:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
            elif token.startswith(settings.token_prefix):
                # For JWT tokens, use a hash of the token
                rate_key = f"rl:token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
            else:
                rate_key = f"rl:ip:{self._get_client_ip(request)}"
        else:
            rate_key = f"rl:ip:{self._get_client_ip(request)}"

        # Check rate limit
        max_requests = settings.rate_limit_requests_per_minute
        allowed, remaining, reset_at = await self._check_rate_limit(rate_key, max_requests)

        if not allowed:
            logger.warning("Rate limit exceeded for key: %s", rate_key[:30])
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "E0400",
                        "title": "Rate Limit Exceeded",
                        "detail": "Rate limit exceeded. Please slow down.",
                        "meta": {"retry_after_seconds": 60},
                    }
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                },
            )

        response = await call_next(request)

        # Add rate limit headers to response (like Together AI)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)

        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request headers or connection."""
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    @staticmethod
    async def _check_rate_limit(key: str, max_requests: int) -> tuple[bool, int, int]:
        """Check if request is within rate limit using true sliding window.

        Returns (allowed, remaining, reset_at).
        """
        if cache.is_available():
            try:
                now = time.time()
                window = 60  # 1 minute window

                # Use Redis sorted set for true sliding window
                # Key: rate limit counter, Member: timestamp
                window_key = f"{key}:window"

                # Get current window count
                current_json = await cache.get(window_key)
                if current_json is None:
                    # New window
                    await cache.set(
                        window_key,
                        json.dumps([now]),
                        ttl_seconds=window + 10,
                    )
                    return True, max_requests - 1, int(now + window)

                timestamps = json.loads(current_json)
                cutoff = now - window

                # Filter to only recent timestamps
                timestamps = [t for t in timestamps if t > cutoff]
                current_count = len(timestamps)
                remaining = max(0, max_requests - current_count)

                if current_count >= max_requests:
                    reset_at = int(min(timestamps) + window) if timestamps else int(now + window)
                    return False, 0, reset_at

                # Add current request
                timestamps.append(now)
                await cache.set(
                    window_key,
                    json.dumps(timestamps),
                    ttl_seconds=window + 10,
                )
                return True, remaining - 1, int(now + window)

            except Exception as e:
                logger.warning("Redis rate limit error, falling back to memory: %s", e)
                return _memory_limiter.is_allowed(key, max_requests)

        return _memory_limiter.is_allowed(key, max_requests)
