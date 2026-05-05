"""Rate limiting middleware using sliding window with Redis or in-memory fallback."""

import time
import logging
from collections import defaultdict

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.cache import cache

logger = logging.getLogger("harchos.rate_limit")


class InMemoryRateLimiter:
    """Simple in-memory sliding window rate limiter for single-instance deployments."""

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        now = time.time()
        cutoff = now - window_seconds

        # Remove expired entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        if len(self._requests[key]) >= max_requests:
            return False

        self._requests[key].append(now)
        return True


# Global in-memory limiter instance
_memory_limiter = InMemoryRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.

    Uses Redis sliding window when available, falls back to in-memory.
    Limits by API key (authenticated) or IP address (unauthenticated).
    """

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        path = request.url.path
        if path in ("/docs", "/redoc", "/openapi.json", "/v1/health", "/v1/monitoring/health/detailed", "/"):
            return await call_next(request)

        # Determine rate limit key
        api_key = request.headers.get("X-API-Key") or ""
        auth_header = request.headers.get("Authorization", "")
        
        if api_key and api_key.startswith(settings.api_key_prefix):
            key = f"rl:apikey:{api_key[:12]}"
        elif auth_header and "Bearer" in auth_header:
            token = auth_header.replace("Bearer ", "").strip()
            if token.startswith(settings.api_key_prefix):
                key = f"rl:apikey:{token[:12]}"
            elif token.startswith(settings.token_prefix):
                key = f"rl:token:{token[:16]}"
            else:
                key = f"rl:ip:{self._get_client_ip(request)}"
        else:
            key = f"rl:ip:{self._get_client_ip(request)}"

        # Check rate limit
        max_requests = settings.rate_limit_requests_per_minute
        allowed = await self._check_rate_limit(key, max_requests)

        if not allowed:
            logger.warning("Rate limit exceeded for key: %s", key[:20])
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "retry_after_seconds": 60,
                },
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)
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
    async def _check_rate_limit(key: str, max_requests: int) -> bool:
        """Check if request is within rate limit using Redis or in-memory."""
        if cache.is_available():
            try:
                # Use Redis INCR + EXPIRE for sliding window
                import json
                current = await cache.get(key)
                if current is None:
                    await cache.set(key, "1", ttl_seconds=60)
                    return True
                
                count = int(current)
                if count >= max_requests:
                    return False
                
                await cache.set(key, str(count + 1), ttl_seconds=60)
                return True
            except Exception as e:
                logger.warning("Redis rate limit error, falling back to memory: %s", e)
                return _memory_limiter.is_allowed(key, max_requests)
        
        return _memory_limiter.is_allowed(key, max_requests)
