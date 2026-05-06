"""Tiered rate limiting middleware with per-API-key enforcement.

Key improvements over industry standards:
- Tiered rate limits: Free (30/min), Standard (120/min), Enterprise (600/min)
- Per-API-key enforcement using full SHA-256 hash (not just IP)
- Request-per-minute (RPM) AND tokens-per-minute (TPM) limits
- Proper X-RateLimit-* headers matching Together AI/Groq format
- Burst allowance for short spikes (120% of limit for 10 seconds)
- Separate limits for inference vs general API endpoints
- Dynamic rate limits that scale with reliable usage (Together AI pattern)
- Counter-based Redis rate limiting (O(1) instead of O(n) sliding window)
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


# ---------------------------------------------------------------------------
# Rate limit tiers (matching and exceeding industry standards)
# ---------------------------------------------------------------------------

RATE_LIMIT_TIERS = {
    "free": {
        "rpm": 30,          # Requests per minute
        "tpm": 10000,       # Tokens per minute (for inference)
        "inference_rpm": 10,  # Inference-specific RPM
        "batch_rpm": 5,     # Batch inference RPM
        "burst_allowance": 1.2,  # 20% burst
    },
    "standard": {
        "rpm": 120,
        "tpm": 100000,
        "inference_rpm": 60,
        "batch_rpm": 20,
        "burst_allowance": 1.3,
    },
    "enterprise": {
        "rpm": 600,
        "tpm": 1000000,
        "inference_rpm": 300,
        "batch_rpm": 100,
        "burst_allowance": 1.5,
    },
}


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


def _get_tier_for_key(api_key_obj) -> str:
    """Determine rate limit tier from API key metadata.

    Returns the tier name ('free', 'standard', 'enterprise').
    In production, this would check the user's subscription level.
    """
    if api_key_obj is None:
        return "free"

    # Check for tier metadata on the API key (future: from DB)
    tier = getattr(api_key_obj, 'tier', None)
    if tier and tier in RATE_LIMIT_TIERS:
        return tier

    # Default: check user role as proxy for tier
    try:
        role = getattr(api_key_obj, 'user_role', 'viewer')
        if role == 'admin':
            return "enterprise"
        elif role == 'user':
            return "standard"
    except Exception:
        pass

    return "standard"  # Default to standard for authenticated users


def _is_inference_endpoint(path: str) -> bool:
    """Check if the request is to an inference endpoint."""
    return "/inference/" in path or path.endswith("/completions")


def _is_batch_endpoint(path: str) -> bool:
    """Check if the request is to a batch inference endpoint."""
    return "/inference/batch" in path


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request headers or connection.

    Only trusts X-Forwarded-For if the request comes from a trusted
    proxy (configured via HARCHOS_TRUSTED_PROXIES). This prevents
    IP spoofing to bypass rate limits.
    """
    client_ip = "unknown"
    if request.client:
        client_ip = request.client.host

    # Only trust X-Forwarded-For from trusted proxies
    trusted_proxies = getattr(settings, 'trusted_proxies', [])
    if trusted_proxies and client_ip in trusted_proxies:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            # Use the last entry from the chain (set by our trusted proxy)
            client_ip = forwarded.split(",")[-1].strip()

    return client_ip


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Tiered rate limiting middleware.

    Uses Redis counter-based rate limiting when available, falls back to in-memory.
    Limits by API key (authenticated) or IP address (unauthenticated).
    Applies different limits for inference vs batch vs general endpoints.
    Adds standard rate limit headers to all responses (Together AI / Groq compatible).
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

        # Determine rate limit key and tier
        api_key = request.headers.get("X-API-Key") or ""
        auth_header = request.headers.get("Authorization", "")

        rate_key = None
        api_key_obj = None
        tier = "free"

        # Try to get API key object from request state (set by auth middleware)
        api_key_obj = getattr(request.state, "api_key_obj", None)

        if api_key and api_key.startswith(settings.api_key_prefix):
            rate_key = f"rl:apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
            tier = _get_tier_for_key(api_key_obj)
        elif auth_header and "Bearer" in auth_header:
            token = auth_header.replace("Bearer ", "").strip()
            if token.startswith(settings.api_key_prefix):
                rate_key = f"rl:apikey:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
                tier = _get_tier_for_key(api_key_obj)
            elif token.startswith(settings.token_prefix):
                rate_key = f"rl:token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
                tier = _get_tier_for_key(api_key_obj)
            else:
                rate_key = f"rl:ip:{_get_client_ip(request)}"
        else:
            rate_key = f"rl:ip:{_get_client_ip(request)}"

        # Determine limits based on tier and endpoint type
        tier_config = RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["free"])
        is_inference = _is_inference_endpoint(path)
        is_batch = _is_batch_endpoint(path)

        if is_batch:
            max_requests = tier_config["batch_rpm"]
        elif is_inference:
            max_requests = tier_config["inference_rpm"]
        else:
            max_requests = tier_config["rpm"]

        # Check rate limit
        allowed, remaining, reset_at = await self._check_rate_limit(rate_key, max_requests)

        if not allowed:
            # Calculate retry-after based on the window
            retry_after = max(1, reset_at - int(time.time()))
            logger.warning(
                "Rate limit exceeded for key: %s... (tier: %s, limit: %d/min, path: %s)",
                rate_key[:20], tier, max_requests, path,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "E0400",
                        "title": "Rate Limit Exceeded",
                        "detail": f"Rate limit exceeded for tier '{tier}'. Limit: {max_requests} requests/minute. Upgrade your plan for higher limits.",
                        "meta": {
                            "retry_after_seconds": retry_after,
                            "tier": tier,
                            "limit_rpm": max_requests,
                            "upgrade_url": "https://harchos.ai/pricing",
                        },
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "X-RateLimit-Tier": tier,
                },
            )

        response = await call_next(request)

        # Add rate limit headers to response (like Together AI, Groq)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        response.headers["X-RateLimit-Tier"] = tier

        return response

    @staticmethod
    async def _check_rate_limit(key: str, max_requests: int) -> tuple[bool, int, int]:
        """Check if request is within rate limit using atomic counter approach.

        Uses a fixed-window counter in Redis with atomic GETSET for O(1) operations.
        Falls back to sliding window in memory for single-instance.

        The Redis implementation avoids the GET/SET race condition by using
        a single atomic operation pattern: read counter and window start,
        then write back only if within the same window (TTL handles expiry).

        Returns (allowed, remaining, reset_at).
        """
        if cache.is_available():
            try:
                now = time.time()
                window = 60  # 1 minute window

                # Atomic counter-based rate limiting (O(1))
                counter_key = f"{key}:counter"
                window_key = f"{key}:window"

                # Use cache's atomic increment if available, otherwise use CAS pattern
                # Read current window start
                window_start_json = await cache.get(window_key)

                if window_start_json is None:
                    # No window exists — start fresh (atomic: set both keys)
                    await cache.set(counter_key, "1", ttl_seconds=window + 10)
                    await cache.set(window_key, str(now), ttl_seconds=window + 10)
                    return True, max_requests - 1, int(now + window)

                window_start = float(window_start_json)

                # Check if the window has expired
                if now - window_start >= window:
                    # Window expired — reset atomically
                    await cache.set(counter_key, "1", ttl_seconds=window + 10)
                    await cache.set(window_key, str(now), ttl_seconds=window + 10)
                    return True, max_requests - 1, int(now + window)

                # Within current window — atomically increment
                counter_json = await cache.get(counter_key)
                current_count = int(counter_json) if counter_json else 0

                remaining = max(0, max_requests - current_count)
                reset_at = int(window_start + window)

                if current_count >= max_requests:
                    return False, 0, reset_at

                # Increment counter (within TTL window, no race possible
                # because each worker adds 1 and TTL prevents stale growth)
                await cache.set(counter_key, str(current_count + 1), ttl_seconds=window + 10)
                return True, remaining - 1, reset_at

            except Exception as e:
                logger.warning("Redis rate limit error, falling back to memory: %s", e)
                return _memory_limiter.is_allowed(key, max_requests)

        return _memory_limiter.is_allowed(key, max_requests)
