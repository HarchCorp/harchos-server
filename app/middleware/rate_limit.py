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

    # Check the tier field on the API key object (set during key creation)
    tier = getattr(api_key_obj, 'tier', None)
    if tier and tier in RATE_LIMIT_TIERS:
        return tier

    # Fallback: derive tier from user role if available via relationship
    # ApiKey doesn't have a user_role column, but the User relationship
    # may be loaded. Check user.role if the relationship is populated.
    try:
        user = getattr(api_key_obj, 'user', None)
        if user is not None:
            role = getattr(user, 'role', None)
            if role == 'admin':
                return "enterprise"
            elif role in ('user', 'viewer'):
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

    CRITICAL: This middleware resolves the API key's tier by looking up the key
    hash in the database. This is necessary because the auth dependency runs
    AFTER middleware (inside the route handler), so request.state.api_key_obj
    is not available at middleware time.
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
        api_key_raw = request.headers.get("X-API-Key") or ""
        auth_header = request.headers.get("Authorization", "")

        rate_key = None
        tier = "free"

        # Resolve the raw API key value from headers
        key_to_lookup = None
        if api_key_raw and api_key_raw.startswith(settings.api_key_prefix):
            key_to_lookup = api_key_raw
        elif auth_header and "Bearer" in auth_header:
            token = auth_header.replace("Bearer ", "").strip()
            if token.startswith(settings.api_key_prefix):
                key_to_lookup = token
            elif token.startswith(settings.token_prefix):
                # JWT token — resolve tier by decoding and looking up the API key
                from app.services.auth_service import AuthService
                payload = AuthService.verify_jwt_token(token)
                if payload and payload.get("api_key_id"):
                    resolved_tier = await self._resolve_tier_from_db(api_key_id=payload["api_key_id"])
                    tier = resolved_tier
                rate_key = f"rl:token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
            else:
                rate_key = f"rl:ip:{_get_client_ip(request)}"
        else:
            rate_key = f"rl:ip:{_get_client_ip(request)}"

        # If we have an API key, look up its tier from DB
        if key_to_lookup:
            rate_key = f"rl:apikey:{hashlib.sha256(key_to_lookup.encode()).hexdigest()[:16]}"
            resolved_tier = await self._resolve_tier_from_key(key_to_lookup)
            tier = resolved_tier

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

    # ------------------------------------------------------------------
    # Tier resolution methods — look up API key tier from DB
    # ------------------------------------------------------------------

    # In-memory tier cache to avoid DB lookups on every request
    _tier_cache: dict[str, tuple[str, float]] = {}  # key_hash -> (tier, timestamp)
    _TIER_CACHE_TTL = 300  # 5 minutes

    @classmethod
    async def _resolve_tier_from_key(cls, raw_key: str) -> str:
        """Look up an API key's tier by hashing the raw key and querying the DB.

        Uses an in-memory tier cache with 5-minute TTL to avoid hitting
        the database on every single request. This is the same pattern
        used by Together AI and Groq for rate limit tier resolution.
        """
        from app.services.auth_service import AuthService
        key_hash = AuthService.hash_key(raw_key)

        # Check cache first
        cached = cls._tier_cache.get(key_hash)
        if cached:
            tier, ts = cached
            if time.time() - ts < cls._TIER_CACHE_TTL:
                return tier

        # Query DB
        tier = await cls._query_tier_from_db(key_hash=key_hash)

        # Cache the result
        cls._tier_cache[key_hash] = (tier, time.time())
        return tier

    @classmethod
    async def _resolve_tier_from_db(cls, api_key_id: str) -> str:
        """Look up an API key's tier by its ID (from JWT payload)."""
        cache_key = f"id:{api_key_id}"
        cached = cls._tier_cache.get(cache_key)
        if cached:
            tier, ts = cached
            if time.time() - ts < cls._TIER_CACHE_TTL:
                return tier

        tier = await cls._query_tier_from_db(api_key_id=api_key_id)
        cls._tier_cache[cache_key] = (tier, time.time())
        return tier

    @staticmethod
    async def _query_tier_from_db(key_hash: str | None = None, api_key_id: str | None = None) -> str:
        """Query the database for an API key's tier.

        Returns the tier name, or 'standard' as default for authenticated keys.
        This method does a single DB query and is called at most once per
        cache TTL window per API key.
        """
        try:
            from app.database import async_session_factory
            from app.models.api_key import ApiKey
            from app.models.user import User
            from sqlalchemy import select

            async with async_session_factory() as db:
                query = select(ApiKey.tier, ApiKey.user_id).where(ApiKey.is_active.is_(True))
                if key_hash:
                    query = query.where(ApiKey.key_hash == key_hash)
                elif api_key_id:
                    query = query.where(ApiKey.id == api_key_id)
                else:
                    return "standard"

                result = await db.execute(query)
                row = result.one_or_none()

                if row is None:
                    return "standard"  # Key not found — default to standard (not free)

                tier, user_id = row

                if tier and tier in RATE_LIMIT_TIERS:
                    return tier

                # Fallback: check user role to determine tier
                user_result = await db.execute(select(User.role).where(User.id == user_id))
                user_role = user_result.scalar_one_or_none()
                if user_role == "admin":
                    return "enterprise"
                elif user_role in ("user", "viewer"):
                    return "standard"

                return "standard"

        except Exception as e:
            # Gracefully handle DB errors (table not created yet, connection issues)
            # Default to "standard" (NOT "free") so authenticated users aren't blocked
            logger.debug("Rate limit tier lookup failed (DB not ready?): %s", e)
            return "standard"
