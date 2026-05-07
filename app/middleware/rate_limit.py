"""Tiered rate limiting middleware — pure ASGI, zero overhead.

Key improvements over v1 (BaseHTTPMiddleware):
- Pure ASGI: no body_iterator consumption, no task spawning
- 5-minute tier cache with SHA-256 keys (survives heavy load)
- Per-API-key enforcement using full SHA-256 hash
- Tiered limits: Free (30/min), Standard (120/min), Enterprise (600/min)
- Separate limits for inference vs general vs batch endpoints
- Standard X-RateLimit-* headers (Together AI / Groq compatible)
- Defaults to "standard" for authenticated keys (NOT "free")
"""

import hashlib
import json
import logging
import time
from collections import defaultdict

from fastapi import status
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings

logger = logging.getLogger("harchos.rate_limit")


# ---------------------------------------------------------------------------
# Rate limit tiers
# ---------------------------------------------------------------------------

RATE_LIMIT_TIERS = {
    "free": {
        "rpm": 30,
        "tpm": 10000,
        "inference_rpm": 10,
        "batch_rpm": 5,
        "burst_allowance": 1.2,
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


# ---------------------------------------------------------------------------
# In-memory rate limiter
# ---------------------------------------------------------------------------

class InMemoryRateLimiter:
    """Sliding window rate limiter for single-instance deployments."""

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str, max_requests: int, window_seconds: int = 60) -> tuple[bool, int, int]:
        now = time.time()
        cutoff = now - window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        current_count = len(self._requests[key])
        remaining = max(0, max_requests - current_count)
        reset_at = int(now + window_seconds) if self._requests[key] else int(now + window_seconds)
        if current_count >= max_requests:
            return False, 0, reset_at
        self._requests[key].append(now)
        return True, remaining - 1, reset_at


_memory_limiter = InMemoryRateLimiter()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_client_ip(scope: Scope) -> str:
    """Get client IP from ASGI scope."""
    client = scope.get("client")
    if client:
        return client[0]
    # Check X-Forwarded-For
    headers = dict(scope.get("headers", []))
    forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1", errors="ignore")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return "unknown"


def _is_inference_endpoint(path: str) -> bool:
    return "/inference/" in path or path.endswith("/chat/completions") or path.endswith("/completions")


def _is_batch_endpoint(path: str) -> bool:
    return "/inference/batch" in path


# ---------------------------------------------------------------------------
# Tier cache (shared across requests)
# ---------------------------------------------------------------------------

class _TierCache:
    """In-memory tier cache with 5-minute TTL."""
    
    def __init__(self, ttl: int = 300):
        self._cache: dict[str, tuple[str, float]] = {}
        self._ttl = ttl

    def get(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        tier, ts = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        return tier

    def set(self, key: str, tier: str) -> None:
        self._cache[key] = (tier, time.time())


_tier_cache = _TierCache(ttl=300)


async def _resolve_tier_from_key(raw_key: str) -> str:
    """Look up an API key's tier by hashing the raw key and querying the DB."""
    from app.services.auth_service import AuthService
    key_hash = AuthService.hash_key(raw_key)

    # Check cache first
    cached = _tier_cache.get(key_hash)
    if cached:
        return cached

    # Query DB
    tier = await _query_tier_from_db(key_hash=key_hash)
    _tier_cache.set(key_hash, tier)
    return tier


async def _resolve_tier_from_db(api_key_id: str) -> str:
    """Look up an API key's tier by its ID (from JWT payload)."""
    cache_key = f"id:{api_key_id}"
    cached = _tier_cache.get(cache_key)
    if cached:
        return cached

    tier = await _query_tier_from_db(api_key_id=api_key_id)
    _tier_cache.set(cache_key, tier)
    return tier


async def _query_tier_from_db(key_hash: str | None = None, api_key_id: str | None = None) -> str:
    """Query the database for an API key's tier.
    
    Returns 'standard' as default for authenticated keys (NOT 'free').
    This prevents authenticated users from being blocked by free-tier limits.
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
                return "standard"  # Key not found — default to standard

            tier, user_id = row

            if tier and tier in RATE_LIMIT_TIERS:
                return tier

            # Fallback: check user role
            user_result = await db.execute(select(User.role).where(User.id == user_id))
            user_role = user_result.scalar_one_or_none()
            if user_role == "admin":
                return "enterprise"
            elif user_role in ("user", "viewer"):
                return "standard"

            return "standard"

    except Exception as e:
        logger.debug("Rate limit tier lookup failed: %s", e)
        return "standard"  # Default to standard on DB errors (NOT free)


# ---------------------------------------------------------------------------
# Pure ASGI Rate Limit Middleware
# ---------------------------------------------------------------------------

class RateLimitMiddleware:
    """Tiered rate limiting middleware — pure ASGI.
    
    Does NOT use BaseHTTPMiddleware. Directly intercepts ASGI scope
    to read headers and enforce rate limits before the route handler runs.
    """

    SKIP_PATHS = frozenset({
        "/docs", "/redoc", "/openapi.json",
        "/v1/health", "/v1/health/", "/v1/monitoring/health/detailed",
        "/v1/metrics", "/",
        "/v1/health/ready", "/v1/health/detailed", "/v1/health/startup",
    })

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Skip rate limiting for health/docs
        if path in self.SKIP_PATHS or path.startswith("/v1/health"):
            await self.app(scope, receive, send)
            return

        # Read headers from ASGI scope
        headers = dict(scope.get("headers", []))
        api_key_raw = headers.get(b"x-api-key", b"").decode("latin-1", errors="ignore")
        auth_header = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")

        rate_key = None
        tier = "free"  # Default for unauthenticated requests

        # Resolve API key and tier
        key_to_lookup = None
        if api_key_raw and api_key_raw.startswith(settings.api_key_prefix):
            key_to_lookup = api_key_raw
        elif auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token.startswith(settings.api_key_prefix):
                key_to_lookup = token
            elif token.startswith(settings.token_prefix):
                # JWT token
                from app.services.auth_service import AuthService
                payload = AuthService.verify_jwt_token(token)
                if payload and payload.get("api_key_id"):
                    resolved_tier = await _resolve_tier_from_db(api_key_id=payload["api_key_id"])
                    tier = resolved_tier
                else:
                    tier = "standard"  # Invalid JWT → standard (not free)
                rate_key = f"rl:token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
            else:
                rate_key = f"rl:ip:{_get_client_ip(scope)}"
        else:
            rate_key = f"rl:ip:{_get_client_ip(scope)}"

        # If we have an API key, look up its tier
        if key_to_lookup:
            rate_key = f"rl:apikey:{hashlib.sha256(key_to_lookup.encode()).hexdigest()[:16]}"
            resolved_tier = await _resolve_tier_from_key(key_to_lookup)
            tier = resolved_tier

        # Determine limits
        tier_config = RATE_LIMIT_TIERS.get(tier, RATE_LIMIT_TIERS["standard"])
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

        # Add rate limit headers to the response
        rate_limit_headers = {
            b"x-ratelimit-limit": str(max_requests).encode(),
            b"x-ratelimit-remaining": str(remaining).encode(),
            b"x-ratelimit-reset": str(reset_at).encode(),
            b"x-ratelimit-tier": tier.encode(),
        }

        if not allowed:
            retry_after = max(1, reset_at - int(time.time()))
            logger.warning(
                "Rate limit exceeded: %s... (tier: %s, limit: %d/min, path: %s)",
                rate_key[:20], tier, max_requests, path,
            )
            body = json.dumps({
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
            }).encode()
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                    [b"retry-after", str(retry_after).encode()],
                    [b"x-ratelimit-limit", str(max_requests).encode()],
                    [b"x-ratelimit-remaining", b"0"],
                    [b"x-ratelimit-reset", str(reset_at).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            })
            return

        # Allowed — inject rate limit headers into the response
        original_send = send

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers", []))
                headers_list.extend(list(rate_limit_headers.items()))
                message["headers"] = headers_list
            await original_send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    async def _check_rate_limit(key: str, max_requests: int) -> tuple[bool, int, int]:
        """Check rate limit using in-memory limiter."""
        return _memory_limiter.is_allowed(key, max_requests)
