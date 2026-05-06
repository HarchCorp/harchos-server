"""Redis caching layer using Upstash Redis (REST API) or in-memory fallback.

When HARCHOS_UPSTASH_REDIS_URL and HARCHOS_UPSTASH_REDIS_TOKEN are set,
caching uses Upstash Redis via the REST API (no TCP connection needed,
perfect for serverless/Railway). Otherwise, falls back to an in-memory
TTL cache that resets on each deploy.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger("harchos.cache")


class InMemoryCache:
    """Simple in-memory TTL cache as fallback when Redis is unavailable.

    Data is LOST on every deploy/restart — this is intentional as a
    fallback. For persistent caching, configure Upstash Redis.

    Includes a maximum size limit to prevent unbounded memory growth.
    When the limit is reached, the oldest entries are evicted first.
    """

    MAX_ENTRIES = 10000  # Prevent unbounded memory growth

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}
        logger.info("Cache: using in-memory fallback (data lost on restart, max %d entries)", self.MAX_ENTRIES)

    async def get(self, key: str) -> Optional[str]:
        """Get a cached value by key. Returns None if expired or missing."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    async def set(self, key: str, value: str, ttl_seconds: int = 1800) -> None:
        """Set a cached value with TTL in seconds.

        Evicts expired entries and oldest entries when MAX_ENTRIES is reached.
        """
        # Evict expired entries first
        now = time.time()
        expired_keys = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired_keys:
            del self._store[k]

        # If still at capacity, evict the oldest entries (smallest expiry time)
        if len(self._store) >= self.MAX_ENTRIES:
            # Sort by expiry time and remove the 10% oldest
            sorted_keys = sorted(self._store.keys(), key=lambda k: self._store[k][1])
            to_remove = sorted_keys[:max(1, len(sorted_keys) // 10)]
            for k in to_remove:
                del self._store[k]

        self._store[key] = (value, now + ttl_seconds)

    async def delete(self, key: str) -> None:
        """Delete a cached key."""
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        val = await self.get(key)
        return val is not None

    def is_available(self) -> bool:
        return True  # Always available (in-memory)

    async def clear_pattern(self, prefix: str) -> int:
        """Delete all keys matching a prefix. Returns count deleted."""
        keys_to_delete = [k for k in self._store if k.startswith(prefix)]
        for k in keys_to_delete:
            del self._store[k]
        return len(keys_to_delete)


class UpstashRedisCache:
    """Redis cache using Upstash REST API.

    Uses the upstash-redis Python SDK for serverless-friendly caching.
    No persistent TCP connection — all requests go over HTTPS.
    """

    def __init__(self, url: str, token: str):
        try:
            from upstash_redis.asyncio import Redis
            self._redis = Redis(url=url, token=token)
            self._available = True
            logger.info("Cache: using Upstash Redis (%s)", url.split("//")[1].split(".")[0])
        except ImportError:
            logger.warning("upstash-redis not installed, falling back to in-memory cache")
            self._available = False
            self._redis = None
        except Exception as exc:
            logger.warning("Upstash Redis init failed: %s, falling back", exc)
            self._available = False
            self._redis = None

    async def get(self, key: str) -> Optional[str]:
        """Get a cached value by key."""
        if not self._available or not self._redis:
            return None
        try:
            return await self._redis.get(key)
        except Exception as exc:
            logger.warning("Redis GET error for key %s: %s", key, exc)
            return None

    async def set(self, key: str, value: str, ttl_seconds: int = 1800) -> None:
        """Set a cached value with TTL in seconds."""
        if not self._available or not self._redis:
            return
        try:
            await self._redis.set(key, value, ex=ttl_seconds)
        except Exception as exc:
            logger.warning("Redis SET error for key %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        """Delete a cached key."""
        if not self._available or not self._redis:
            return
        try:
            await self._redis.delete(key)
        except Exception as exc:
            logger.warning("Redis DELETE error for key %s: %s", key, exc)

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not self._available or not self._redis:
            return False
        try:
            result = await self._redis.exists(key)
            return bool(result)
        except Exception as exc:
            logger.warning("Redis EXISTS error for key %s: %s", key, exc)
            return False

    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._available

    async def clear_pattern(self, prefix: str) -> int:
        """Delete all keys matching a prefix using SCAN."""
        if not self._available or not self._redis:
            return 0
        try:
            # Upstash doesn't support SCAN well, use keys() for small datasets
            keys = await self._redis.keys(f"{prefix}*")
            if keys:
                await self._redis.delete(*keys)
                return len(keys)
            return 0
        except Exception as exc:
            logger.warning("Redis clear_pattern error: %s", exc)
            return 0


# ---------------------------------------------------------------------------
# Global cache instance — auto-detects Upstash vs in-memory
# ---------------------------------------------------------------------------

def _create_cache():
    """Create the appropriate cache backend based on configuration."""
    if settings.upstash_redis_url and settings.upstash_redis_token:
        return UpstashRedisCache(
            url=settings.upstash_redis_url,
            token=settings.upstash_redis_token,
        )
    return InMemoryCache()


cache = _create_cache()


# ---------------------------------------------------------------------------
# Helper functions for common caching patterns
# ---------------------------------------------------------------------------

async def get_cached_json(key: str) -> Optional[dict | list]:
    """Get a cached JSON value by key."""
    raw = await cache.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def set_cached_json(key: str, value: dict | list, ttl_seconds: int = 1800) -> None:
    """Set a cached JSON value with TTL."""
    await cache.set(key, json.dumps(value, default=str), ttl_seconds=ttl_seconds)


async def get_or_fetch(
    key: str,
    fetch_fn,
    ttl_seconds: int = 1800,
) -> Any:
    """Get from cache, or fetch and cache the result.

    Args:
        key: Cache key
        fetch_fn: Async callable that returns the value to cache
        ttl_seconds: Time-to-live in seconds (default 30 min)

    Returns:
        The cached or freshly fetched value
    """
    # Try cache first
    cached = await get_cached_json(key)
    if cached is not None:
        return cached

    # Fetch fresh data
    result = await fetch_fn()

    # Cache the result
    if result is not None:
        await set_cached_json(key, result, ttl_seconds=ttl_seconds)

    return result
