"""Shared httpx.AsyncClient with connection pooling.

All outgoing HTTP requests should use this shared client instead of creating
ephemeral `async with httpx.AsyncClient()` instances. This enables:
- Connection reuse (keep-alive) across requests
- Configurable pool limits
- Consistent timeout defaults
- Proper lifecycle management (created on startup, closed on shutdown)
"""

import logging

import httpx

logger = logging.getLogger("harchos.http_client")

# Module-level shared client — initialized on startup, closed on shutdown
_shared_client: httpx.AsyncClient | None = None


def get_shared_client() -> httpx.AsyncClient:
    """Get the shared httpx.AsyncClient instance.

    Returns the existing client if already initialized, otherwise creates
    a new one with sensible defaults.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=60,
            ),
            follow_redirects=True,
        )
    return _shared_client


async def close_shared_client() -> None:
    """Close the shared httpx.AsyncClient. Called on app shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        logger.info("Shared httpx.AsyncClient closed")
    _shared_client = None
