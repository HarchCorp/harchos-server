"""Health check endpoints — lightweight and detailed.

Provides both a simple liveness check and a comprehensive readiness
check that tests all dependencies (database, cache, carbon APIs).
"""

import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Simple health response."""
    status: str
    version: str
    environment: str


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    status: str  # healthy, degraded, unhealthy
    latency_ms: float | None = None
    detail: str | None = None


class DetailedHealthResponse(BaseModel):
    """Comprehensive health response with all component statuses."""
    status: str  # healthy, degraded, unhealthy
    version: str
    environment: str
    uptime_seconds: float
    components: dict[str, ComponentHealth]


# Track startup time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Lightweight liveness check — returns immediately.

    Use this for load balancer health checks. Does NOT test dependencies.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Comprehensive readiness check — tests all dependencies.

    Returns the health status of each component:
    - database: PostgreSQL connectivity
    - cache: Redis/Upstash connectivity
    - carbon_api: Electricity Maps API reachability
    """
    components: dict[str, ComponentHealth] = {}
    overall_status = "healthy"

    # Check database
    try:
        start = time.perf_counter()
        from app.database import engine
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        components["database"] = ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 1),
            detail="PostgreSQL connection OK",
        )
    except Exception as e:
        components["database"] = ComponentHealth(
            status="unhealthy",
            detail=f"Database error: {str(e)[:100]}",
        )
        overall_status = "unhealthy"

    # Check cache
    try:
        from app.cache import cache
        if cache.is_available():
            start = time.perf_counter()
            await cache.get("__health_check__")
            latency = (time.perf_counter() - start) * 1000
            components["cache"] = ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 1),
                detail="Redis/Upstash connection OK",
            )
        else:
            components["cache"] = ComponentHealth(
                status="degraded",
                detail="Cache not configured (using in-memory fallback)",
            )
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        components["cache"] = ComponentHealth(
            status="degraded",
            detail=f"Cache error: {str(e)[:100]}",
        )
        if overall_status == "healthy":
            overall_status = "degraded"

    # Check carbon API (non-blocking, best-effort)
    try:
        if settings.electricity_maps_api_key:
            import httpx
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    "https://api.electricitymap.org/v3/carbon-intensity/latest",
                    params={"zone": "MA"},
                    headers={"auth-token": settings.electricity_maps_api_key},
                )
            latency = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                components["carbon_api"] = ComponentHealth(
                    status="healthy",
                    latency_ms=round(latency, 1),
                    detail="Electricity Maps API OK",
                )
            else:
                components["carbon_api"] = ComponentHealth(
                    status="degraded",
                    latency_ms=round(latency, 1),
                    detail=f"Electricity Maps returned {resp.status_code}",
                )
        else:
            components["carbon_api"] = ComponentHealth(
                status="degraded",
                detail="No API key configured (using static fallback)",
            )
        if overall_status == "healthy" and components["carbon_api"].status == "degraded":
            overall_status = "degraded"
    except Exception as e:
        components["carbon_api"] = ComponentHealth(
            status="degraded",
            detail=f"Carbon API unreachable: {str(e)[:80]}",
        )
        if overall_status == "healthy":
            overall_status = "degraded"

    return DetailedHealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - _start_time, 1),
        components=components,
    )
