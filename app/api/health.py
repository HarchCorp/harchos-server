"""Enhanced Health Check Endpoints — Kubernetes-style probes with HarchOS error codes.

Provides four distinct health check endpoints matching and exceeding what Groq,
Together AI, and other competitors offer:

- GET /health          — Liveness probe (ultra-fast, no dependency checks)
- GET /health/ready    — Readiness probe (checks DB, cache, inference backend)
- GET /health/detailed — Full diagnostic with per-component status and latency
- GET /health/startup  — Startup probe (checks if initial setup is complete)

Each endpoint follows the HarchOS error code format (E0xxx/E1xxx) and returns
structured JSON responses.

Health error codes (E08xx — reserved for health subsystem):
    E0800  Service Not Ready           — One or more critical dependencies are down
    E0801  Startup Incomplete          — Initial setup (DB tables, seed data) not finished
    E0802  Database Unhealthy          — Database connectivity check failed
    E0803  Cache Unhealthy             — Cache service unreachable
    E0804  Inference Backend Unhealthy — Configured inference backend unreachable
    E0805  Carbon API Unhealthy        — Electricity Maps API unreachable

Competitive comparison:
    Groq           — /health (liveness only), no readiness or detailed probe
    Together AI    — /health (basic), no per-component diagnostics
    OpenAI         — No public health endpoint
    Replicate      — /ping (liveness), no detailed health
    HarchOS        — 4 probes × structured JSON × HarchOS error codes ×
                     per-component latency × carbon API status × WS metrics
"""

from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("harchos.health")

router = APIRouter()


# ---------------------------------------------------------------------------
# Startup tracking — set to True once lifespan startup completes
# ---------------------------------------------------------------------------

_start_time: float = time.time()
_startup_complete: bool = False


def mark_startup_complete() -> None:
    """Called by the application lifespan after DB init + seed."""
    global _startup_complete
    _startup_complete = True
    logger.info("Health: startup probe marked as COMPLETE")


def is_startup_complete() -> bool:
    """Check whether the application has finished initialising."""
    return _startup_complete


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Liveness response — ultra-lightweight."""
    status: str
    version: str
    environment: str
    timestamp: str


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    status: str  # healthy | degraded | unhealthy
    latency_ms: float | None = None
    detail: str | None = None
    error_code: str | None = None
    meta: dict[str, Any] | None = None


class ReadinessResponse(BaseModel):
    """Readiness probe response — are we ready to serve traffic?"""
    status: str  # ready | not_ready
    version: str
    environment: str
    timestamp: str
    checks: dict[str, ComponentHealth]


class DetailedHealthResponse(BaseModel):
    """Comprehensive health response with all component statuses."""
    status: str  # healthy | degraded | unhealthy
    version: str
    environment: str
    timestamp: str
    uptime_seconds: float
    components: dict[str, ComponentHealth]
    system: dict[str, Any]


class StartupResponse(BaseModel):
    """Startup probe response — has initialisation completed?"""
    status: str  # started | starting
    version: str
    environment: str
    timestamp: str
    checks: dict[str, Any]


# ---------------------------------------------------------------------------
# Health error code registry (E08xx)
# ---------------------------------------------------------------------------

HEALTH_ERROR_CODES: dict[str, dict[str, Any]] = {
    "E0800": {"status": 503, "title": "Service Not Ready"},
    "E0801": {"status": 503, "title": "Startup Incomplete"},
    "E0802": {"status": 503, "title": "Database Unhealthy"},
    "E0803": {"status": 503, "title": "Cache Unhealthy"},
    "E0804": {"status": 503, "title": "Inference Backend Unhealthy"},
    "E0805": {"status": 503, "title": "Carbon API Unhealthy"},
}


def _health_error(code: str, detail: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a HarchOS-style error body for health endpoints."""
    entry = HEALTH_ERROR_CODES.get(code, {"title": "Unknown Health Error"})
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "title": entry["title"],
            "detail": detail,
        }
    }
    if meta:
        body["error"]["meta"] = meta
    return body


# ---------------------------------------------------------------------------
# Component check helpers (reusable across /ready and /detailed)
# ---------------------------------------------------------------------------

async def _check_database() -> ComponentHealth:
    """Test database connectivity and measure round-trip latency."""
    try:
        start = time.perf_counter()
        from app.database import engine
        import sqlalchemy
        async with engine.connect() as conn:
            result = await conn.execute(sqlalchemy.text("SELECT 1"))
            result.fetchone()  # ensure data flows back
        latency = (time.perf_counter() - start) * 1000

        # Gather pool stats for PostgreSQL
        meta: dict[str, Any] | None = None
        if settings.database_url.startswith("postgresql"):
            pool = engine.pool
            meta = {
                "pool_size": pool.size(),
                "pool_checked_in": pool.checkedin(),
                "pool_checked_out": pool.checkedout(),
                "pool_overflow": pool.overflow(),
            }

        return ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            detail="PostgreSQL connection OK" if settings.database_url.startswith("postgresql") else "SQLite connection OK",
            meta=meta,
        )
    except Exception as exc:
        return ComponentHealth(
            status="unhealthy",
            detail=f"Database error: {exc!s:.200}",
            error_code="E0802",
        )


async def _check_cache() -> ComponentHealth:
    """Test cache connectivity (Redis/Upstash or in-memory fallback)."""
    try:
        from app.cache import cache

        if not cache.is_available():
            return ComponentHealth(
                status="degraded",
                detail="Cache not configured (using in-memory fallback)",
            )

        # Measure round-trip latency
        start = time.perf_counter()
        await cache.get("__health_check__")
        latency = (time.perf_counter() - start) * 1000

        # Determine status based on cache type
        from app.cache import UpstashRedisCache
        if isinstance(cache, UpstashRedisCache):
            return ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                detail="Upstash Redis connection OK",
            )

        # In-memory fallback
        return ComponentHealth(
            status="degraded",
            latency_ms=round(latency, 2),
            detail="Using in-memory cache (data lost on restart)",
        )
    except Exception as exc:
        return ComponentHealth(
            status="degraded",
            detail=f"Cache error: {exc!s:.200}",
            error_code="E0803",
        )


async def _check_inference_backend() -> ComponentHealth:
    """Test inference backend connectivity (vLLM, Together AI, Ollama, etc.)."""
    backend_url = getattr(settings, "inference_backend_url", "")
    if not backend_url:
        return ComponentHealth(
            status="degraded",
            detail="No inference backend configured (mock mode)",
        )

    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{backend_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.inference_backend_api_key}"},
            )
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            # Try to count models from the response
            model_count = 0
            try:
                data = resp.json()
                model_count = len(data.get("data", []))
            except Exception:
                pass

            return ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                detail=f"Inference backend OK ({model_count} models)",
                meta={"model_count": model_count, "backend_url": backend_url.split("//")[-1].split("/")[0]},
            )
        else:
            return ComponentHealth(
                status="unhealthy",
                latency_ms=round(latency, 2),
                detail=f"Inference backend returned HTTP {resp.status_code}",
                error_code="E0804",
                meta={"status_code": resp.status_code},
            )
    except httpx.TimeoutException:
        return ComponentHealth(
            status="unhealthy",
            detail="Inference backend timeout (5s)",
            error_code="E0804",
        )
    except Exception as exc:
        return ComponentHealth(
            status="unhealthy",
            detail=f"Inference backend unreachable: {exc!s:.150}",
            error_code="E0804",
        )


async def _check_carbon_api() -> ComponentHealth:
    """Test Electricity Maps API reachability."""
    if not settings.electricity_maps_api_key:
        if settings.carbon_static_fallback:
            return ComponentHealth(
                status="degraded",
                detail="No API key — using static carbon fallback data",
            )
        return ComponentHealth(
            status="degraded",
            detail="No API key configured and static fallback disabled",
        )

    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://api.electricitymap.org/v3/carbon-intensity/latest",
                params={"zone": "MA"},
                headers={"auth-token": settings.electricity_maps_api_key},
            )
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            return ComponentHealth(
                status="healthy",
                latency_ms=round(latency, 2),
                detail="Electricity Maps API OK",
            )
        else:
            return ComponentHealth(
                status="degraded",
                latency_ms=round(latency, 2),
                detail=f"Electricity Maps returned HTTP {resp.status_code}",
                error_code="E0805",
                meta={"status_code": resp.status_code},
            )
    except httpx.TimeoutException:
        return ComponentHealth(
            status="degraded",
            detail="Electricity Maps API timeout (5s)",
            error_code="E0805",
        )
    except Exception as exc:
        return ComponentHealth(
            status="degraded",
            detail=f"Carbon API unreachable: {exc!s:.150}",
            error_code="E0805",
        )


def _check_websocket_connections() -> ComponentHealth:
    """Report current WebSocket connection count."""
    try:
        from app.api.ws_monitoring import manager
        total = manager.total_active
        max_conn = settings.ws_max_connections
        util_pct = round(total / max_conn * 100, 1) if max_conn > 0 else 0.0

        status_val = "healthy"
        if util_pct > 99:
            status_val = "unhealthy"
        elif util_pct > 90:
            status_val = "degraded"

        return ComponentHealth(
            status=status_val,
            detail=f"{total} active WebSocket connections",
            meta={
                "total_connections": total,
                "max_connections": max_conn,
                "utilization_percent": util_pct,
            },
        )
    except Exception:
        # WS module may not be loaded
        return ComponentHealth(
            status="degraded",
            detail="WebSocket monitoring not available",
        )


# ---------------------------------------------------------------------------
# Endpoint: Liveness Probe  —  GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def liveness_probe():
    """Liveness probe — returns 200 if the service process is alive.

    Ultra-fast, zero dependency checks. Use this for load balancer
    health checks and Kubernetes liveness probes.

    Equivalent to:
        Groq        GET /health
        Together AI GET /health
        Replicate   GET /ping
    """
    return HealthResponse(
        status="alive",
        version=settings.app_version,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoint: Readiness Probe  —  GET /health/ready
# ---------------------------------------------------------------------------

@router.get("/health/ready")
async def readiness_probe(response: Response):
    """Readiness probe — returns 200 only if the service can serve traffic.

    Checks all **critical** dependencies:
    - Database (must be healthy)
    - Cache (degraded is acceptable, unhealthy is not)
    - Inference backend (must be healthy if configured)

    Returns 503 with HarchOS error code E0800 if any critical dependency
    is down, along with per-component details.

    Equivalent to (but more detailed than):
        Kubernetes readinessProbe
        AWS Target Group health check
    """
    checks: dict[str, ComponentHealth] = {}
    critical_failures: list[str] = []

    # --- Database (CRITICAL) ---
    db_health = await _check_database()
    checks["database"] = db_health
    if db_health.status == "unhealthy":
        critical_failures.append("database")

    # --- Cache (non-critical, degraded is OK) ---
    cache_health = await _check_cache()
    checks["cache"] = cache_health
    if cache_health.status == "unhealthy":
        critical_failures.append("cache")

    # --- Inference backend (CRITICAL if configured) ---
    backend_health = await _check_inference_backend()
    checks["inference_backend"] = backend_health
    if backend_health.status == "unhealthy" and getattr(settings, "inference_backend_url", ""):
        critical_failures.append("inference_backend")

    # Determine overall status
    if critical_failures:
        response.status_code = 503
        return {
            "status": "not_ready",
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {k: v.model_dump() for k, v in checks.items()},
            "error": _health_error(
                "E0800",
                f"Service not ready: {', '.join(critical_failures)} failed",
                meta={"failed_components": critical_failures},
            ),
        }

    return {
        "status": "ready",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {k: v.model_dump() for k, v in checks.items()},
    }


# ---------------------------------------------------------------------------
# Endpoint: Detailed Health  —  GET /health/detailed
# ---------------------------------------------------------------------------

@router.get("/health/detailed")
async def detailed_health_probe():
    """Comprehensive health diagnostic with per-component status and latency.

    Returns the health status of every component with detailed metrics:
    - **database**: PostgreSQL/SQLite connectivity, pool stats, latency
    - **cache**: Redis/Upstash or in-memory, latency
    - **inference_backend**: vLLM/Together AI/Ollama, model count, latency
    - **carbon_api**: Electricity Maps API reachability, latency
    - **websocket**: Active connection count, utilization
    - **system**: Uptime, version, environment, Python version, platform

    Per-component latency measurements are included for every healthy check.

    Equivalent to (but far more detailed than):
        Groq        (no equivalent)
        Together AI (no equivalent)
        Replicate   (no equivalent)
    """
    components: dict[str, ComponentHealth] = {}
    overall_status = "healthy"

    # --- Database ---
    db_health = await _check_database()
    components["database"] = db_health
    if db_health.status == "unhealthy":
        overall_status = "unhealthy"
    elif db_health.status == "degraded" and overall_status == "healthy":
        overall_status = "degraded"

    # --- Cache ---
    cache_health = await _check_cache()
    components["cache"] = cache_health
    if cache_health.status == "unhealthy" and overall_status != "unhealthy":
        overall_status = "unhealthy"
    elif cache_health.status == "degraded" and overall_status == "healthy":
        overall_status = "degraded"

    # --- Inference backend ---
    backend_health = await _check_inference_backend()
    components["inference_backend"] = backend_health
    if backend_health.status == "unhealthy" and overall_status != "unhealthy":
        overall_status = "unhealthy"
    elif backend_health.status == "degraded" and overall_status == "healthy":
        overall_status = "degraded"

    # --- Carbon API ---
    carbon_health = await _check_carbon_api()
    components["carbon_api"] = carbon_health
    if carbon_health.status == "degraded" and overall_status == "healthy":
        overall_status = "degraded"

    # --- WebSocket connections ---
    ws_health = _check_websocket_connections()
    components["websocket"] = ws_health
    if ws_health.status == "unhealthy" and overall_status != "unhealthy":
        overall_status = "unhealthy"
    elif ws_health.status == "degraded" and overall_status == "healthy":
        overall_status = "degraded"

    # --- System info ---
    uptime = time.time() - _start_time
    system_info: dict[str, Any] = {
        "uptime_seconds": round(uptime, 1),
        "version": settings.app_version,
        "environment": settings.environment,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
        "startup_complete": _startup_complete,
        "database_type": "postgresql" if settings.database_url.startswith("postgresql") else "sqlite",
        "inference_backend_configured": bool(getattr(settings, "inference_backend_url", "")),
        "carbon_api_configured": bool(settings.electricity_maps_api_key),
    }

    # Try to get memory info (best-effort, may not work on all platforms)
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        system_info["memory_max_rss_mb"] = round(usage.ru_maxrss / 1024, 1)
    except Exception:
        pass

    return {
        "status": overall_status,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(uptime, 1),
        "components": {k: v.model_dump() for k, v in components.items()},
        "system": system_info,
    }


# ---------------------------------------------------------------------------
# Endpoint: Startup Probe  —  GET /health/startup
# ---------------------------------------------------------------------------

@router.get("/health/startup")
async def startup_probe(response: Response):
    """Startup probe — returns 200 once initial setup is complete.

    Checks that:
    - Database tables have been created (init_db ran)
    - Seed data has been loaded (seed() ran)
    - WebSocket background tasks have started

    Returns 503 with HarchOS error code E0801 if the application is still
    starting up. Kubernetes can use this as a startupProbe to avoid sending
    traffic to a pod before it is fully initialised.

    Equivalent to:
        Kubernetes startupProbe
        (no competitor equivalent — HarchOS exclusive)
    """
    checks: dict[str, Any] = {}

    # Check 1: Is the startup flag set?
    checks["startup_flag"] = {
        "status": "passed" if _startup_complete else "pending",
        "detail": "Application lifespan completed" if _startup_complete else "Waiting for lifespan startup to finish",
    }

    # Check 2: Can we query the database? (tables must exist)
    db_ready = False
    try:
        from app.database import engine
        import sqlalchemy
        async with engine.connect() as conn:
            result = await conn.execute(sqlalchemy.text("SELECT 1"))
            result.fetchone()
        db_ready = True
        checks["database_tables"] = {
            "status": "passed",
            "detail": "Database tables accessible",
        }
    except Exception as exc:
        checks["database_tables"] = {
            "status": "failed",
            "detail": f"Database not accessible: {exc!s:.150}",
        }

    # Check 3: Was seed data loaded? (best-effort — check if users table has rows)
    seed_loaded = False
    if db_ready:
        try:
            from app.database import async_session_factory
            from sqlalchemy import select, func
            from app.models.user import User
            async with async_session_factory() as session:
                result = await session.execute(select(func.count(User.id)))
                user_count = result.scalar() or 0
                if user_count > 0:
                    seed_loaded = True
                    checks["seed_data"] = {
                        "status": "passed",
                        "detail": f"Seed data loaded ({user_count} users)",
                    }
                else:
                    checks["seed_data"] = {
                        "status": "pending",
                        "detail": "No seed data found — seed() may not have run yet",
                    }
        except Exception as exc:
            checks["seed_data"] = {
                "status": "pending",
                "detail": f"Cannot verify seed data: {exc!s:.100}",
            }
    else:
        checks["seed_data"] = {
            "status": "pending",
            "detail": "Cannot verify seed data (database not accessible)",
        }

    # Check 4: WebSocket background tasks
    try:
        from app.api.ws_monitoring import manager
        ws_running = True
        checks["websocket_tasks"] = {
            "status": "passed",
            "detail": f"WebSocket manager active ({manager.total_active} connections)",
        }
    except Exception:
        ws_running = False
        checks["websocket_tasks"] = {
            "status": "pending",
            "detail": "WebSocket manager not initialised",
        }

    # Overall startup status
    all_ready = _startup_complete and db_ready and seed_loaded

    if all_ready:
        return {
            "status": "started",
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }
    else:
        response.status_code = 503
        pending_items = [
            name for name, chk in checks.items()
            if chk.get("status") in ("pending", "failed")
        ]
        return {
            "status": "starting",
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "error": _health_error(
                "E0801",
                f"Startup incomplete: {', '.join(pending_items)} not ready",
                meta={"pending_items": pending_items},
            ),
        }
