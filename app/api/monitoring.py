"""Monitoring and metrics API endpoints.

Provides platform-wide metrics (GPU utilization, energy, carbon) and
detailed health checks for operational monitoring and alerting.
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.hub import Hub
from app.models.workload import Workload
from app.models.carbon import CarbonOptimizationLog
from app.config import settings
from app.cache import cache

router = APIRouter()

# Track server start time for uptime calculation
_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PlatformMetrics(BaseModel):
    """Platform-wide aggregate metrics."""

    total_hubs: int = Field(..., description="Total number of GPU hubs")
    total_gpus: int = Field(..., description="Total GPUs across all hubs")
    available_gpus: int = Field(..., description="Currently available GPUs")
    gpu_utilization_percent: float = Field(..., description="GPU utilization percentage")
    total_workloads: int = Field(..., description="Total workloads ever created")
    active_workloads: int = Field(..., description="Currently running workloads")
    total_energy_kwh: float = Field(..., description="Estimated total energy consumption (kWh)")
    avg_renewable_percentage: float = Field(..., description="Average renewable energy %")
    avg_carbon_intensity: float = Field(..., description="Average grid carbon intensity (gCO2/kWh)")
    avg_pue: float = Field(..., description="Average Power Usage Effectiveness")
    total_co2_saved_kg: float = Field(..., description="Total CO2 saved via carbon-aware scheduling")


class CacheStatus(BaseModel):
    """Cache backend status."""
    available: bool = Field(..., description="Whether cache is available")
    backend: str = Field(..., description="Cache backend type (upstash_redis / in_memory)")


class DetailedHealth(BaseModel):
    """Detailed health check response."""

    status: str = Field("healthy", description="Overall system status")
    database_status: str = Field(..., description="Database connectivity status")
    cache_status: CacheStatus = Field(..., description="Cache backend status")
    api_version: str = Field(..., description="API version")
    environment: str = Field(..., description="Current environment (dev/staging/production)")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    total_endpoints: int = Field(..., description="Total number of API endpoints")
    active_connections: int = Field(0, description="Active database connections (estimated)")
    carbon_api_configured: bool = Field(..., description="Whether Electricity Maps API key is set")
    timestamp: datetime = Field(..., description="Current server timestamp")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/metrics", response_model=PlatformMetrics)
async def get_platform_metrics(
    db: AsyncSession = Depends(get_db),
):
    """Get platform-wide aggregate metrics.

    Returns GPU utilization, workload counts, energy estimates,
    carbon metrics, and CO2 savings from carbon-aware scheduling.
    """
    # Hub metrics
    hub_count_result = await db.execute(select(func.count(Hub.id)))
    total_hubs = hub_count_result.scalar() or 0

    total_gpus_result = await db.execute(select(func.sum(Hub.total_gpus)))
    total_gpus = total_gpus_result.scalar() or 0

    available_gpus_result = await db.execute(select(func.sum(Hub.available_gpus)))
    available_gpus = available_gpus_result.scalar() or 0

    avg_renewable_result = await db.execute(select(func.avg(Hub.renewable_percentage)))
    avg_renewable = avg_renewable_result.scalar() or 0.0

    avg_carbon_result = await db.execute(select(func.avg(Hub.grid_carbon_intensity)))
    avg_carbon = avg_carbon_result.scalar() or 0.0

    avg_pue_result = await db.execute(select(func.avg(Hub.pue)))
    avg_pue = avg_pue_result.scalar() or 1.0

    # Workload metrics
    total_workloads_result = await db.execute(select(func.count(Workload.id)))
    total_workloads = total_workloads_result.scalar() or 0

    active_workloads_result = await db.execute(
        select(func.count(Workload.id)).where(
            Workload.status.in_(["running", "scheduled"])
        )
    )
    active_workloads = active_workloads_result.scalar() or 0

    # CO2 savings from carbon optimization logs
    co2_saved_result = await db.execute(
        select(func.sum(CarbonOptimizationLog.carbon_saved_kg))
    )
    total_co2_saved = co2_saved_result.scalar() or 0.0

    # Estimate energy consumption (rough: 0.3 kW per GPU × available hours)
    # Using total GPUs as a proxy for capacity utilization
    gpu_utilization = (
        ((total_gpus - available_gpus) / total_gpus * 100)
        if total_gpus > 0
        else 0.0
    )

    # Rough energy estimate: running GPUs × 0.3 kW × 24h (daily snapshot)
    running_gpus = total_gpus - available_gpus
    estimated_energy_kwh = running_gpus * 0.3 * 24  # Daily estimate

    return PlatformMetrics(
        total_hubs=total_hubs,
        total_gpus=total_gpus,
        available_gpus=available_gpus,
        gpu_utilization_percent=round(gpu_utilization, 2),
        total_workloads=total_workloads,
        active_workloads=active_workloads,
        total_energy_kwh=round(estimated_energy_kwh, 2),
        avg_renewable_percentage=round(float(avg_renewable), 2),
        avg_carbon_intensity=round(float(avg_carbon), 2),
        avg_pue=round(float(avg_pue), 4),
        total_co2_saved_kg=round(float(total_co2_saved), 4),
    )


@router.get("/health/detailed", response_model=DetailedHealth)
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
):
    """Detailed health check with system information.

    Returns database status, cache status, API version, uptime, endpoint count,
    and active connection estimate.  Useful for monitoring dashboards
    and alerting systems.
    """
    # Test database connectivity
    db_status = "healthy"
    try:
        await db.execute(select(1))
    except Exception:
        db_status = "unhealthy"

    # Cache status
    cache_backend = "upstash_redis" if settings.upstash_redis_url else "in_memory"
    cache_available = cache.is_available()

    # Count total API endpoints from the app (avoid circular import by late import)
    total_endpoints = 0
    try:
        import importlib
        main_module = importlib.import_module("app.main")
        app_obj = getattr(main_module, "app", None)
        if app_obj:
            for route in app_obj.routes:
                if hasattr(route, "methods"):
                    total_endpoints += len(route.methods)
    except Exception:
        total_endpoints = 0

    uptime = time.time() - _start_time

    overall_status = "healthy"
    if db_status == "unhealthy":
        overall_status = "degraded"
    elif not cache_available:
        overall_status = "degraded"

    return DetailedHealth(
        status=overall_status,
        database_status=db_status,
        cache_status=CacheStatus(
            available=cache_available,
            backend=cache_backend,
        ),
        api_version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(uptime, 2),
        total_endpoints=total_endpoints,
        active_connections=0,
        carbon_api_configured=bool(settings.electricity_maps_api_key),
        timestamp=datetime.now(timezone.utc),
    )
