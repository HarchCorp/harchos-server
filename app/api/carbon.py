"""Carbon-aware scheduling API endpoints.

All endpoints are under ``/v1/carbon/`` and require authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.schemas.carbon import (
    CarbonDashboardResponse,
    CarbonForecastResponse,
    CarbonIntensityZoneListResponse,
    CarbonIntensityZoneResponse,
    CarbonMetricsResponse,
    CarbonOptimalHubRequest,
    CarbonOptimalHubResponse,
    CarbonOptimizeRequest,
    CarbonOptimizeResponse,
)
from app.services.carbon_service import CarbonService
from app.api.deps import require_auth

router = APIRouter()


# ---------------------------------------------------------------------------
# Zone-level carbon intensity
# ---------------------------------------------------------------------------

@router.get("/intensity/{zone}", response_model=CarbonIntensityZoneResponse)
async def get_zone_carbon_intensity(
    zone: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get real-time carbon intensity for an electricity zone.

    Resolution: live API → cached DB record (< 30 min) → static fallback.

    Example zones: MA (Morocco), FR (France), DE (Germany), GB (UK),
    SE (Sweden), NO (Norway), IS (Iceland), etc.
    """
    return await CarbonService.get_zone_intensity(db, zone.upper())


@router.get("/intensity", response_model=CarbonIntensityZoneListResponse)
async def get_all_zone_intensities(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get carbon intensity for all known electricity zones."""
    return await CarbonService.get_all_zone_intensities(db)


# ---------------------------------------------------------------------------
# Optimal hub selection
# ---------------------------------------------------------------------------

@router.post("/optimal-hub", response_model=CarbonOptimalHubResponse)
async def find_optimal_hub(
    request: CarbonOptimalHubRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Find the carbon-optimal hub for a workload.

    Ranks all eligible hubs by current carbon intensity and recommends
    the greenest option.  If no hub meets the carbon threshold and
    ``defer_ok`` is true, it recommends deferring to the next green
    window.
    """
    return await CarbonService.find_optimal_hub(db, request)


# ---------------------------------------------------------------------------
# Workload carbon optimization
# ---------------------------------------------------------------------------

@router.post("/optimize", response_model=CarbonOptimizeResponse)
async def optimize_workload(
    request: CarbonOptimizeRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Optimize a workload's scheduling based on carbon intensity.

    This is the primary endpoint for carbon-aware scheduling.  It:

    1. Ranks hubs by carbon intensity
    2. Selects the greenest hub that meets compute requirements
    3. Decides whether to schedule immediately, defer, or reject
    4. Estimates carbon savings vs. the worst-case hub
    5. Logs the decision for audit and dashboards

    Returns the recommended action, selected hub, carbon metrics,
    and (if deferred) the next green window.
    """
    return await CarbonService.optimize_workload(db, request)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

@router.get("/forecast/{zone}", response_model=CarbonForecastResponse)
async def get_carbon_forecast(
    zone: str,
    hours: int = Query(24, ge=1, le=72, description="Forecast horizon in hours"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get a carbon intensity forecast for a zone.

    Tries the Electricity Maps forecast API first; falls back to a
    synthetic forecast based on daily solar/wind patterns.

    Returns hourly (or 15-min) forecast points plus identified green
    windows within the forecast period.
    """
    return await CarbonService.get_forecast(db, zone.upper(), hours)


# ---------------------------------------------------------------------------
# Metrics & Dashboard
# ---------------------------------------------------------------------------

@router.get("/metrics", response_model=CarbonMetricsResponse)
async def get_carbon_metrics(
    period_days: int = Query(30, ge=1, le=365, description="Metrics period in days"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get aggregate carbon metrics for the platform.

    Returns total carbon saved, workloads optimized/deferred,
    average carbon intensity, and best/worst hub info.
    """
    return await CarbonService.get_metrics(db, period_days)


@router.get("/dashboard", response_model=CarbonDashboardResponse)
async def get_carbon_dashboard(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get full carbon-aware dashboard data.

    Combines metrics, per-hub carbon intensities, recent optimization
    logs, and upcoming green windows into a single response for
    dashboard rendering.
    """
    return await CarbonService.get_dashboard(db)
