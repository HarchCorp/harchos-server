"""Carbon-aware scheduling Pydantic schemas.

These schemas define the request/response shapes for the
``/v1/carbon/*`` family of endpoints and match the SDK's
``CarbonOptimalHub``, ``CarbonIntensityZone``, etc. models.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Zone-level carbon intensity
# ---------------------------------------------------------------------------

class FuelMixEntry(BaseModel):
    """Single fuel type contribution in a zone's energy mix."""
    fuel_type: str = Field(..., description="Fuel type: solar, wind, hydro, nuclear, gas, coal, etc.")
    percentage: float = Field(..., ge=0, le=100, description="Percentage of this fuel in the mix")
    power_mw: Optional[float] = Field(None, ge=0, description="Power output in MW")


class CarbonIntensityZoneResponse(BaseModel):
    """Real-time carbon intensity for a single electricity zone."""
    zone: str = Field(..., description="Electricity Maps zone code (e.g. 'MA', 'FR')")
    zone_name: str = Field("", description="Human-readable zone name")
    carbon_intensity_gco2_kwh: float = Field(..., ge=0, description="Grid carbon intensity in gCO2/kWh")
    renewable_percentage: float = Field(..., ge=0, le=100, description="Renewable energy %")
    fossil_percentage: float = Field(..., ge=0, le=100, description="Fossil fuel %")
    fuel_mix: list[FuelMixEntry] = Field(default_factory=list, description="Detailed fuel mix breakdown")
    source: str = Field("electricity_maps", description="Data source")
    is_forecast: bool = Field(False, description="Whether this is a forecast")
    reading_datetime: datetime = Field(..., alias="datetime", description="Timestamp of the reading")
    updated_at: datetime = Field(..., description="When this data was fetched")

    model_config = {"from_attributes": True}


class CarbonIntensityZoneListResponse(BaseModel):
    """List of carbon intensity readings across multiple zones."""
    zones: list[CarbonIntensityZoneResponse]
    total: int


# ---------------------------------------------------------------------------
# Carbon-optimal hub selection
# ---------------------------------------------------------------------------

class CarbonOptimalHubRequest(BaseModel):
    """Request body for finding the carbon-optimal hub."""
    region: Optional[str] = Field(
        None, description="Target region filter (e.g. 'europe', 'africa')"
    )
    gpu_count: Optional[int] = Field(
        None, ge=1, description="Minimum number of GPUs required"
    )
    gpu_type: Optional[str] = Field(
        None, description="GPU type required (e.g. 'A100', 'H100')"
    )
    carbon_max_gco2: Optional[float] = Field(
        None, ge=0, description="Maximum acceptable carbon intensity in gCO2/kWh"
    )
    priority: Optional[str] = Field(
        "normal", description="Workload priority: low | normal | high | critical"
    )
    defer_ok: bool = Field(
        True, description="Whether it's acceptable to defer the workload for lower carbon"
    )


class CarbonOptimalHubResponse(BaseModel):
    """Result of carbon-optimal hub selection."""
    recommended_hub_id: Optional[str] = Field(None, description="ID of the recommended hub")
    recommended_hub_name: str = Field("", description="Name of the recommended hub")
    hub_region: str = Field("", description="Region of the recommended hub")
    hub_zone: str = Field("", description="Electricity zone of the recommended hub")
    carbon_intensity_gco2_kwh: float = Field(
        ..., ge=0, description="Current carbon intensity at the recommended hub"
    )
    renewable_percentage: float = Field(
        ..., ge=0, le=100, description="Current renewable % at the recommended hub"
    )
    available_gpus: int = Field(0, ge=0, description="Available GPUs at the recommended hub")
    action: str = Field(
        ..., description="Recommended action: schedule_now | defer | no_suitable_hub"
    )
    defer_hours: float = Field(
        0.0, ge=0, description="Hours to defer if action=defer"
    )
    defer_reason: str = Field(
        "", description="Why deferral is recommended"
    )
    estimated_carbon_saved_kg: float = Field(
        0.0, ge=0, description="Estimated CO2 saved vs worst-case hub"
    )
    alternative_hubs: list[dict] = Field(
        default_factory=list,
        description="Other hubs sorted by carbon intensity (top 5)"
    )
    analyzed_at: datetime = Field(..., description="When this analysis was performed")


# ---------------------------------------------------------------------------
# Carbon optimization (scheduling) request & response
# ---------------------------------------------------------------------------

class CarbonOptimizeRequest(BaseModel):
    """Request to optimize a workload's scheduling based on carbon intensity."""
    workload_name: str = Field(..., min_length=1, description="Workload name")
    workload_type: str = Field("training", description="Workload type")
    gpu_count: int = Field(1, ge=1, description="Number of GPUs needed")
    gpu_type: Optional[str] = Field(None, description="GPU type required")
    cpu_cores: int = Field(4, ge=1, description="CPU cores needed")
    memory_gb: float = Field(16.0, gt=0, description="Memory in GB needed")
    priority: str = Field("normal", description="low | normal | high | critical")
    carbon_aware: bool = Field(True, description="Enable carbon-aware scheduling")
    carbon_max_gco2: Optional[float] = Field(
        None, ge=0, description="Maximum carbon intensity threshold in gCO2/kWh"
    )
    region: Optional[str] = Field(None, description="Preferred region")
    estimated_duration_hours: float = Field(
        1.0, gt=0, description="Estimated workload duration in hours"
    )


class CarbonOptimizeResponse(BaseModel):
    """Result of carbon-aware workload optimization."""
    action: str = Field(
        ..., description="schedule_now | defer | reject"
    )
    workload_name: str = Field(..., description="Name of the workload")
    selected_hub_id: Optional[str] = Field(None)
    selected_hub_name: str = Field("")
    carbon_intensity_at_schedule_gco2_kwh: float = Field(0.0)
    carbon_saved_kg: float = Field(0.0, ge=0)
    baseline_carbon_kg: float = Field(0.0, ge=0)
    actual_carbon_kg: float = Field(0.0, ge=0)
    deferred_hours: float = Field(0.0, ge=0)
    reason: str = Field("", description="Human-readable explanation")
    estimated_green_window: Optional[dict] = Field(
        None, description="Next green window if deferred"
    )
    optimized_at: datetime = Field(..., description="When optimization was computed")


# ---------------------------------------------------------------------------
# Carbon metrics & dashboard
# ---------------------------------------------------------------------------

class CarbonMetricsResponse(BaseModel):
    """Aggregate carbon metrics across the platform."""
    total_carbon_saved_kg: float = Field(0.0, ge=0, description="Total CO2 saved through carbon-aware scheduling")
    total_workloads_optimized: int = Field(0, ge=0, description="Total workloads scheduled carbon-optimally")
    total_workloads_deferred: int = Field(0, ge=0, description="Total workloads deferred for lower carbon")
    average_carbon_intensity_gco2_kwh: float = Field(0.0, ge=0)
    best_hub_id: Optional[str] = Field(None)
    best_hub_name: str = Field("")
    best_hub_carbon_intensity: float = Field(0.0)
    worst_hub_carbon_intensity: float = Field(0.0)
    period_start: datetime
    period_end: datetime


class CarbonDashboardResponse(BaseModel):
    """Full carbon-aware dashboard data."""
    metrics: CarbonMetricsResponse
    hub_intensities: list[CarbonIntensityZoneResponse] = Field(
        default_factory=list, description="Current carbon intensity per hub zone"
    )
    optimization_log: list[dict] = Field(
        default_factory=list, description="Recent optimization decisions (last 20)"
    )
    green_windows: list[dict] = Field(
        default_factory=list, description="Upcoming green scheduling windows"
    )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class CarbonForecastPoint(BaseModel):
    """A single point in a carbon intensity forecast."""
    reading_datetime: datetime = Field(..., alias="datetime")
    carbon_intensity_gco2_kwh: float = Field(..., ge=0)
    renewable_percentage: float = Field(..., ge=0, le=100)
    is_green: bool = Field(False, description="Whether this point meets the green threshold")


class CarbonForecastResponse(BaseModel):
    """Carbon intensity forecast for a zone."""
    zone: str
    zone_name: str = ""
    forecast: list[CarbonForecastPoint] = Field(default_factory=list)
    green_windows: list[dict] = Field(
        default_factory=list,
        description="Identified green windows in the forecast period"
    )
