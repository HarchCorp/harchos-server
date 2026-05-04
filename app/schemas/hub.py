"""Hub Pydantic schemas matching the SDK model.

The SDK expects Kubernetes-style nested responses with
``metadata`` and ``spec`` top-level keys.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Nested sub-schemas used inside the response
# ---------------------------------------------------------------------------

class HubMetadata(BaseModel):
    """Resource metadata – matches SDK ResourceMetadata."""

    id: str
    name: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class DataResidencySpec(BaseModel):
    """Data residency policy – matches SDK DataResidencyPolicy."""

    allowed_regions: list[str] = Field(default_factory=lambda: ["morocco"])
    restricted_regions: list[str] = Field(default_factory=list)
    data_classification: str = "confidential"
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_management_region: str | None = None


class HubSpec(BaseModel):
    """Hub specification – matches SDK HubSpec."""

    name: str
    region: str
    tier: str = "standard"
    sovereignty_level: str = "strict"
    data_residency: DataResidencySpec | None = None
    gpu_types: list[str] = Field(default_factory=lambda: ["a100"])
    auto_scale: bool = True
    min_gpu_count: int = 0
    max_gpu_count: int = 8
    carbon_aware_scheduling: bool = True
    labels: dict[str, str] = Field(default_factory=dict)


class HubCapacity(BaseModel):
    """Hub capacity information – matches SDK HubCapacity."""

    total_gpus: int = 0
    available_gpus: int = 0
    total_cpu_cores: int = 0
    available_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    total_storage_gb: float = 0.0
    available_storage_gb: float = 0.0


class HubCarbonMetrics(BaseModel):
    """Carbon metrics – matches SDK CarbonMetrics."""

    co2_grams: float = 0.0
    energy_kwh: float = 0.0
    pue: float = 1.0
    region_grid_intensity: float = 0.0
    renewable_percentage: float = 0.0
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Legacy sub-schemas (still used by create / update paths)
# ---------------------------------------------------------------------------

class HubLocation(BaseModel):
    """Hub geographic location."""
    latitude: float = 0.0
    longitude: float = 0.0
    city: str = ""
    country: str = ""


class HubEnergy(BaseModel):
    """Hub energy information."""
    renewable_percentage: float = 0.0
    grid_carbon_intensity: float = 0.0
    pue: float = 1.0


# ---------------------------------------------------------------------------
# Create / Update schemas (accept what the SDK sends)
# ---------------------------------------------------------------------------

class HubCreate(BaseModel):
    """Schema for creating a hub – accepts SDK HubSpec fields."""

    name: str
    region: str
    tier: str = "standard"
    sovereignty_level: str = "strict"
    data_residency: DataResidencySpec | None = None
    gpu_types: list[str] = Field(default_factory=lambda: ["a100"])
    auto_scale: bool = True
    min_gpu_count: int = 0
    max_gpu_count: int = 8
    carbon_aware_scheduling: bool = True
    labels: dict[str, str] = Field(default_factory=dict)
    # Legacy fields that may still be sent by older clients
    capacity: HubCapacity | None = None
    location: HubLocation | None = None
    energy: HubEnergy | None = None
    data_residency_policy: str | None = None


class HubUpdate(BaseModel):
    """Schema for updating a hub."""
    name: str | None = None
    region: str | None = None
    status: str | None = None
    tier: str | None = None
    sovereignty_level: str | None = None
    data_residency: DataResidencySpec | None = None
    gpu_types: list[str] | None = None
    auto_scale: bool | None = None
    min_gpu_count: int | None = None
    max_gpu_count: int | None = None
    carbon_aware_scheduling: bool | None = None
    labels: dict[str, str] | None = None
    capacity: HubCapacity | None = None
    location: HubLocation | None = None
    energy: HubEnergy | None = None
    data_residency_policy: str | None = None


# ---------------------------------------------------------------------------
# Response schema – matches the SDK Hub model exactly
# ---------------------------------------------------------------------------

class HubResponse(BaseModel):
    """Full hub response matching the SDK Hub model."""

    metadata: HubMetadata
    spec: HubSpec
    status: str = "creating"
    capacity: HubCapacity | None = None
    carbon_metrics: HubCarbonMetrics | None = None
    endpoint: str | None = None
    active_workloads: int = 0

    class Config:
        from_attributes = True
