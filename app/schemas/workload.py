"""Workload Pydantic schemas matching the SDK model.

The SDK expects Kubernetes-style nested responses with
``metadata`` and ``spec`` top-level keys.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Nested sub-schemas
# ---------------------------------------------------------------------------

class WorkloadCompute(BaseModel):
    """Compute requirements – matches SDK ComputeRequirements."""

    gpu_count: int = 0
    gpu_type: str | None = None
    cpu_cores: int = 1
    memory_gb: float = 1.0
    storage_gb: float = 10.0
    ephemeral_storage_gb: float | None = None

class WorkloadCarbonMetrics(BaseModel):
    """Carbon metrics – matches SDK CarbonMetrics."""

    co2_grams: float = 0.0
    energy_kwh: float = 0.0
    pue: float = 1.0
    region_grid_intensity: float = 0.0
    renewable_percentage: float = 0.0
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WorkloadSovereignty(BaseModel):
    """Sovereignty configuration – legacy, kept for backward-compat create path."""

    level: str = "standard"
    data_residency_policy: str = ""
    carbon_metrics: WorkloadCarbonMetrics = Field(default_factory=WorkloadCarbonMetrics)

class WorkloadMetadata(BaseModel):
    """Resource metadata – matches SDK ResourceMetadata."""

    id: str
    name: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    resource_type: str = "workload"
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

class WorkloadSpecSchema(BaseModel):
    """Workload specification – matches SDK WorkloadSpec."""

    name: str
    type: str
    model_id: str | None = None
    hub_id: str | None = None
    compute: WorkloadCompute = Field(default_factory=WorkloadCompute)
    priority: str = "normal"
    image: str | None = None
    command: list[str] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    sovereignty_level: str = "strict"
    data_residency: DataResidencySpec | None = None
    carbon_budget_grams: float | None = None
    max_duration_seconds: int | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    auto_restart: bool = False

# ---------------------------------------------------------------------------
# Create / Update schemas
# ---------------------------------------------------------------------------

class WorkloadCreate(BaseModel):
    """Schema for creating a workload – accepts SDK WorkloadSpec fields."""

    name: str
    type: str  # training/inference/fine_tuning/evaluation/data_pipeline/batch
    model_id: str | None = None
    hub_id: str | None = None
    compute: WorkloadCompute = Field(default_factory=WorkloadCompute)
    priority: str = "normal"  # low/normal/high/critical
    image: str | None = None
    command: list[str] | None = None
    env: dict[str, str] = Field(default_factory=dict)
    sovereignty_level: str = "strict"
    data_residency: DataResidencySpec | None = None
    carbon_budget_grams: float | None = None
    max_duration_seconds: int | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    auto_restart: bool = False
    # Legacy fields (older clients)
    sovereignty: WorkloadSovereignty | None = None

class WorkloadUpdate(BaseModel):
    """Schema for updating a workload."""
    name: str | None = None
    type: str | None = None
    status: str | None = None
    compute: WorkloadCompute | None = None
    hub_id: str | None = None
    priority: str | None = None
    sovereignty: WorkloadSovereignty | None = None
    sovereignty_level: str | None = None
    data_residency: DataResidencySpec | None = None

# ---------------------------------------------------------------------------
# Response schema – matches the SDK Workload model exactly
# ---------------------------------------------------------------------------

class WorkloadResponse(BaseModel):
    """Full workload response matching the SDK Workload model."""

    metadata: WorkloadMetadata
    spec: WorkloadSpecSchema
    status: str = "pending"
    hub_id: str | None = None
    carbon_metrics: WorkloadCarbonMetrics | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0

    class Config:
        from_attributes = True
