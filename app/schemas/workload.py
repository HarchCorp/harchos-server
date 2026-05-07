"""Workload Pydantic schemas matching the SDK model with comprehensive validation.

The SDK expects Kubernetes-style nested responses with
``metadata`` and ``spec`` top-level keys.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import DataResidencySpec
from app.schemas.validators import (
    validate_name,
    validate_labels,
    validate_command_list,
    validate_env_dict,
    validate_positive_int,
    validate_positive_float,
    validate_gpu_type,
    sanitize_string,
)


# ---------------------------------------------------------------------------
# Nested sub-schemas
# ---------------------------------------------------------------------------

class WorkloadCompute(BaseModel):
    """Compute requirements – matches SDK ComputeRequirements."""

    gpu_count: int = Field(0, ge=0, le=1024, description="Number of GPUs required")
    gpu_type: str | None = Field(None, description="GPU type (e.g. 'h100', 'a100')")
    cpu_cores: int = Field(1, ge=1, le=1024, description="CPU cores required")
    memory_gb: float = Field(1.0, gt=0, le=16384, description="Memory in GB")
    storage_gb: float = Field(10.0, gt=0, le=102400, description="Storage in GB")
    ephemeral_storage_gb: float | None = Field(None, ge=0, le=102400, description="Ephemeral storage in GB")

    @field_validator("gpu_type")
    @classmethod
    def validate_gpu_type_field(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_gpu_type(v)


class WorkloadCarbonMetrics(BaseModel):
    """Carbon metrics – matches SDK CarbonMetrics."""

    co2_grams: float = Field(0.0, ge=0, description="CO2 emissions in grams")
    energy_kwh: float = Field(0.0, ge=0, description="Energy consumption in kWh")
    pue: float = Field(1.0, ge=1.0, le=3.0, description="Power Usage Effectiveness")
    region_grid_intensity: float = Field(0.0, ge=0, description="Grid carbon intensity in gCO2/kWh")
    renewable_percentage: float = Field(0.0, ge=0, le=100, description="Renewable energy percentage")
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkloadSovereignty(BaseModel):
    """Sovereignty configuration – legacy, kept for backward-compat create path."""

    level: str = "standard"
    data_residency_policy: str = ""
    carbon_metrics: WorkloadCarbonMetrics = Field(default_factory=WorkloadCarbonMetrics)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("strict", "standard", "moderate", "minimal"):
            raise ValueError("Sovereignty level must be one of: strict, standard, moderate, minimal")
        return v


class WorkloadMetadata(BaseModel):
    """Resource metadata – matches SDK ResourceMetadata."""

    id: str
    name: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    resource_type: str = "workload"
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def validate_labels_field(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_labels(v)


class WorkloadSpecSchema(BaseModel):
    """Workload specification – matches SDK WorkloadSpec."""

    name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(..., description="Workload type")
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

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        return validate_name(v)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.strip().lower()
        valid_types = ("training", "inference", "fine_tuning", "evaluation", "data_pipeline", "batch")
        if v not in valid_types:
            raise ValueError(f"Workload type must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("low", "normal", "high", "critical"):
            raise ValueError("Priority must be one of: low, normal, high, critical")
        return v

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: list[str] | None) -> list[str] | None:
        return validate_command_list(v)

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_env_dict(v)

    @field_validator("sovereignty_level")
    @classmethod
    def validate_sovereignty(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("strict", "standard", "moderate", "minimal"):
            raise ValueError("Sovereignty level must be one of: strict, standard, moderate, minimal")
        return v

    @field_validator("carbon_budget_grams")
    @classmethod
    def validate_carbon_budget(cls, v: float | None) -> float | None:
        if v is not None:
            return validate_positive_float(v, "carbon_budget_grams", max_val=1e9)
        return v

    @field_validator("max_duration_seconds")
    @classmethod
    def validate_max_duration(cls, v: int | None) -> int | None:
        if v is not None:
            return validate_positive_int(v, "max_duration_seconds", max_val=86400 * 30)  # 30 days max
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels_field(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_labels(v)

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if len(v) > 512:
            raise ValueError("Image name too long (max 512 chars)")
        # Validate Docker image reference format
        if not re.match(r'^[a-zA-Z0-9._:/-]+$', v):
            raise ValueError("Invalid image format. Use Docker image reference syntax.")
        return v


# Need re import for image validation
import re

# ---------------------------------------------------------------------------
# Create / Update schemas
# ---------------------------------------------------------------------------

class WorkloadCreate(BaseModel):
    """Schema for creating a workload – accepts SDK WorkloadSpec fields."""

    name: str = Field(..., min_length=1, max_length=128)
    type: str = Field(..., description="training/inference/fine_tuning/evaluation/data_pipeline/batch")
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
    # Legacy fields (older clients)
    sovereignty: WorkloadSovereignty | None = None

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        return validate_name(v)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.strip().lower()
        valid_types = ("training", "inference", "fine_tuning", "evaluation", "data_pipeline", "batch")
        if v not in valid_types:
            raise ValueError(f"Workload type must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("low", "normal", "high", "critical"):
            raise ValueError("Priority must be one of: low, normal, high, critical")
        return v

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: list[str] | None) -> list[str] | None:
        return validate_command_list(v)

    @field_validator("env")
    @classmethod
    def validate_env(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_env_dict(v)

    @field_validator("sovereignty_level")
    @classmethod
    def validate_sovereignty(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("strict", "standard", "moderate", "minimal"):
            raise ValueError("Sovereignty level must be one of: strict, standard, moderate, minimal")
        return v

    @field_validator("carbon_budget_grams")
    @classmethod
    def validate_carbon_budget(cls, v: float | None) -> float | None:
        if v is not None:
            return validate_positive_float(v, "carbon_budget_grams", max_val=1e9)
        return v

    @field_validator("max_duration_seconds")
    @classmethod
    def validate_max_duration(cls, v: int | None) -> int | None:
        if v is not None:
            return validate_positive_int(v, "max_duration_seconds", max_val=86400 * 30)
        return v

    @field_validator("labels")
    @classmethod
    def validate_labels_field(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_labels(v)

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if len(v) > 512:
            raise ValueError("Image name too long (max 512 chars)")
        if not re.match(r'^[a-zA-Z0-9._:/-]+$', v):
            raise ValueError("Invalid image format. Use Docker image reference syntax.")
        return v


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

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_name(v)
        return v

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().lower()
            valid_types = ("training", "inference", "fine_tuning", "evaluation", "data_pipeline", "batch")
            if v not in valid_types:
                raise ValueError(f"Workload type must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().lower()
            valid_statuses = ("pending", "scheduled", "running", "paused", "completed", "failed", "cancelled")
            if v not in valid_statuses:
                raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().lower()
            if v not in ("low", "normal", "high", "critical"):
                raise ValueError("Priority must be one of: low, normal, high, critical")
        return v

    @field_validator("sovereignty_level")
    @classmethod
    def validate_sovereignty(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip().lower()
            if v not in ("strict", "standard", "moderate", "minimal"):
                raise ValueError("Sovereignty level must be one of: strict, standard, moderate, minimal")
        return v


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

    model_config = {'from_attributes': True}
