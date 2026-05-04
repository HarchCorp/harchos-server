"""Model Pydantic schemas matching the SDK model.

The SDK expects Kubernetes-style nested responses with
``metadata`` and ``spec`` top-level keys.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Nested sub-schemas
# ---------------------------------------------------------------------------

class ModelMetadata(BaseModel):
    """Resource metadata – matches SDK ResourceMetadata."""

    id: str
    name: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


class ModelSize(BaseModel):
    """Size information – matches SDK ModelSize."""

    parameters_billions: float | None = None
    size_bytes: int | None = None
    memory_required_gb: float | None = None


class ModelCapabilities(BaseModel):
    """Capabilities – matches SDK ModelCapabilities."""

    max_context_length: int = 2048
    max_output_tokens: int = 512
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supported_languages: list[str] = Field(default_factory=lambda: ["en"])
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    output_modalities: list[str] = Field(default_factory=lambda: ["text"])


class DataResidencySpec(BaseModel):
    """Data residency policy – matches SDK DataResidencyPolicy."""

    allowed_regions: list[str] = Field(default_factory=lambda: ["morocco"])
    restricted_regions: list[str] = Field(default_factory=list)
    data_classification: str = "confidential"
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_management_region: str | None = None


class ModelSpecSchema(BaseModel):
    """Model specification – matches SDK ModelSpec."""

    name: str
    framework: str = "pytorch"
    task: str = "text_generation"
    version: str = "1.0.0"
    description: str | None = None
    size: ModelSize | None = None
    capabilities: ModelCapabilities | None = None
    hub_id: str | None = None
    source_uri: str | None = None
    sovereignty_level: str = "strict"
    data_residency: DataResidencySpec | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Create / Update schemas
# ---------------------------------------------------------------------------

class ModelCreate(BaseModel):
    """Schema for creating a model – accepts SDK ModelSpec fields."""

    name: str
    framework: str = "pytorch"  # pytorch/tensorflow/jax/onnx/other
    task: str = "text_generation"
    version: str = "1.0.0"
    description: str | None = None
    size: ModelSize | None = None
    capabilities: ModelCapabilities | None = None
    hub_id: str | None = None
    source_uri: str | None = None
    sovereignty_level: str = "strict"
    data_residency: DataResidencySpec | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    # Legacy fields
    status: str | None = None
    metrics: dict[str, Any] | None = None


class ModelUpdate(BaseModel):
    """Schema for updating a model."""
    name: str | None = None
    framework: str | None = None
    task: str | None = None
    status: str | None = None
    version: str | None = None
    description: str | None = None
    size: ModelSize | None = None
    capabilities: ModelCapabilities | None = None
    hub_id: str | None = None
    source_uri: str | None = None
    sovereignty_level: str | None = None
    data_residency: DataResidencySpec | None = None
    labels: dict[str, str] | None = None
    tags: list[str] | None = None
    # Legacy fields
    metrics: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Response schema – matches the SDK Model model exactly
# ---------------------------------------------------------------------------

class ModelResponse(BaseModel):
    """Full model response matching the SDK Model model."""

    metadata: ModelMetadata
    spec: ModelSpecSchema
    status: str = "available"
    framework: str = "pytorch"
    task: str = "text_generation"
    size: ModelSize | None = None
    capabilities: ModelCapabilities | None = None
    deployed_at: datetime | None = None
    inference_endpoint: str | None = None

    model_config = {'from_attributes': True}
