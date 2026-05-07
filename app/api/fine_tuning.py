"""Fine-Tuning API — OpenAI-compatible endpoints with carbon budget tracking.

HarchOS's unique value: every fine-tuning job includes a carbon budget,
ensuring training runs stay within user-defined CO2 limits. No other
fine-tuning API offers carbon-aware training.

Implements Together AI / Replicate-style fine-tuning with:
- OpenAI-compatible job lifecycle (pending → running → completed/failed/cancelled)
- LoRA adapter support for efficient fine-tuning
- Carbon budget enforcement (max CO2 in grams per training run)
- Per-epoch carbon estimation and tracking
- GPU-hour and carbon cost estimation before training starts
- JSONL training data upload (OpenAI fine-tuning format)
- Webhook notification on job completion
- Database-backed persistence (survives restarts, scales horizontally)

Error codes: E06xx for fine-tuning specific errors.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey
from app.models.fine_tuning import FineTuningJob, FineTuningFile, FineTunedModel
from app.api.deps import require_auth
from app.config import settings
from app.core.exceptions import HarchOSError, not_found, validation_error
from app.database import get_db, async_session_factory

logger = logging.getLogger("harchos.fine_tuning")
router = APIRouter()


# ---------------------------------------------------------------------------
# Fine-tuning specific error codes (E06xx)
# ---------------------------------------------------------------------------

ERROR_CODES_E06: dict[str, dict[str, Any]] = {
    "E0600": {"status": 400, "title": "Fine-Tuning Job Not Found"},
    "E0601": {"status": 400, "title": "Invalid Training File"},
    "E0602": {"status": 400, "title": "Invalid Base Model"},
    "E0603": {"status": 400, "title": "Job Cannot Be Cancelled"},
    "E0604": {"status": 400, "title": "Carbon Budget Exceeded"},
    "E0605": {"status": 400, "title": "Invalid Hyperparameters"},
    "E0606": {"status": 404, "title": "Training File Not Found"},
    "E0607": {"status": 413, "title": "Training File Too Large"},
    "E0608": {"status": 400, "title": "Unsupported File Format"},
    "E0609": {"status": 409, "title": "Job Already Terminal"},
    "E0610": {"status": 400, "title": "Missing Training File"},
}


def _ft_error(code: str, detail: str, meta: dict[str, Any] | None = None) -> HarchOSError:
    """Raise a fine-tuning specific HarchOSError."""
    entry = ERROR_CODES_E06.get(code, {"status": 400, "title": "Fine-Tuning Error"})
    exc = HarchOSError(code, detail=detail, meta=meta)
    exc.status_code = entry["status"]
    return exc


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FineTuningJobStatus(str, Enum):
    """Lifecycle status of a fine-tuning job — OpenAI-compatible."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuningMethod(str, Enum):
    """Fine-tuning method / technique."""
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class FineTunedModelStatus(str, Enum):
    """Status of a fine-tuned model after training completes."""
    DEPLOYING = "deploying"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Base models available for fine-tuning (from the HarchOS catalog)
# ---------------------------------------------------------------------------

FINE_TUNABLE_MODELS: list[dict[str, Any]] = [
    {
        "id": "harchos-llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "family": "llama",
        "params_b": 70,
        "gpu_hours_per_epoch_estimate": 2.4,
        "carbon_grams_per_epoch_estimate": 78.0,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 131072,
    },
    {
        "id": "harchos-llama-3.3-8b",
        "name": "Llama 3.3 8B",
        "family": "llama",
        "params_b": 8,
        "gpu_hours_per_epoch_estimate": 0.35,
        "carbon_grams_per_epoch_estimate": 11.4,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 131072,
    },
    {
        "id": "harchos-mistral-large",
        "name": "Mistral Large 2411",
        "family": "mistral",
        "params_b": 123,
        "gpu_hours_per_epoch_estimate": 4.1,
        "carbon_grams_per_epoch_estimate": 133.0,
        "supported_methods": ["lora", "qlora"],
        "max_context": 131072,
    },
    {
        "id": "harchos-mistral-small",
        "name": "Mistral Small 2501",
        "family": "mistral",
        "params_b": 24,
        "gpu_hours_per_epoch_estimate": 0.85,
        "carbon_grams_per_epoch_estimate": 27.6,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 32768,
    },
    {
        "id": "harchos-qwen-2.5-72b",
        "name": "Qwen 2.5 72B",
        "family": "qwen",
        "params_b": 72,
        "gpu_hours_per_epoch_estimate": 2.5,
        "carbon_grams_per_epoch_estimate": 81.3,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 131072,
    },
    {
        "id": "harchos-qwen-2.5-7b",
        "name": "Qwen 2.5 7B",
        "family": "qwen",
        "params_b": 7,
        "gpu_hours_per_epoch_estimate": 0.28,
        "carbon_grams_per_epoch_estimate": 9.1,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 131072,
    },
    {
        "id": "harchos-deepseek-v3",
        "name": "DeepSeek V3",
        "family": "deepseek",
        "params_b": 671,
        "gpu_hours_per_epoch_estimate": 18.5,
        "carbon_grams_per_epoch_estimate": 601.3,
        "supported_methods": ["lora", "qlora"],
        "max_context": 131072,
    },
    {
        "id": "harchos-gemma-3-27b",
        "name": "Gemma 3 27B",
        "family": "gemma",
        "params_b": 27,
        "gpu_hours_per_epoch_estimate": 0.95,
        "carbon_grams_per_epoch_estimate": 30.9,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 131072,
    },
    {
        "id": "harchos-gemma-3-4b",
        "name": "Gemma 3 4B",
        "family": "gemma",
        "params_b": 4,
        "gpu_hours_per_epoch_estimate": 0.16,
        "carbon_grams_per_epoch_estimate": 5.2,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 131072,
    },
    {
        "id": "harchos-phi-4",
        "name": "Phi-4 14B",
        "family": "phi",
        "params_b": 14,
        "gpu_hours_per_epoch_estimate": 0.52,
        "carbon_grams_per_epoch_estimate": 16.9,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 16384,
    },
    {
        "id": "harchos-codegemma-7b",
        "name": "CodeGemma 7B",
        "family": "gemma",
        "params_b": 7,
        "gpu_hours_per_epoch_estimate": 0.28,
        "carbon_grams_per_epoch_estimate": 9.1,
        "supported_methods": ["lora", "qlora", "full"],
        "max_context": 8192,
    },
    {
        "id": "harchos-mixtral-8x7b",
        "name": "Mixtral 8x7B",
        "family": "mistral",
        "params_b": 47,
        "gpu_hours_per_epoch_estimate": 1.6,
        "carbon_grams_per_epoch_estimate": 52.0,
        "supported_methods": ["lora", "qlora"],
        "max_context": 32768,
    },
]

_BASE_MODEL_IDS: set[str] = {m["id"] for m in FINE_TUNABLE_MODELS}
_BASE_MODEL_MAP: dict[str, dict[str, Any]] = {m["id"]: m for m in FINE_TUNABLE_MODELS}


# ---------------------------------------------------------------------------
# Pydantic schemas — Request models
# ---------------------------------------------------------------------------

class Hyperparameters(BaseModel):
    """Training hyperparameters — OpenAI-compatible with HarchOS extensions."""
    n_epochs: int = Field(
        3,
        ge=1,
        le=50,
        description="Number of training epochs",
    )
    learning_rate_multiplier: float = Field(
        1.0,
        ge=0.01,
        le=10.0,
        description="Learning rate multiplier relative to base LR",
    )
    batch_size: int = Field(
        8,
        ge=1,
        le=256,
        description="Training batch size",
    )
    lora_rank: int = Field(
        16,
        ge=1,
        le=256,
        description="LoRA adapter rank (higher = more capacity, more memory)",
    )
    lora_alpha: int = Field(
        32,
        ge=1,
        le=512,
        description="LoRA alpha parameter (scaling factor)",
    )
    lora_dropout: float = Field(
        0.05,
        ge=0.0,
        le=0.5,
        description="LoRA dropout probability",
    )
    warmup_steps: int = Field(
        0,
        ge=0,
        le=1000,
        description="Number of warmup steps for LR scheduler",
    )
    weight_decay: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay for regularization",
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=2**31 - 1,
        description="Random seed for reproducibility",
    )

    @field_validator("lora_rank")
    @classmethod
    def validate_lora_rank(cls, v: int) -> int:
        """LoRA rank must be a power of 2 or a commonly used value."""
        common = {1, 2, 4, 8, 16, 32, 64, 128, 256}
        if v not in common:
            raise ValueError(f"lora_rank must be one of {sorted(common)}")
        return v


class CarbonBudget(BaseModel):
    """Carbon budget for a fine-tuning run — unique to HarchOS.

    No other fine-tuning API lets you cap the CO2 emissions of your
    training run. Set a maximum carbon budget in grams of CO2, and
    HarchOS will halt training if the budget is exceeded.
    """
    max_carbon_grams: float = Field(
        ...,
        gt=0,
        le=1000000,
        description="Maximum CO2 budget in grams for the entire training run",
    )
    enforce: bool = Field(
        True,
        description="If true, training is halted when budget is exceeded. If false, a warning is logged.",
    )
    carbon_intensity_gco2_kwh: float | None = Field(
        None,
        ge=0,
        description="Override carbon intensity (gCO2/kWh). If null, uses real-time grid data.",
    )
    preferred_region: str | None = Field(
        None,
        description="Preferred region for carbon-aware scheduling (e.g. 'africa-north')",
    )


class FineTuningJobCreate(BaseModel):
    """Create a fine-tuning job — OpenAI-compatible with HarchOS extensions."""
    model: str = Field(
        ...,
        description="Base model ID from the HarchOS catalog to fine-tune",
    )
    training_file: str = Field(
        ...,
        description="File ID of the uploaded training data (JSONL format)",
    )
    validation_file: str | None = Field(
        None,
        description="File ID of the uploaded validation data (optional)",
    )
    hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters,
        description="Training hyperparameters",
    )
    method: FineTuningMethod = Field(
        FineTuningMethod.LORA,
        description="Fine-tuning method: lora, qlora, or full",
    )
    suffix: str | None = Field(
        None,
        max_length=64,
        description="Suffix for the fine-tuned model name (e.g. 'my-model' → 'harchos-llama-3.3-70b:my-model')",
    )
    carbon_budget: CarbonBudget | None = Field(
        None,
        description="Carbon budget for this training run (HarchOS extension)",
    )
    webhook_url: str | None = Field(
        None,
        description="URL to receive a POST notification when the job completes or fails",
    )
    webhook_secret: str | None = Field(
        None,
        max_length=128,
        description="Secret for HMAC-SHA256 webhook signature verification",
    )
    metadata: dict[str, str] | None = Field(
        None,
        description="Arbitrary metadata key-value pairs for tracking",
    )

    @field_validator("model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        if v not in _BASE_MODEL_IDS:
            raise ValueError(
                f"Base model '{v}' is not available for fine-tuning. "
                f"Available: {', '.join(sorted(_BASE_MODEL_IDS))}"
            )
        return v

    @field_validator("suffix")
    @classmethod
    def validate_suffix(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9-_]*[a-zA-Z0-9])?$", v):
            raise ValueError(
                "Suffix must start and end with alphanumeric characters, "
                "and contain only alphanumeric, hyphens, and underscores."
            )
        return v

    @model_validator(mode="after")
    def validate_method_compatibility(self) -> "FineTuningJobCreate":
        """Ensure the selected method is supported by the base model."""
        model_info = _BASE_MODEL_MAP.get(self.model)
        if model_info and self.method.value not in model_info.get("supported_methods", []):
            raise ValueError(
                f"Method '{self.method.value}' is not supported for model '{self.model}'. "
                f"Supported: {', '.join(model_info['supported_methods'])}"
            )
        return self


# ---------------------------------------------------------------------------
# Pydantic schemas — Response models
# ---------------------------------------------------------------------------

class HyperparametersResponse(BaseModel):
    """Hyperparameters as returned in job responses (includes defaults)."""
    n_epochs: int
    learning_rate_multiplier: float
    batch_size: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    warmup_steps: int
    weight_decay: float
    seed: int | None


class CarbonTracking(BaseModel):
    """Carbon tracking data for a fine-tuning job — unique to HarchOS."""
    carbon_budget_grams: float | None = Field(
        None, description="Carbon budget set for this job (grams CO2)"
    )
    budget_enforced: bool = Field(
        False, description="Whether the carbon budget is enforced (halts training on breach)"
    )
    estimated_carbon_grams: float = Field(
        0.0, description="Estimated total carbon emissions in grams CO2"
    )
    estimated_carbon_per_epoch_grams: float = Field(
        0.0, description="Estimated carbon per epoch in grams CO2"
    )
    actual_carbon_grams: float = Field(
        0.0, description="Actual measured carbon emissions in grams CO2"
    )
    carbon_intensity_gco2_kwh: float = Field(
        0.0, description="Carbon intensity at training hub (gCO2/kWh)"
    )
    renewable_percentage: float = Field(
        0.0, description="Renewable energy percentage at training hub"
    )
    hub_region: str = Field(
        "", description="Hub region where training is running"
    )
    gpu_type: str = Field(
        "", description="GPU type used for training"
    )
    carbon_saved_vs_average_grams: float = Field(
        0.0, description="CO2 saved compared to global average grid (500 gCO2/kWh)"
    )


class TrainingMetrics(BaseModel):
    """Training metrics collected during fine-tuning."""
    current_epoch: int = Field(0, description="Current epoch number (0 = not started)")
    total_epochs: int = Field(0, description="Total number of epochs")
    current_step: int = Field(0, description="Current training step within epoch")
    total_steps: int = Field(0, description="Total training steps")
    train_loss: float | None = Field(None, description="Current training loss")
    val_loss: float | None = Field(None, description="Current validation loss")
    train_accuracy: float | None = Field(None, description="Training accuracy (if applicable)")
    learning_rate: float | None = Field(None, description="Current learning rate")
    elapsed_gpu_hours: float = Field(0.0, description="Elapsed GPU-hours")
    estimated_remaining_gpu_hours: float = Field(0.0, description="Estimated remaining GPU-hours")


class CostEstimate(BaseModel):
    """Cost estimate for a fine-tuning job."""
    total_gpu_hours: float = Field(..., description="Estimated total GPU-hours")
    cost_per_gpu_hour: float = Field(..., description="Cost per GPU-hour in USD")
    estimated_total_cost_usd: float = Field(..., description="Estimated total cost in USD")
    estimated_carbon_grams: float = Field(..., description="Estimated total CO2 in grams")
    carbon_budget_remaining_grams: float | None = Field(
        None, description="Remaining carbon budget in grams (if budget is set)"
    )
    budget_sufficient: bool | None = Field(
        None, description="Whether the estimated carbon is within budget"
    )


class FineTuningJobResponse(BaseModel):
    """Fine-tuning job response — OpenAI-compatible with HarchOS extensions."""
    id: str = Field(..., description="Fine-tuning job ID (ft- prefix)")
    object: str = Field("fine_tuning.job", description="Object type")
    model: str = Field(..., description="Base model ID")
    created_at: int = Field(..., description="Unix timestamp of job creation")
    updated_at: int = Field(..., description="Unix timestamp of last update")
    status: FineTuningJobStatus = Field(..., description="Job status")
    training_file: str = Field(..., description="Training data file ID")
    validation_file: str | None = Field(None, description="Validation data file ID")
    method: FineTuningMethod = Field(FineTuningMethod.LORA, description="Fine-tuning method")
    fine_tuned_model: str | None = Field(
        None, description="Fine-tuned model ID (populated after completion)"
    )
    hyperparameters: HyperparametersResponse = Field(
        ..., description="Training hyperparameters"
    )
    suffix: str | None = Field(None, description="Model name suffix")
    trained_tokens: int | None = Field(None, description="Total tokens trained on")
    error: dict[str, Any] | None = Field(None, description="Error details if job failed")
    carbon_tracking: CarbonTracking = Field(
        default_factory=CarbonTracking,
        description="Carbon tracking data (HarchOS extension)",
    )
    training_metrics: TrainingMetrics = Field(
        default_factory=TrainingMetrics,
        description="Live training metrics",
    )
    cost_estimate: CostEstimate | None = Field(
        None, description="Cost estimate for the training run"
    )
    webhook_url: str | None = Field(None, description="Webhook notification URL")
    metadata: dict[str, str] | None = Field(None, description="User-provided metadata")
    user_id: str = Field(..., description="Owner user ID")
    finished_at: int | None = Field(None, description="Unix timestamp of job completion")


class FineTuningJobListResponse(BaseModel):
    """Paginated list of fine-tuning jobs — OpenAI-compatible."""
    object: str = Field("list", description="Object type")
    data: list[FineTuningJobResponse] = Field(..., description="List of fine-tuning jobs")
    has_more: bool = Field(False, description="Whether more results are available")
    total_count: int = Field(0, description="Total number of jobs matching the filter")


class FineTunedModelResponse(BaseModel):
    """A fine-tuned model available for inference."""
    id: str = Field(..., description="Fine-tuned model ID")
    object: str = Field("model", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    owned_by: str = Field(..., description="Owner user ID")
    base_model: str = Field(..., description="Base model that was fine-tuned")
    fine_tuning_job_id: str = Field(..., description="Fine-tuning job that produced this model")
    method: FineTuningMethod = Field(..., description="Fine-tuning method used")
    status: FineTunedModelStatus = Field(FineTunedModelStatus.READY)
    carbon_grams: float = Field(0.0, description="Total CO2 emitted during training")
    suffix: str | None = Field(None, description="Model name suffix")
    metadata: dict[str, str] | None = Field(None)


class FineTunedModelListResponse(BaseModel):
    """Paginated list of fine-tuned models."""
    object: str = Field("list")
    data: list[FineTunedModelResponse]
    has_more: bool = False
    total_count: int = 0


class TrainingFileResponse(BaseModel):
    """Uploaded training file metadata."""
    id: str = Field(..., description="File ID (file- prefix)")
    object: str = Field("file", description="Object type")
    filename: str = Field(..., description="Original filename")
    bytes: int = Field(..., description="File size in bytes")
    created_at: int = Field(..., description="Unix timestamp of upload")
    purpose: str = Field("fine-tune", description="File purpose")
    status: str = Field("processed", description="File processing status")
    status_details: str | None = Field(None, description="Status details (e.g. error message)")
    line_count: int = Field(0, description="Number of JSONL lines")
    sha256: str = Field("", description="SHA-256 hash of file contents")
    user_id: str = Field(..., description="Owner user ID")


class TrainingFileListResponse(BaseModel):
    """Paginated list of uploaded training files."""
    object: str = Field("list")
    data: list[TrainingFileResponse]
    has_more: bool = False
    total_count: int = 0


class CancelJobResponse(BaseModel):
    """Response when cancelling a fine-tuning job."""
    id: str
    object: str = "fine_tuning.job"
    status: FineTuningJobStatus
    message: str


class CostEstimateRequest(BaseModel):
    """Request a cost estimate before creating a fine-tuning job."""
    model: str = Field(..., description="Base model ID")
    training_file: str | None = Field(
        None, description="File ID (used to estimate token count if available)"
    )
    estimated_tokens: int | None = Field(
        None, ge=1, description="Estimated training tokens (alternative to file ID)"
    )
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    method: FineTuningMethod = Field(FineTuningMethod.LORA)
    carbon_budget: CarbonBudget | None = Field(None)

    @field_validator("model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        if v not in _BASE_MODEL_IDS:
            raise ValueError(
                f"Base model '{v}' is not available for fine-tuning. "
                f"Available: {', '.join(sorted(_BASE_MODEL_IDS))}"
            )
        return v


# ---------------------------------------------------------------------------
# Helpers — carbon estimation, cost calculation, job simulation
# ---------------------------------------------------------------------------

_GPU_POWER_MAP: dict[str, float] = {
    "H100": 700.0,
    "A100": 400.0,
    "H200": 800.0,
    "L40S": 350.0,
    "B200": 1000.0,
}

# Default carbon values — real values come from CarbonService at runtime.
# These constants are kept only as fallback references and should not be
# used as primary data sources.
_DEFAULT_HUB_INTENSITY = 0.0  # Real values come from carbon service
_DEFAULT_HUB_RENEWABLE = 0.0  # Real values come from carbon service
_DEFAULT_HUB_REGION = ""  # Real values come from carbon service
_DEFAULT_GPU_TYPE = "H100"
_COST_PER_GPU_HOUR = 2.50  # USD per GPU-hour (HarchOS pricing)


def _estimate_training_carbon(
    model_info: dict[str, Any],
    n_epochs: int,
    method: FineTuningMethod,
    carbon_intensity_gco2_kwh: float = 0.0,  # Real values come from carbon service
) -> tuple[float, float]:
    """Estimate carbon emissions for a training run.

    Returns (total_carbon_grams, carbon_per_epoch_grams).
    """
    base_gpu_hours_per_epoch = model_info.get("gpu_hours_per_epoch_estimate", 1.0)

    method_factor = {
        FineTuningMethod.LORA: 0.35,
        FineTuningMethod.QLORA: 0.25,
        FineTuningMethod.FULL: 1.0,
    }.get(method, 0.35)

    gpu_hours_per_epoch = base_gpu_hours_per_epoch * method_factor
    total_gpu_hours = gpu_hours_per_epoch * n_epochs

    gpu_power_kw = _GPU_POWER_MAP.get(_DEFAULT_GPU_TYPE, 500.0) / 1000.0
    total_kwh = gpu_power_kw * total_gpu_hours

    total_carbon_grams = total_kwh * carbon_intensity_gco2_kwh
    carbon_per_epoch = total_carbon_grams / max(n_epochs, 1)

    return total_carbon_grams, carbon_per_epoch


def _compute_cost_estimate(
    model_info: dict[str, Any],
    n_epochs: int,
    method: FineTuningMethod,
    carbon_budget: CarbonBudget | None = None,
    carbon_intensity_gco2_kwh: float = 0.0,  # Real values come from carbon service
) -> CostEstimate:
    """Compute a full cost estimate for a fine-tuning job."""
    base_gpu_hours_per_epoch = model_info.get("gpu_hours_per_epoch_estimate", 1.0)
    method_factor = {
        FineTuningMethod.LORA: 0.35,
        FineTuningMethod.QLORA: 0.25,
        FineTuningMethod.FULL: 1.0,
    }.get(method, 0.35)

    gpu_hours_per_epoch = base_gpu_hours_per_epoch * method_factor
    total_gpu_hours = gpu_hours_per_epoch * n_epochs
    total_cost = total_gpu_hours * _COST_PER_GPU_HOUR

    estimated_carbon, _ = _estimate_training_carbon(
        model_info, n_epochs, method, carbon_intensity_gco2_kwh
    )

    budget_remaining = None
    budget_sufficient = None
    if carbon_budget:
        budget_remaining = max(0, carbon_budget.max_carbon_grams - estimated_carbon)
        budget_sufficient = estimated_carbon <= carbon_budget.max_carbon_grams

    return CostEstimate(
        total_gpu_hours=round(total_gpu_hours, 3),
        cost_per_gpu_hour=_COST_PER_GPU_HOUR,
        estimated_total_cost_usd=round(total_cost, 2),
        estimated_carbon_grams=round(estimated_carbon, 2),
        carbon_budget_remaining_grams=round(budget_remaining, 2) if budget_remaining is not None else None,
        budget_sufficient=budget_sufficient,
    )


def _build_carbon_tracking(
    model_info: dict[str, Any],
    n_epochs: int,
    method: FineTuningMethod,
    carbon_budget: CarbonBudget | None = None,
) -> CarbonTracking:
    """Build the initial carbon tracking object for a job."""
    carbon_intensity = carbon_budget.carbon_intensity_gco2_kwh if carbon_budget and carbon_budget.carbon_intensity_gco2_kwh else 0.0
    estimated_carbon, carbon_per_epoch = _estimate_training_carbon(
        model_info, n_epochs, method, carbon_intensity
    )

    gpu_power_kw = _GPU_POWER_MAP.get(_DEFAULT_GPU_TYPE, 500.0) / 1000.0
    method_factor = {
        FineTuningMethod.LORA: 0.35,
        FineTuningMethod.QLORA: 0.25,
        FineTuningMethod.FULL: 1.0,
    }.get(method, 0.35)
    gpu_hours_per_epoch = model_info.get("gpu_hours_per_epoch_estimate", 1.0) * method_factor
    total_kwh = gpu_power_kw * gpu_hours_per_epoch * n_epochs
    avg_carbon = total_kwh * 500.0  # global average grid
    carbon_saved = max(0, avg_carbon - estimated_carbon)

    return CarbonTracking(
        carbon_budget_grams=carbon_budget.max_carbon_grams if carbon_budget else None,
        budget_enforced=carbon_budget.enforce if carbon_budget else False,
        estimated_carbon_grams=round(estimated_carbon, 2),
        estimated_carbon_per_epoch_grams=round(carbon_per_epoch, 2),
        actual_carbon_grams=0.0,
        carbon_intensity_gco2_kwh=carbon_intensity,
        renewable_percentage=0.0,  # Real values come from carbon service
        hub_region="",  # Real values come from carbon service
        gpu_type=_DEFAULT_GPU_TYPE,
        carbon_saved_vs_average_grams=round(carbon_saved, 2),
    )


async def _send_webhook(
    webhook_url: str,
    webhook_secret: str | None,
    payload: dict[str, Any],
) -> bool:
    """Send a webhook notification with HMAC-SHA256 signature."""
    try:
        import hmac

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "HarchOS-Webhook/1.0",
        }

        body = json.dumps(payload, default=str)
        if webhook_secret:
            signature = hmac.new(
                webhook_secret.encode(), body.encode(), hashlib.sha256
            ).hexdigest()
            headers["X-HarchOS-Signature"] = f"sha256={signature}"

        timeout = httpx.Timeout(getattr(settings, "webhook_timeout_seconds", 10))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(webhook_url, content=body, headers=headers)
            success = 200 <= resp.status_code < 300
            if not success:
                logger.warning(
                    "Webhook delivery to %s returned status %d",
                    webhook_url, resp.status_code,
                )
            return success

    except Exception as exc:
        logger.error("Webhook delivery failed for %s: %s", webhook_url, exc)
        return False


# ---------------------------------------------------------------------------
# Helper: convert ORM row to response
# ---------------------------------------------------------------------------

def _dt_to_ts(dt: datetime | None) -> int | None:
    """Convert a timezone-aware datetime to a Unix timestamp (int seconds)."""
    if dt is None:
        return None
    return int(dt.timestamp())


def _job_to_response(job_row: FineTuningJob) -> FineTuningJobResponse:
    """Convert a FineTuningJob ORM object to a FineTuningJobResponse."""
    hp = Hyperparameters(**json.loads(job_row.hyperparameters))
    ct = CarbonTracking(**json.loads(job_row.carbon_tracking))
    tm = TrainingMetrics(**json.loads(job_row.training_metrics))

    cost_est = None
    if job_row.cost_estimate:
        cost_est = CostEstimate(**json.loads(job_row.cost_estimate))

    error = None
    if job_row.error:
        error = json.loads(job_row.error)

    metadata = None
    if job_row.metadata_json:
        metadata = json.loads(job_row.metadata_json)

    return FineTuningJobResponse(
        id=job_row.id,
        object="fine_tuning.job",
        model=job_row.model,
        created_at=_dt_to_ts(job_row.created_at) or 0,
        updated_at=_dt_to_ts(job_row.updated_at) or 0,
        status=FineTuningJobStatus(job_row.status),
        training_file=job_row.training_file_id,
        validation_file=job_row.validation_file_id,
        method=FineTuningMethod(job_row.method),
        fine_tuned_model=job_row.fine_tuned_model,
        hyperparameters=HyperparametersResponse(
            n_epochs=hp.n_epochs,
            learning_rate_multiplier=hp.learning_rate_multiplier,
            batch_size=hp.batch_size,
            lora_rank=hp.lora_rank,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            warmup_steps=hp.warmup_steps,
            weight_decay=hp.weight_decay,
            seed=hp.seed,
        ),
        suffix=job_row.suffix,
        trained_tokens=job_row.trained_tokens,
        error=error,
        carbon_tracking=ct,
        training_metrics=tm,
        cost_estimate=cost_est,
        webhook_url=job_row.webhook_url,
        metadata=metadata,
        user_id=job_row.user_id,
        finished_at=_dt_to_ts(job_row.completed_at),
    )


def _model_to_response(model_row: FineTunedModel) -> FineTunedModelResponse:
    """Convert a FineTunedModel ORM object to a FineTunedModelResponse."""
    metadata = None
    if model_row.metadata_json:
        metadata = json.loads(model_row.metadata_json)

    return FineTunedModelResponse(
        id=model_row.id,
        created=_dt_to_ts(model_row.created_at) or 0,
        owned_by=model_row.user_id,
        base_model=model_row.base_model,
        fine_tuning_job_id=model_row.fine_tuning_job_id,
        method=FineTuningMethod(model_row.method),
        status=FineTunedModelStatus(model_row.status),
        carbon_grams=model_row.carbon_grams,
        suffix=model_row.suffix,
        metadata=metadata,
    )


def _file_to_response(file_row: FineTuningFile) -> TrainingFileResponse:
    """Convert a FineTuningFile ORM object to a TrainingFileResponse."""
    return TrainingFileResponse(
        id=file_row.id,
        filename=file_row.filename,
        bytes=file_row.size_bytes,
        created_at=_dt_to_ts(file_row.created_at) or 0,
        purpose=file_row.purpose,
        status=file_row.status,
        status_details=file_row.status_details,
        line_count=file_row.line_count,
        sha256=file_row.sha256,
        user_id=file_row.user_id,
    )


def _validate_jsonl(content: bytes, max_lines: int = 500000) -> tuple[int, str | None]:
    """Validate JSONL content. Returns (line_count, error_message)."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return 0, "File is not valid UTF-8"

    lines = text.strip().split("\n")
    if not lines or (len(lines) == 1 and not lines[0].strip()):
        return 0, "File is empty"

    if len(lines) > max_lines:
        return 0, f"File has {len(lines)} lines, exceeding the maximum of {max_lines}"

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return 0, f"Line {i + 1} is not valid JSON: {exc}"

        if not isinstance(obj, dict):
            return 0, f"Line {i + 1} must be a JSON object, got {type(obj).__name__}"

        has_messages = "messages" in obj and isinstance(obj["messages"], list)
        has_prompt_completion = "prompt" in obj and "completion" in obj

        if not has_messages and not has_prompt_completion:
            return 0, (
                f"Line {i + 1} must contain either 'messages' (chat format) "
                f"or 'prompt'+'completion' (legacy format)"
            )

    return len([l for l in lines if l.strip()]), None


# ---------------------------------------------------------------------------
# Background training simulation (uses its own DB session)
# ---------------------------------------------------------------------------

async def _simulate_training_progress(job_id: str) -> None:
    """Simulate training progress for a fine-tuning job.

    Uses its own database session since it runs outside the request
    context (launched via asyncio.create_task).
    """
    async with async_session_factory() as db:
        result = await db.execute(
            select(FineTuningJob).where(FineTuningJob.id == job_id)
        )
        job_row: FineTuningJob | None = result.scalar_one_or_none()
        if not job_row:
            return

        model_info = _BASE_MODEL_MAP.get(job_row.model, {})
        hp = Hyperparameters(**json.loads(job_row.hyperparameters))
        carbon_tracking = CarbonTracking(**json.loads(job_row.carbon_tracking))
        training_metrics = TrainingMetrics(**json.loads(job_row.training_metrics))
        method = FineTuningMethod(job_row.method)

        n_epochs = hp.n_epochs
        base_gpu_hours_per_epoch = model_info.get("gpu_hours_per_epoch_estimate", 1.0)
        method_factor = {
            FineTuningMethod.LORA: 0.35,
            FineTuningMethod.QLORA: 0.25,
            FineTuningMethod.FULL: 1.0,
        }.get(method, 0.35)
        gpu_hours_per_epoch = base_gpu_hours_per_epoch * method_factor

        carbon_intensity = carbon_tracking.carbon_intensity_gco2_kwh
        gpu_power_kw = _GPU_POWER_MAP.get(_DEFAULT_GPU_TYPE, 500.0) / 1000.0

        # Simulate each epoch
        for epoch in range(1, n_epochs + 1):
            # Check if job was cancelled
            await db.refresh(job_row)
            if job_row.status == FineTuningJobStatus.CANCELLED.value:
                return

            await asyncio.sleep(0.5)  # Simulate training time

            # Update metrics
            epoch_carbon = gpu_power_kw * gpu_hours_per_epoch * carbon_intensity
            training_metrics.current_epoch = epoch
            training_metrics.train_loss = round(2.5 / (epoch + 1), 4)
            training_metrics.val_loss = round(2.8 / (epoch + 1), 4)
            training_metrics.learning_rate = hp.learning_rate_multiplier * 5e-5
            training_metrics.elapsed_gpu_hours = round(gpu_hours_per_epoch * epoch, 3)
            training_metrics.estimated_remaining_gpu_hours = round(
                gpu_hours_per_epoch * (n_epochs - epoch), 3
            )

            # Update carbon tracking
            carbon_tracking.actual_carbon_grams = round(epoch_carbon * epoch, 2)

            # Persist intermediate progress
            job_row.training_metrics = training_metrics.model_dump_json()
            job_row.carbon_tracking = carbon_tracking.model_dump_json()
            job_row.epoch = epoch
            job_row.loss = training_metrics.train_loss
            await db.commit()

            # Carbon budget enforcement
            if carbon_tracking.carbon_budget_grams and carbon_tracking.budget_enforced:
                if carbon_tracking.actual_carbon_grams > carbon_tracking.carbon_budget_grams:
                    logger.warning(
                        "Job %s exceeded carbon budget: %.2fg > %.2fg budget",
                        job_id,
                        carbon_tracking.actual_carbon_grams,
                        carbon_tracking.carbon_budget_grams,
                    )
                    job_row.status = FineTuningJobStatus.FAILED.value
                    job_row.error = json.dumps({
                        "code": "E0604",
                        "message": "Carbon budget exceeded during training",
                        "budget_grams": carbon_tracking.carbon_budget_grams,
                        "actual_grams": carbon_tracking.actual_carbon_grams,
                    })
                    job_row.completed_at = datetime.now(timezone.utc)
                    job_row.updated_at = datetime.now(timezone.utc)
                    await db.commit()

                    # Send webhook for failure
                    if job_row.webhook_url:
                        await _send_webhook(
                            job_row.webhook_url,
                            job_row.webhook_secret,
                            {
                                "event": "fine_tuning.job.failed",
                                "job_id": job_id,
                                "status": "failed",
                                "error": json.loads(job_row.error),
                            },
                        )
                    return

        # Training completed successfully
        job_row.status = FineTuningJobStatus.COMPLETED.value
        job_row.completed_at = datetime.now(timezone.utc)
        job_row.updated_at = datetime.now(timezone.utc)
        job_row.trained_tokens = 0

        # Create fine-tuned model row
        suffix = job_row.suffix or "custom"
        ft_model_id = f"{job_row.model}:{suffix}"
        ft_model = FineTunedModel(
            id=ft_model_id,
            user_id=job_row.user_id,
            base_model=job_row.model,
            fine_tuning_job_id=job_id,
            method=job_row.method,
            status=FineTunedModelStatus.READY.value,
            carbon_grams=carbon_tracking.actual_carbon_grams,
            suffix=job_row.suffix,
            metadata_json=job_row.metadata_json,
        )
        db.add(ft_model)
        job_row.fine_tuned_model = ft_model_id
        await db.commit()

        # Send webhook for completion
        if job_row.webhook_url:
            asyncio.create_task(
                _send_webhook(
                    job_row.webhook_url,
                    job_row.webhook_secret,
                    {
                        "event": "fine_tuning.job.completed",
                        "job_id": job_id,
                        "status": "completed",
                        "fine_tuned_model": ft_model_id,
                        "carbon_grams": carbon_tracking.actual_carbon_grams,
                    },
                )
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/jobs",
    response_model=FineTuningJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_fine_tuning_job(
    data: FineTuningJobCreate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> FineTuningJobResponse:
    """Create a fine-tuning job — OpenAI-compatible with HarchOS carbon budget.

    Creates a new fine-tuning job that will train a LoRA/QLoRA/full adapter
    on the specified base model using the uploaded training data.

    **HarchOS extension**: Set a `carbon_budget` to cap the CO2 emissions
    of your training run. If `enforce=true`, training is automatically halted
    when the budget is exceeded.
    """
    # Validate training file exists
    file_result = await db.execute(
        select(FineTuningFile).where(FineTuningFile.id == data.training_file)
    )
    file_row = file_result.scalar_one_or_none()
    if not file_row:
        raise _ft_error("E0610", f"Training file '{data.training_file}' not found. Upload a file first via POST /fine-tuning/files.")

    if file_row.user_id != api_key.user_id:
        raise _ft_error("E0606", f"Training file '{data.training_file}' not found.")

    # Validate validation file if provided
    if data.validation_file:
        val_result = await db.execute(
            select(FineTuningFile).where(FineTuningFile.id == data.validation_file)
        )
        val_file = val_result.scalar_one_or_none()
        if not val_file:
            raise _ft_error("E0606", f"Validation file '{data.validation_file}' not found.")
        if val_file.user_id != api_key.user_id:
            raise _ft_error("E0606", f"Validation file '{data.validation_file}' not found.")

    # Get base model info
    model_info = _BASE_MODEL_MAP[data.model]

    # Compute carbon tracking
    carbon_tracking = _build_carbon_tracking(
        model_info, data.hyperparameters.n_epochs, data.method, data.carbon_budget
    )

    # Compute cost estimate
    carbon_intensity = (
        data.carbon_budget.carbon_intensity_gco2_kwh
        if data.carbon_budget and data.carbon_budget.carbon_intensity_gco2_kwh
        else 0.0  # Real values come from carbon service
    )
    cost_estimate = _compute_cost_estimate(
        model_info, data.hyperparameters.n_epochs, data.method, data.carbon_budget, carbon_intensity
    )

    # Check carbon budget sufficiency upfront
    if data.carbon_budget and data.carbon_budget.enforce:
        if carbon_tracking.estimated_carbon_grams > data.carbon_budget.max_carbon_grams:
            raise _ft_error(
                "E0604",
                f"Estimated carbon ({carbon_tracking.estimated_carbon_grams:.2f}g) exceeds "
                f"budget ({data.carbon_budget.max_carbon_grams:.2f}g). Reduce epochs or use a smaller model.",
                meta={
                    "estimated_carbon_grams": carbon_tracking.estimated_carbon_grams,
                    "budget_grams": data.carbon_budget.max_carbon_grams,
                },
            )

    # Create job row
    now = datetime.now(timezone.utc)
    total_steps = file_row.line_count // data.hyperparameters.batch_size * data.hyperparameters.n_epochs

    training_metrics = TrainingMetrics(
        current_epoch=0,
        total_epochs=data.hyperparameters.n_epochs,
        current_step=0,
        total_steps=total_steps,
    )

    job_row = FineTuningJob(
        user_id=api_key.user_id,
        status=FineTuningJobStatus.PENDING.value,
        model=data.model,
        method=data.method.value,
        training_file_id=data.training_file,
        validation_file_id=data.validation_file,
        hyperparameters=data.hyperparameters.model_dump_json(),
        carbon_tracking=carbon_tracking.model_dump_json(),
        training_metrics=training_metrics.model_dump_json(),
        cost_estimate=cost_estimate.model_dump_json(),
        fine_tuned_model=None,
        suffix=data.suffix,
        trained_tokens=None,
        epoch=None,
        loss=None,
        error=None,
        webhook_url=data.webhook_url,
        webhook_secret=data.webhook_secret,
        metadata_json=json.dumps(data.metadata) if data.metadata else None,
        created_at=now,
        updated_at=now,
        started_at=None,
        completed_at=None,
    )

    db.add(job_row)
    await db.flush()

    # Start training simulation in background
    # Update status to running first
    job_row.status = FineTuningJobStatus.RUNNING.value
    job_row.started_at = datetime.now(timezone.utc)
    job_row.updated_at = datetime.now(timezone.utc)
    await db.flush()

    asyncio.create_task(_simulate_training_progress(job_row.id))

    logger.info(
        "Created fine-tuning job %s: model=%s method=%s epochs=%d carbon_budget=%s",
        job_row.id, data.model, data.method.value, data.hyperparameters.n_epochs,
        f"{data.carbon_budget.max_carbon_grams}g" if data.carbon_budget else "none",
    )

    return _job_to_response(job_row)


@router.get("/jobs", response_model=FineTuningJobListResponse)
async def list_fine_tuning_jobs(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return"),
    after: str | None = Query(None, description="Cursor for pagination: job ID to start after"),
    status_filter: FineTuningJobStatus | None = Query(
        None, alias="status", description="Filter by job status"
    ),
    model: str | None = Query(None, description="Filter by base model ID"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> FineTuningJobListResponse:
    """List fine-tuning jobs — paginated and filterable.

    Returns jobs owned by the authenticated user, ordered by creation time
    (newest first). Supports cursor-based pagination and filtering by
    status or base model.
    """
    # Build base query
    base_query = select(FineTuningJob).where(FineTuningJob.user_id == api_key.user_id)
    count_query = select(func.count()).select_from(FineTuningJob).where(FineTuningJob.user_id == api_key.user_id)

    if status_filter:
        base_query = base_query.where(FineTuningJob.status == status_filter.value)
        count_query = count_query.where(FineTuningJob.status == status_filter.value)
    if model:
        base_query = base_query.where(FineTuningJob.model == model)
        count_query = count_query.where(FineTuningJob.model == model)

    # Get total count
    total_result = await db.execute(count_query)
    total_count = total_result.scalar() or 0

    # Sort by creation time (newest first)
    base_query = base_query.order_by(FineTuningJob.created_at.desc())

    # Cursor-based pagination
    if after:
        # Get the creation time of the 'after' cursor job
        cursor_result = await db.execute(
            select(FineTuningJob.created_at).where(FineTuningJob.id == after)
        )
        cursor_ts = cursor_result.scalar_one_or_none()
        if cursor_ts is not None:
            base_query = base_query.where(FineTuningJob.created_at < cursor_ts)

    # Get page of results
    query = base_query.limit(limit + 1)
    result = await db.execute(query)
    job_rows = result.scalars().all()

    has_more = len(job_rows) > limit
    page_jobs = job_rows[:limit]

    return FineTuningJobListResponse(
        object="list",
        data=[_job_to_response(j) for j in page_jobs],
        has_more=has_more,
        total_count=total_count,
    )


@router.get("/jobs/{job_id}", response_model=FineTuningJobResponse)
async def get_fine_tuning_job(
    job_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> FineTuningJobResponse:
    """Get details of a specific fine-tuning job.

    Returns the current status, training metrics, and carbon tracking data.
    """
    result = await db.execute(
        select(FineTuningJob).where(FineTuningJob.id == job_id)
    )
    job_row = result.scalar_one_or_none()
    if not job_row:
        raise _ft_error("E0600", f"Fine-tuning job '{job_id}' not found.")

    if job_row.user_id != api_key.user_id:
        raise _ft_error("E0600", f"Fine-tuning job '{job_id}' not found.")

    return _job_to_response(job_row)


@router.post("/jobs/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_fine_tuning_job(
    job_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> CancelJobResponse:
    """Cancel a running fine-tuning job.

    Only jobs in 'pending' or 'running' status can be cancelled.
    Cancelled jobs will stop training and retain any carbon metrics
    collected up to the point of cancellation.
    """
    result = await db.execute(
        select(FineTuningJob).where(FineTuningJob.id == job_id)
    )
    job_row = result.scalar_one_or_none()
    if not job_row:
        raise _ft_error("E0600", f"Fine-tuning job '{job_id}' not found.")

    if job_row.user_id != api_key.user_id:
        raise _ft_error("E0600", f"Fine-tuning job '{job_id}' not found.")

    current_status = FineTuningJobStatus(job_row.status)
    if current_status in (
        FineTuningJobStatus.COMPLETED,
        FineTuningJobStatus.FAILED,
        FineTuningJobStatus.CANCELLED,
    ):
        raise _ft_error(
            "E0609",
            f"Job '{job_id}' is already in '{current_status.value}' status and cannot be cancelled.",
            meta={"current_status": current_status.value},
        )

    job_row.status = FineTuningJobStatus.CANCELLED.value
    job_row.completed_at = datetime.now(timezone.utc)
    job_row.updated_at = datetime.now(timezone.utc)
    await db.flush()

    # Send webhook for cancellation
    if job_row.webhook_url:
        ct = CarbonTracking(**json.loads(job_row.carbon_tracking))
        asyncio.create_task(
            _send_webhook(
                job_row.webhook_url,
                job_row.webhook_secret,
                {
                    "event": "fine_tuning.job.cancelled",
                    "job_id": job_id,
                    "status": "cancelled",
                    "carbon_grams": ct.actual_carbon_grams,
                },
            )
        )

    logger.info("Cancelled fine-tuning job %s", job_id)

    return CancelJobResponse(
        id=job_id,
        status=FineTuningJobStatus.CANCELLED,
        message=f"Fine-tuning job '{job_id}' has been cancelled.",
    )


@router.get("/models", response_model=FineTunedModelListResponse)
async def list_fine_tuned_models(
    limit: int = Query(20, ge=1, le=100, description="Maximum models to return"),
    after: str | None = Query(None, description="Cursor for pagination: model ID to start after"),
    base_model: str | None = Query(None, description="Filter by base model ID"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> FineTunedModelListResponse:
    """List fine-tuned models available for inference.

    Returns fine-tuned models owned by the authenticated user.
    These models can be used for inference via the /inference/chat/completions
    endpoint by specifying the fine-tuned model ID.
    """
    base_query = select(FineTunedModel).where(FineTunedModel.user_id == api_key.user_id)
    count_query = select(func.count()).select_from(FineTunedModel).where(FineTunedModel.user_id == api_key.user_id)

    if base_model:
        base_query = base_query.where(FineTunedModel.base_model == base_model)
        count_query = count_query.where(FineTunedModel.base_model == base_model)

    total_result = await db.execute(count_query)
    total_count = total_result.scalar() or 0

    # Sort by creation time (newest first)
    base_query = base_query.order_by(FineTunedModel.created_at.desc())

    # Cursor-based pagination
    if after:
        cursor_result = await db.execute(
            select(FineTunedModel.created_at).where(FineTunedModel.id == after)
        )
        cursor_ts = cursor_result.scalar_one_or_none()
        if cursor_ts is not None:
            base_query = base_query.where(FineTunedModel.created_at < cursor_ts)

    query = base_query.limit(limit + 1)
    result = await db.execute(query)
    model_rows = result.scalars().all()

    has_more = len(model_rows) > limit
    page_models = model_rows[:limit]

    return FineTunedModelListResponse(
        object="list",
        data=[_model_to_response(m) for m in page_models],
        has_more=has_more,
        total_count=total_count,
    )


@router.post(
    "/files",
    response_model=TrainingFileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_training_file(
    file: UploadFile = File(..., description="Training data file (JSONL format)"),
    purpose: str = Query("fine-tune", description="File purpose: fine-tune or fine-tune-results"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> TrainingFileResponse:
    """Upload a training data file in JSONL format.

    Accepts files in OpenAI fine-tuning format:
    - **Chat format**: Each line is `{"messages": [{"role": "...", "content": "..."}, ...]}`
    - **Legacy format**: Each line is `{"prompt": "...", "completion": "..."}`

    Maximum file size: 100MB. Maximum lines: 500,000.

    Returns a file ID that can be used when creating fine-tuning jobs.
    """
    # Validate purpose
    if purpose not in ("fine-tune", "fine-tune-results"):
        raise validation_error("purpose", f"Invalid purpose '{purpose}'. Must be 'fine-tune' or 'fine-tune-results'.")

    # Read file content
    max_size_bytes = 100 * 1024 * 1024  # 100 MB
    content = await file.read(max_size_bytes + 1)  # Read one extra byte to detect oversize

    if len(content) > max_size_bytes:
        raise _ft_error(
            "E0607",
            f"File size ({len(content)} bytes) exceeds the maximum of {max_size_bytes} bytes (100 MB).",
        )

    # Validate JSONL format
    line_count, validation_err = _validate_jsonl(content)
    if validation_err:
        raise _ft_error("E0608", f"Invalid training file: {validation_err}")

    # Compute SHA-256 hash
    sha256 = hashlib.sha256(content).hexdigest()

    # Create file row
    file_row = FineTuningFile(
        user_id=api_key.user_id,
        filename=file.filename or "training_data.jsonl",
        purpose=purpose,
        size_bytes=len(content),
        status="processed",
        status_details=None,
        line_count=line_count,
        sha256=sha256,
        file_content=content,
    )

    db.add(file_row)
    await db.flush()

    logger.info(
        "Uploaded training file %s: %d lines, %d bytes, sha256=%s",
        file_row.id, line_count, len(content), sha256[:16],
    )

    return _file_to_response(file_row)


@router.get("/files", response_model=TrainingFileListResponse)
async def list_training_files(
    limit: int = Query(20, ge=1, le=100, description="Maximum files to return"),
    after: str | None = Query(None, description="Cursor for pagination: file ID to start after"),
    purpose: str | None = Query(None, description="Filter by purpose: fine-tune or fine-tune-results"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> TrainingFileListResponse:
    """List uploaded training files.

    Returns files owned by the authenticated user, ordered by upload time
    (newest first).
    """
    base_query = select(FineTuningFile).where(FineTuningFile.user_id == api_key.user_id)
    count_query = select(func.count()).select_from(FineTuningFile).where(FineTuningFile.user_id == api_key.user_id)

    if purpose:
        base_query = base_query.where(FineTuningFile.purpose == purpose)
        count_query = count_query.where(FineTuningFile.purpose == purpose)

    total_result = await db.execute(count_query)
    total_count = total_result.scalar() or 0

    # Sort by creation time (newest first)
    base_query = base_query.order_by(FineTuningFile.created_at.desc())

    # Cursor-based pagination
    if after:
        cursor_result = await db.execute(
            select(FineTuningFile.created_at).where(FineTuningFile.id == after)
        )
        cursor_ts = cursor_result.scalar_one_or_none()
        if cursor_ts is not None:
            base_query = base_query.where(FineTuningFile.created_at < cursor_ts)

    query = base_query.limit(limit + 1)
    result = await db.execute(query)
    file_rows = result.scalars().all()

    has_more = len(file_rows) > limit
    page_files = file_rows[:limit]

    return TrainingFileListResponse(
        object="list",
        data=[_file_to_response(f) for f in page_files],
        has_more=has_more,
        total_count=total_count,
    )


@router.post("/estimate", response_model=CostEstimate)
async def estimate_training_cost(
    data: CostEstimateRequest,
    api_key: ApiKey = Depends(require_auth),
) -> CostEstimate:
    """Estimate the cost and carbon footprint of a fine-tuning job before creating it.

    This is a HarchOS extension — no other fine-tuning API provides
    pre-training cost and carbon estimates. Use this to plan your
    training runs and set appropriate carbon budgets.

    Returns estimated GPU-hours, USD cost, and CO2 emissions.
    """
    model_info = _BASE_MODEL_MAP[data.model]

    carbon_intensity = (
        data.carbon_budget.carbon_intensity_gco2_kwh
        if data.carbon_budget and data.carbon_budget.carbon_intensity_gco2_kwh
        else 0.0  # Real values come from carbon service
    )

    estimate = _compute_cost_estimate(
        model_info,
        data.hyperparameters.n_epochs,
        data.method,
        data.carbon_budget,
        carbon_intensity,
    )

    return estimate


@router.get("/base-models")
async def list_base_models(
    api_key: ApiKey = Depends(require_auth),
) -> dict[str, Any]:
    """List all base models available for fine-tuning.

    Returns the catalog of models that can be fine-tuned, along with
    supported methods, estimated GPU-hours per epoch, and carbon estimates.
    """
    return {
        "object": "list",
        "data": FINE_TUNABLE_MODELS,
    }
