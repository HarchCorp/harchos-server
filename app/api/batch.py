"""Batch Inference API — Together AI-style batch processing with carbon tracking.

HarchOS batch inference allows submitting up to 100 inference requests in a
single API call. Batches are processed asynchronously at 50% reduced cost,
with full carbon footprint tracking per item and aggregate for the batch.

Features:
- Submit batch of up to 100 inference requests
- Track batch progress via polling or SSE streaming
- 50% cost reduction reflected in carbon_footprint (half the gCO2)
- Per-item request_id, status, and result tracking
- Aggregate carbon footprint for the entire batch
- Cancel pending batches
- Paginated batch listing per user
- Separate rate limiting: 5/min (free), 20/min (standard)
- Database-backed persistence (survives restarts, scales horizontally)
- OpenAI-compatible error format

Batch statuses: pending → processing → completed | failed | cancelled
Item statuses: pending → processing → completed | failed
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from app.api.deps import require_auth
from app.config import settings
from app.core.exceptions import HarchOSError, rate_limit_exceeded, not_found, validation_error
from app.database import get_db, async_session_factory
from app.models.api_key import ApiKey
from app.models.batch import BatchJob
from app.models.model import Model as DBModel

logger = logging.getLogger("harchos.batch")
router = APIRouter()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BatchStatus(str, Enum):
    """Lifecycle status of a batch job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchItemStatus(str, Enum):
    """Status of an individual item within a batch."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE = 100
BATCH_COST_DISCOUNT = 0.50  # 50% cost reduction for batch
BATCH_TTL_SECONDS = 86400 * 7  # Keep batches for 7 days
CLEANUP_INTERVAL_SECONDS = 3600  # Run cleanup every hour
BATCH_SUBMISSION_RATE_LIMITS = {
    "free": 5,       # 5 batch submissions per minute
    "standard": 20,  # 20 batch submissions per minute
    "enterprise": 100,
}

# Default model catalog for validation (used before DB is available).
# In production, models are loaded from the database via _get_valid_model_ids().
DEFAULT_VALID_MODEL_IDS = [
    "harchos-llama-3.3-70b",
    "harchos-llama-3.3-8b",
    "harchos-llama-4-maverick",
    "harchos-mistral-large",
    "harchos-mistral-small",
    "harchos-qwen-2.5-72b",
    "harchos-qwen-2.5-7b",
    "harchos-deepseek-v3",
    "harchos-deepseek-r1-70b",
    "harchos-gemma-3-27b",
    "harchos-gemma-3-4b",
    "harchos-phi-4",
    "harchos-codegemma-7b",
    "harchos-starcoder2-15b",
    "harchos-cohere-command-r",
    "harchos-cohere-command-r-plus",
    "harchos-mixtral-8x22b",
    "harchos-mixtral-8x7b",
    "harchos-yi-1.5-34b",
    "harchos-solar-10.7b",
    "harchos-llama-guard-4",
    "harchos-embedding-3-large",
]


# Cache for valid model IDs loaded from DB
_cached_model_ids: list[str] | None = None


async def _get_valid_model_ids(db: AsyncSession | None = None) -> list[str]:
    """Get valid model IDs, preferring DB data when available."""
    global _cached_model_ids

    if db is not None:
        try:
            result = await db.execute(select(DBModel).where(DBModel.status == "ready"))
            db_models = result.scalars().all()
            if db_models:
                model_ids = [
                    f"harchos-{m.name.lower().replace(' ', '-')}"
                    for m in db_models
                ]
                _cached_model_ids = model_ids
                return model_ids
        except Exception:
            pass

    if _cached_model_ids is not None:
        return _cached_model_ids

    return DEFAULT_VALID_MODEL_IDS


# ---------------------------------------------------------------------------
# In-memory rate limiter (short-lived timestamps — not worth persisting)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Thread-safe in-memory rate limiter for batch submissions.

    Rate limiting data is inherently ephemeral (sliding-window timestamps),
    so it stays in-memory even though batch data is persisted. This is
    acceptable because:
    - Rate limits reset on restart (a brief grace period is fine)
    - Rate limiting is per-instance anyway (each worker tracks its own)
    """

    def __init__(self) -> None:
        self._submission_counts: dict[str, list[float]] = {}
        self._last_cleanup: float = time.time()

    def check_submission_rate(self, key: str, tier: str = "standard") -> tuple[bool, int]:
        """Check if a batch submission is within rate limits.

        Returns (allowed, retry_after_seconds).
        """
        now = time.time()
        window = 60  # 1 minute
        limit = BATCH_SUBMISSION_RATE_LIMITS.get(tier, 5)

        timestamps = self._submission_counts.get(key, [])
        # Prune old timestamps
        timestamps = [t for t in timestamps if t > now - window]

        if len(timestamps) >= limit:
            oldest = min(timestamps) if timestamps else now
            retry_after = int(oldest + window - now) + 1
            return False, max(1, retry_after)

        timestamps.append(now)
        self._submission_counts[key] = timestamps

        # Periodic cleanup of stale entries
        if now - self._last_cleanup > CLEANUP_INTERVAL_SECONDS:
            self._last_cleanup = now
            cutoff = now - 120
            for k in list(self._submission_counts):
                self._submission_counts[k] = [
                    t for t in self._submission_counts[k] if t > cutoff
                ]
                if not self._submission_counts[k]:
                    del self._submission_counts[k]

        return True, 0


# Global rate limiter singleton
_rate_limiter = _RateLimiter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="system, user, or assistant")
    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("system", "user", "assistant"):
            raise ValueError("role must be 'system', 'user', or 'assistant'")
        return v


class BatchRequestItem(BaseModel):
    """A single inference request within a batch.

    Mirrors the OpenAI chat completion request format, with HarchOS
    carbon-aware extensions.
    """
    request_id: str = Field(
        ...,
        description="Unique identifier for this request within the batch",
        min_length=1,
        max_length=128,
    )
    model: str = Field(
        "harchos-llama-3.3-70b",
        description="Model ID to use for inference",
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Chat messages for completion",
    )
    temperature: float = Field(
        0.7,
        ge=0,
        le=2,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=32768,
        description="Maximum tokens to generate",
    )
    top_p: float = Field(1.0, ge=0, le=1, description="Top-p sampling")
    stop: list[str] | None = Field(None, description="Stop sequences")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        # Use default list for Pydantic validation (DB not available at parse time)
        # Runtime validation also checks against DB in the endpoint
        if v not in DEFAULT_VALID_MODEL_IDS:
            raise ValueError(
                f"Unknown model '{v}'. Available: {', '.join(DEFAULT_VALID_MODEL_IDS[:5])}... "
                f"({len(DEFAULT_VALID_MODEL_IDS)} total)"
            )
        return v


class BatchSubmissionRequest(BaseModel):
    """Request body for submitting a batch of inference requests."""
    requests: list[BatchRequestItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of inference requests (1–{MAX_BATCH_SIZE})",
    )
    metadata: dict[str, str] | None = Field(
        None,
        description="Optional user-provided metadata for the batch",
    )
    carbon_aware: bool = Field(
        True,
        description="Route to greenest hub for batch inference",
    )
    stream: bool = Field(
        False,
        description="Enable SSE streaming for batch progress updates",
    )

    @field_validator("requests")
    @classmethod
    def validate_request_ids_unique(cls, v: list[BatchRequestItem]) -> list[BatchRequestItem]:
        ids = [r.request_id for r in v]
        duplicates = [rid for rid in ids if ids.count(rid) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate request_id values found: {', '.join(set(duplicates))}. "
                "Each request_id must be unique within a batch."
            )
        return v

    @model_validator(mode="after")
    def validate_request_ids_pattern(self) -> "BatchSubmissionRequest":
        """Ensure request IDs match a safe character pattern."""
        safe_pattern = re.compile(r"^[a-zA-Z0-9_\-.:]+$")
        for req in self.requests:
            if not safe_pattern.match(req.request_id):
                raise ValueError(
                    f"request_id '{req.request_id}' contains invalid characters. "
                    "Only alphanumeric, underscore, hyphen, dot, and colon are allowed."
                )
        return self


class CarbonFootprint(BaseModel):
    """Carbon footprint for an inference request — unique to HarchOS."""
    gco2_per_request: float = Field(..., description="Grams of CO2 for this inference")
    hub_region: str = Field(..., description="Hub region used")
    carbon_intensity_gco2_kwh: float = Field(..., description="Carbon intensity at hub")
    renewable_percentage: float = Field(..., description="Renewable energy % at hub")
    gpu_type: str = Field(..., description="GPU type used")
    estimated_power_watts: float = Field(..., description="Estimated GPU power consumption")
    inference_duration_seconds: float = Field(..., description="Inference duration")
    carbon_saved_vs_average_gco2: float = Field(
        0.0,
        description="CO2 saved compared to global average grid (500 gCO2/kWh)",
    )
    batch_discount_applied: bool = Field(
        False,
        description="Whether the 50% batch cost discount was applied",
    )
    batch_discount_factor: float = Field(
        1.0,
        description="Discount factor applied (0.5 for batch)",
    )


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class BatchItemResult(BaseModel):
    """Result for a single item within a batch."""
    request_id: str
    status: BatchItemStatus
    model: str
    message: ChatMessage | None = None
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    carbon_footprint: CarbonFootprint | None = None
    error: dict[str, Any] | None = None
    completed_at: float | None = None


class AggregateCarbonFootprint(BaseModel):
    """Aggregate carbon footprint for an entire batch."""
    total_gco2: float = Field(..., description="Total gCO2 for all completed items")
    total_gco2_without_discount: float = Field(
        ..., description="What the gCO2 would have been without batch discount"
    )
    batch_savings_gco2: float = Field(
        ..., description="gCO2 saved thanks to 50% batch discount"
    )
    items_completed: int = Field(..., description="Number of completed items")
    items_failed: int = Field(..., description="Number of failed items")
    hub_region: str = Field("", description="Primary hub region used")
    average_carbon_intensity_gco2_kwh: float = Field(
        0.0, description="Average carbon intensity across items"
    )
    average_renewable_percentage: float = Field(
        0.0, description="Average renewable percentage across items"
    )


class BatchResponse(BaseModel):
    """Response for a batch submission."""
    id: str = Field(..., description="Unique batch identifier")
    object: str = Field("batch", description="Object type")
    status: BatchStatus
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    created_at: float
    updated_at: float
    metadata: dict[str, str] | None = None
    results: list[BatchItemResult] | None = None
    aggregate_carbon_footprint: AggregateCarbonFootprint | None = None
    estimated_completion_seconds: float | None = None


class BatchListResponse(BaseModel):
    """Paginated response for listing batches."""
    object: str = Field("list", description="Object type")
    data: list[BatchResponse]
    total: int
    has_more: bool


class BatchCancelResponse(BaseModel):
    """Response for cancelling a batch."""
    id: str
    object: str = Field("batch")
    status: BatchStatus
    cancelled_requests: int
    message: str


# ---------------------------------------------------------------------------
# Token estimation (matches inference.py logic)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count — 1 token ≈ 4 chars, CJK ≈ 2 tokens each."""
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
    other_chars = len(text) - cjk_chars
    return max(1, (other_chars // 4) + (cjk_chars * 2))


# ---------------------------------------------------------------------------
# Carbon estimation for batch items
# ---------------------------------------------------------------------------

def _estimate_batch_item_carbon(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    carbon_intensity_gco2_kwh: float = 0.0,  # Real values come from carbon service
    renewable_percentage: float = 0.0,  # Real values come from carbon service
    gpu_type: str = "H100",
    hub_region: str = "",  # Real values come from carbon service
) -> CarbonFootprint:
    """Estimate carbon footprint for a single batch inference item.

    Batch inference receives a 50% cost discount, reflected in the
    carbon footprint as half the gCO2 compared to real-time inference.
    This mirrors Together AI's batch pricing model, but applied to
    environmental cost rather than dollar cost.
    """
    gpu_power_w = {
        "H100": 700, "A100": 400, "L40S": 350, "H200": 800, "B200": 1000,
    }.get(gpu_type, 500)

    inference_seconds = (prompt_tokens / 5000) + (completion_tokens / 100)
    inference_seconds = max(0.5, inference_seconds)

    energy_kwh = (gpu_power_w * inference_seconds) / (1000 * 3600)

    # Full-cost carbon (before discount)
    full_gco2 = energy_kwh * carbon_intensity_gco2_kwh * 1000

    # Apply 50% batch discount
    discounted_gco2 = full_gco2 * BATCH_COST_DISCOUNT

    # CO2 saved vs average (also at discounted rate)
    avg_gco2 = energy_kwh * 500 * 1000 * BATCH_COST_DISCOUNT
    carbon_saved = max(0, avg_gco2 - discounted_gco2)

    return CarbonFootprint(
        gco2_per_request=round(discounted_gco2, 4),
        hub_region=hub_region,
        carbon_intensity_gco2_kwh=carbon_intensity_gco2_kwh,
        renewable_percentage=renewable_percentage,
        gpu_type=gpu_type,
        estimated_power_watts=gpu_power_w,
        inference_duration_seconds=round(inference_seconds, 3),
        carbon_saved_vs_average_gco2=round(carbon_saved, 4),
        batch_discount_applied=True,
        batch_discount_factor=BATCH_COST_DISCOUNT,
    )


def _compute_aggregate_carbon(
    results: list[dict[str, Any]],
    hub_region: str = "",  # Real values come from carbon service
) -> AggregateCarbonFootprint:
    """Compute aggregate carbon footprint across all batch items."""
    total_gco2 = 0.0
    total_gco2_without_discount = 0.0
    items_completed = 0
    items_failed = 0
    carbon_intensities: list[float] = []
    renewable_percentages: list[float] = []

    for item in results:
        if item["status"] == BatchItemStatus.COMPLETED.value:
            items_completed += 1
            cf = item.get("carbon_footprint")
            if cf:
                total_gco2 += cf.get("gco2_per_request", 0.0)
                # Reconstruct without discount: divide by discount factor
                raw_gco2 = cf.get("gco2_per_request", 0.0) / cf.get("batch_discount_factor", 0.5)
                total_gco2_without_discount += raw_gco2
                carbon_intensities.append(cf.get("carbon_intensity_gco2_kwh", 0.0))
                renewable_percentages.append(cf.get("renewable_percentage", 0.0))
        elif item["status"] == BatchItemStatus.FAILED.value:
            items_failed += 1

    return AggregateCarbonFootprint(
        total_gco2=round(total_gco2, 4),
        total_gco2_without_discount=round(total_gco2_without_discount, 4),
        batch_savings_gco2=round(total_gco2_without_discount - total_gco2, 4),
        items_completed=items_completed,
        items_failed=items_failed,
        hub_region=hub_region,
        average_carbon_intensity_gco2_kwh=round(
            sum(carbon_intensities) / len(carbon_intensities), 2
        ) if carbon_intensities else 0.0,
        average_renewable_percentage=round(
            sum(renewable_percentages) / len(renewable_percentages), 1
        ) if renewable_percentages else 0.0,
    )


# ---------------------------------------------------------------------------
# Internal batch processing
# ---------------------------------------------------------------------------

async def _process_batch_item(
    item: dict[str, Any],
    carbon_intensity_gco2_kwh: float = 0.0,  # Real values come from carbon service
    renewable_percentage: float = 0.0,  # Real values come from carbon service
    gpu_type: str = "H100",
    hub_region: str = "",  # Real values come from carbon service
) -> dict[str, Any]:
    """Process a single batch item asynchronously.

    Proxies to the configured inference backend (vLLM, Together AI, etc.)
    for real LLM inference. Falls back to an error if no backend is
    configured — no mock responses.
    """
    request_id = item["request_id"]
    model = item["model"]
    messages = item.get("messages", [])

    try:
        # Update status to processing
        item["status"] = BatchItemStatus.PROCESSING.value

        # Check for inference backend
        backend_url = getattr(settings, "inference_backend_url", "")
        if not backend_url:
            raise RuntimeError(
                "Inference backend not configured. Set HARCHOS_INFERENCE_BACKEND_URL."
            )

        # Map harchos- model IDs to backend model IDs
        backend_model = model.replace("harchos-", "")
        backend_api_key = getattr(settings, "inference_backend_api_key", "")

        # Build OpenAI-compatible request
        body = {
            "model": backend_model,
            "messages": messages,
            "temperature": item.get("temperature", 0.7),
            "top_p": item.get("top_p", 1.0),
        }
        if item.get("max_tokens"):
            body["max_tokens"] = item["max_tokens"]
        if item.get("stop"):
            body["stop"] = item["stop"]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {backend_api_key}",
        }
        url = f"{backend_url.rstrip('/')}/chat/completions"

        # Call the backend with timeout
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=headers)

        if resp.status_code != 200:
            error_detail = f"Backend returned HTTP {resp.status_code}"
            try:
                error_body = resp.json()
                error_detail = error_body.get("error", {}).get("message", error_detail)
            except Exception:
                pass
            raise RuntimeError(error_detail)

        resp_data = resp.json()

        # Extract response from backend
        prompt_tokens = resp_data.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = resp_data.get("usage", {}).get("completion_tokens", 0)

        # Get assistant message from choices
        choices = resp_data.get("choices", [])
        response_text = ""
        finish_reason = "stop"
        if choices:
            response_text = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop")

        # Estimate carbon with batch discount
        carbon = _estimate_batch_item_carbon(
            model_id=model,
            prompt_tokens=prompt_tokens or sum(_estimate_tokens(m.get("content", "")) for m in messages),
            completion_tokens=completion_tokens or min(item.get("max_tokens") or 150, 150),
            carbon_intensity_gco2_kwh=carbon_intensity_gco2_kwh,
            renewable_percentage=renewable_percentage,
            gpu_type=gpu_type,
            hub_region=hub_region,
        )

        item["status"] = BatchItemStatus.COMPLETED.value
        item["message"] = {"role": "assistant", "content": response_text}
        item["finish_reason"] = finish_reason
        item["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        item["carbon_footprint"] = carbon.model_dump()
        item["completed_at"] = time.time()

    except Exception as exc:
        logger.error("Batch item %s failed: %s", request_id, exc)
        item["status"] = BatchItemStatus.FAILED.value
        item["error"] = {
            "code": "E0500",
            "title": "Inference Failed",
            "detail": str(exc),
        }
        item["completed_at"] = time.time()

    return item


async def _process_batch_background(
    batch_id: str,
) -> None:
    """Process all items in a batch asynchronously in the background.

    Uses its own database session since it runs outside the request
    context (launched via asyncio.create_task).
    """
    async with async_session_factory() as db:
        result = await db.execute(
            select(BatchJob).where(BatchJob.id == batch_id)
        )
        batch_row: BatchJob | None = result.scalar_one_or_none()
        if not batch_row:
            return

        # Update status to processing
        batch_row.status = BatchStatus.PROCESSING.value
        batch_row.updated_at = datetime.now(timezone.utc)
        await db.commit()

        # Fetch carbon data from service (use defaults if service unavailable)
        carbon_intensity = 0.0
        renewable_pct = 0.0
        hub_region = ""
        gpu_type = "H100"

        try:
            from app.services.carbon_service import CarbonService

            async with async_session_factory() as carbon_db:
                intensity = await CarbonService.get_zone_intensity(carbon_db, "MA")
                if intensity:
                    carbon_intensity = intensity.carbon_intensity_gco2_kwh
                    renewable_pct = intensity.renewable_percentage
                    hub_region = getattr(intensity, 'zone_name', '') or "Morocco"
        except Exception as exc:
            logger.warning("Could not fetch carbon data for batch %s: %s", batch_id, exc)

        # Process items with bounded concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent items

        async def _process_with_semaphore(item: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await _process_batch_item(
                    item,
                    carbon_intensity_gco2_kwh=carbon_intensity,
                    renewable_percentage=renewable_pct,
                    gpu_type=gpu_type,
                    hub_region=hub_region,
                )

        items: list[dict[str, Any]] = json.loads(batch_row.input_data)

        # Check if batch was cancelled while we were setting up
        await db.refresh(batch_row)
        if batch_row.status == BatchStatus.CANCELLED.value:
            return

        # Run all items concurrently (bounded by semaphore)
        processed_results = await asyncio.gather(
            *[_process_with_semaphore(item) for item in items],
            return_exceptions=True,
        )

        # Update items with results (handle any unexpected exceptions from gather)
        for i, result_item in enumerate(processed_results):
            if isinstance(result_item, Exception):
                items[i]["status"] = BatchItemStatus.FAILED.value
                items[i]["error"] = {
                    "code": "E1000",
                    "title": "Internal Server Error",
                    "detail": str(result_item),
                }
                items[i]["completed_at"] = time.time()
            else:
                items[i] = result_item

        # Refresh batch from DB (may have been cancelled)
        await db.refresh(batch_row)
        if batch_row.status == BatchStatus.CANCELLED.value:
            return

        # Compute aggregate carbon
        aggregate = _compute_aggregate_carbon(items, hub_region=hub_region)

        # Update batch status
        completed_count = sum(
            1 for item in items if item["status"] == BatchItemStatus.COMPLETED.value
        )
        failed_count = sum(
            1 for item in items if item["status"] == BatchItemStatus.FAILED.value
        )

        if failed_count == len(items):
            batch_row.status = BatchStatus.FAILED.value
        else:
            batch_row.status = BatchStatus.COMPLETED.value

        batch_row.completed_items = completed_count
        batch_row.failed_items = failed_count
        batch_row.input_data = json.dumps(items)
        batch_row.results = json.dumps(items)
        batch_row.aggregate_carbon_footprint = aggregate.model_dump_json()
        batch_row.completed_at = datetime.now(timezone.utc)
        batch_row.updated_at = datetime.now(timezone.utc)

        await db.commit()

        logger.info(
            "Batch %s completed: %d/%d succeeded, %d failed, total gCO2: %.4f",
            batch_id, completed_count, len(items), failed_count, aggregate.total_gco2,
        )


# ---------------------------------------------------------------------------
# Helper: determine rate limit tier from API key
# ---------------------------------------------------------------------------

def _get_tier(api_key: ApiKey) -> str:
    """Determine rate limit tier from the API key / user role."""
    role = getattr(api_key, "user_role", None) or getattr(api_key, "role", None)
    if role == "admin":
        return "enterprise"
    elif role == "user":
        return "standard"
    # Check explicit tier attribute
    tier = getattr(api_key, "tier", None)
    if tier and tier in BATCH_SUBMISSION_RATE_LIMITS:
        return tier
    return "standard"  # Default for authenticated users


# ---------------------------------------------------------------------------
# Helper: build a BatchResponse from a BatchJob row
# ---------------------------------------------------------------------------

def _dt_to_ts(dt: datetime | None) -> float | None:
    """Convert a timezone-aware datetime to a Unix timestamp (float)."""
    if dt is None:
        return None
    return dt.timestamp()


def _build_batch_response(
    batch_row: BatchJob,
    include_results: bool = True,
) -> BatchResponse:
    """Convert a BatchJob ORM object to a BatchResponse model."""
    items: list[dict[str, Any]] = json.loads(batch_row.input_data)
    results = None
    if include_results:
        results = [
            BatchItemResult(
                request_id=item["request_id"],
                status=item["status"],
                model=item["model"],
                message=ChatMessage(**item["message"]) if item.get("message") else None,
                finish_reason=item.get("finish_reason"),
                usage=UsageInfo(**item["usage"]) if item.get("usage") else None,
                carbon_footprint=CarbonFootprint(**item["carbon_footprint"]) if item.get("carbon_footprint") else None,
                error=item.get("error"),
                completed_at=item.get("completed_at"),
            )
            for item in items
        ]

    aggregate = None
    if batch_row.aggregate_carbon_footprint:
        aggregate = AggregateCarbonFootprint(**json.loads(batch_row.aggregate_carbon_footprint))

    # Estimate time to completion for pending/processing batches
    est_completion = None
    if batch_row.status in (BatchStatus.PENDING.value, BatchStatus.PROCESSING.value):
        remaining = batch_row.total_items - batch_row.completed_items - batch_row.failed_items
        est_completion = remaining * 0.05  # ~50ms per item estimate

    metadata = None
    if batch_row.metadata_json:
        metadata = json.loads(batch_row.metadata_json)

    return BatchResponse(
        id=batch_row.id,
        status=batch_row.status,
        total_requests=batch_row.total_items,
        completed_requests=batch_row.completed_items,
        failed_requests=batch_row.failed_items,
        created_at=_dt_to_ts(batch_row.created_at) or 0.0,
        updated_at=_dt_to_ts(batch_row.updated_at) or 0.0,
        metadata=metadata,
        results=results,
        aggregate_carbon_footprint=aggregate,
        estimated_completion_seconds=est_completion,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=BatchResponse, status_code=201)
async def submit_batch(
    request: BatchSubmissionRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Submit a batch of inference requests for asynchronous processing.

    Batch inference provides a 50% cost reduction (reflected in carbon
    footprint) compared to real-time inference, similar to Together AI's
    batch API. Batches are processed asynchronously; poll the GET endpoint
    or use `stream=true` for SSE progress updates.

    Rate limits:
    - Free tier: 5 batch submissions/minute
    - Standard tier: 20 batch submissions/minute
    - Enterprise tier: 100 batch submissions/minute

    Maximum batch size: 100 requests per batch.
    """
    # Rate limit check for batch submissions
    tier = _get_tier(api_key)
    rate_key = f"batch:{api_key.id}"
    allowed, retry_after = _rate_limiter.check_submission_rate(rate_key, tier)
    if not allowed:
        raise rate_limit_exceeded(retry_after=retry_after)

    # Create batch items
    items = []
    for req in request.requests:
        items.append({
            "request_id": req.request_id,
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
            "stop": req.stop,
            "status": BatchItemStatus.PENDING.value,
            "message": None,
            "finish_reason": None,
            "usage": None,
            "carbon_footprint": None,
            "error": None,
            "completed_at": None,
        })

    now = datetime.now(timezone.utc)
    expires_at = datetime.fromtimestamp(
        now.timestamp() + BATCH_TTL_SECONDS, tz=timezone.utc
    )

    batch_row = BatchJob(
        user_id=api_key.user_id,
        status=BatchStatus.PENDING.value,
        total_items=len(items),
        completed_items=0,
        failed_items=0,
        input_data=json.dumps(items),
        results=None,
        metadata_json=json.dumps(request.metadata) if request.metadata else None,
        aggregate_carbon_footprint=None,
        carbon_aware=request.carbon_aware,
        created_at=now,
        updated_at=now,
        completed_at=None,
        expires_at=expires_at,
    )

    db.add(batch_row)
    await db.flush()

    # Start background processing
    asyncio.create_task(_process_batch_background(batch_row.id))

    logger.info(
        "Batch %s submitted: %d requests, tier=%s, user=%s",
        batch_row.id, len(items), tier, api_key.user_id,
    )

    # If streaming requested, return SSE
    if request.stream:
        return await _stream_batch_progress(batch_row.id)

    return _build_batch_response(batch_row, include_results=False)


@router.get("/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: str,
    api_key: ApiKey = Depends(require_auth),
    include_results: bool = Query(
        True,
        description="Include individual item results in the response",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Get the status and results of a batch job.

    Returns the current batch status, progress counts, and optionally
    the results for each item. Use `include_results=false` for a
    lightweight status check.

    Batch statuses:
    - `pending`: Batch received, waiting to start processing
    - `processing`: Items are being processed
    - `completed`: All items finished (some may have failed)
    - `failed`: All items failed
    - `cancelled`: Batch was cancelled by the user
    """
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id)
    )
    batch_row = result.scalar_one_or_none()
    if not batch_row:
        raise not_found("batch", batch_id)

    # Verify ownership
    if batch_row.user_id != api_key.user_id:
        raise HarchOSError(
            "E0106",
            detail=f"You do not have access to batch '{batch_id}'.",
            meta={"resource_type": "batch", "resource_id": batch_id},
        )

    return _build_batch_response(batch_row, include_results=include_results)


@router.post("/{batch_id}/cancel", response_model=BatchCancelResponse)
async def cancel_batch(
    batch_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a pending or processing batch.

    Only batches in `pending` or `processing` status can be cancelled.
    Completed, failed, or already-cancelled batches cannot be cancelled.

    When a batch is cancelled:
    - Items still `pending` will remain in `pending` state (not processed)
    - Items already `processing` will continue to completion
    - Items already `completed` or `failed` retain their results
    """
    result = await db.execute(
        select(BatchJob).where(BatchJob.id == batch_id)
    )
    batch_row = result.scalar_one_or_none()
    if not batch_row:
        raise not_found("batch", batch_id)

    # Verify ownership
    if batch_row.user_id != api_key.user_id:
        raise HarchOSError(
            "E0106",
            detail=f"You do not have access to batch '{batch_id}'.",
            meta={"resource_type": "batch", "resource_id": batch_id},
        )

    if batch_row.status not in (BatchStatus.PENDING.value, BatchStatus.PROCESSING.value):
        raise HarchOSError(
            "E0200",
            detail=f"Batch '{batch_id}' is in '{batch_row.status}' status and cannot be cancelled. "
                   "Only pending or processing batches can be cancelled.",
            meta={"batch_id": batch_id, "current_status": batch_row.status},
        )

    # Count pending items that won't be processed
    items: list[dict[str, Any]] = json.loads(batch_row.input_data)
    cancelled_count = sum(
        1 for item in items
        if item["status"] == BatchItemStatus.PENDING.value
    )

    # Mark batch as cancelled
    batch_row.status = BatchStatus.CANCELLED.value
    batch_row.updated_at = datetime.now(timezone.utc)

    # Re-compute aggregate for completed items
    completed_items = [
        item for item in items
        if item["status"] in (BatchItemStatus.COMPLETED.value, BatchItemStatus.FAILED.value)
    ]
    if completed_items:
        aggregate = _compute_aggregate_carbon(completed_items)
        batch_row.aggregate_carbon_footprint = aggregate.model_dump_json()

    batch_row.completed_items = sum(
        1 for item in items if item["status"] == BatchItemStatus.COMPLETED.value
    )
    batch_row.failed_items = sum(
        1 for item in items if item["status"] == BatchItemStatus.FAILED.value
    )

    await db.flush()

    logger.info(
        "Batch %s cancelled by user %s: %d pending items cancelled",
        batch_id, api_key.user_id, cancelled_count,
    )

    return BatchCancelResponse(
        id=batch_id,
        status=BatchStatus.CANCELLED,
        cancelled_requests=cancelled_count,
        message=f"Batch cancelled. {cancelled_count} pending items will not be processed. "
                f"Already completed/processing items retain their results.",
    )


@router.get("", response_model=BatchListResponse)
async def list_batches(
    api_key: ApiKey = Depends(require_auth),
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="Number of batches to return per page",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Offset for pagination",
    ),
    status: BatchStatus | None = Query(
        None,
        description="Filter by batch status",
    ),
    db: AsyncSession = Depends(get_db),
):
    """List batches for the authenticated user, with pagination.

    Returns batches sorted by creation time (newest first).
    Supports filtering by status and standard pagination.
    """
    # Build base query
    base_query = select(BatchJob).where(BatchJob.user_id == api_key.user_id)
    count_query = select(func.count()).select_from(BatchJob).where(BatchJob.user_id == api_key.user_id)

    if status:
        base_query = base_query.where(BatchJob.status == status.value)
        count_query = count_query.where(BatchJob.status == status.value)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results (newest first)
    query = base_query.order_by(BatchJob.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    batch_rows = result.scalars().all()

    batch_responses = [
        _build_batch_response(b, include_results=False)
        for b in batch_rows
    ]

    return BatchListResponse(
        data=batch_responses,
        total=total,
        has_more=(offset + limit) < total,
    )


# ---------------------------------------------------------------------------
# SSE streaming for batch progress
# ---------------------------------------------------------------------------

async def _stream_batch_progress(batch_id: str) -> StreamingResponse:
    """Stream batch progress updates via Server-Sent Events.

    Emits events as the batch transitions through statuses:
    - `batch.pending` — batch received
    - `batch.processing` — items being processed
    - `batch.item_completed` — individual item result
    - `batch.completed` — batch finished
    - `batch.failed` — batch failed
    - `batch.cancelled` — batch was cancelled

    The stream ends when the batch reaches a terminal state
    (completed, failed, cancelled).
    """

    async def generate() -> AsyncIterator[str]:
        # Initial event — batch pending
        initial_event = {
            "event": "batch.pending",
            "batch_id": batch_id,
            "timestamp": time.time(),
        }
        yield f"data: {json.dumps(initial_event)}\n\n"

        last_completed = 0
        last_status = BatchStatus.PENDING.value
        max_wait = 300  # 5 minute timeout
        start = time.time()

        while True:
            # Check timeout
            if time.time() - start > max_wait:
                timeout_event = {
                    "event": "batch.timeout",
                    "batch_id": batch_id,
                    "detail": "SSE stream timed out. Poll the GET endpoint for final results.",
                }
                yield f"data: {json.dumps(timeout_event)}\n\n"
                break

            # Poll the database for current batch state
            async with async_session_factory() as db:
                result = await db.execute(
                    select(BatchJob).where(BatchJob.id == batch_id)
                )
                batch_row = result.scalar_one_or_none()

            if not batch_row:
                error_event = {
                    "event": "batch.error",
                    "batch_id": batch_id,
                    "detail": "Batch not found",
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                break

            current_status = batch_row.status

            # Emit status change events
            if current_status != last_status:
                status_event = {
                    "event": f"batch.{current_status}",
                    "batch_id": batch_id,
                    "timestamp": time.time(),
                    "completed_requests": batch_row.completed_items,
                    "failed_requests": batch_row.failed_items,
                    "total_requests": batch_row.total_items,
                }
                if current_status == BatchStatus.COMPLETED.value and batch_row.aggregate_carbon_footprint:
                    status_event["aggregate_carbon_footprint"] = json.loads(batch_row.aggregate_carbon_footprint)
                yield f"data: {json.dumps(status_event)}\n\n"
                last_status = current_status

                # Terminal state — end stream
                if current_status in (
                    BatchStatus.COMPLETED.value,
                    BatchStatus.FAILED.value,
                    BatchStatus.CANCELLED.value,
                ):
                    break

            # Emit item completion events
            current_completed = batch_row.completed_items + batch_row.failed_items
            if current_completed > last_completed:
                items: list[dict[str, Any]] = json.loads(batch_row.input_data)
                for item in items[last_completed:current_completed]:
                    item_event = {
                        "event": "batch.item_completed",
                        "batch_id": batch_id,
                        "request_id": item.get("request_id"),
                        "status": item.get("status"),
                        "timestamp": item.get("completed_at", time.time()),
                    }
                    yield f"data: {json.dumps(item_event)}\n\n"
                last_completed = current_completed

            # Small sleep to avoid busy-polling
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
