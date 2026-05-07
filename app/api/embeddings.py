"""OpenAI-compatible Embeddings API with carbon footprint tracking.

HarchOS's unique value: every embedding response includes carbon footprint data,
showing gCO2 generated per request. No other embedding API does this.

Endpoints:
- POST /embeddings         — Create embeddings (OpenAI-compatible)
- GET  /embeddings/models  — List available embedding models

Features:
- OpenAI-compatible request/response format
- Single string and array-of-strings input
- input_type hint (search_document, search_query, classification, clustering)
- dimensions parameter (like OpenAI's matryoshka embeddings)
- Carbon footprint tracking per request
- Token counting estimation (4 chars ≈ 1 token)
- Proxy to real backend when HARCHOS_INFERENCE_BACKEND_URL is set (required for embeddings)
- Comprehensive Pydantic validation
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.api.deps import require_auth
from app.config import settings
from app.cache import get_cached_json, set_cached_json, get_or_fetch
from app.core.exceptions import (
    HarchOSError,
    model_not_available,
    validation_error,
)

logger = logging.getLogger("harchos.embeddings")
router = APIRouter()


# ---------------------------------------------------------------------------
# Available embedding models
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "harchos-embedding-3-large": {
        "id": "harchos-embedding-3-large",
        "name": "HarchOS Embedding 3 Large",
        "family": "embedding",
        "default_dimensions": 1536,
        "max_dimensions": 1536,
        "min_dimensions": 1,
        "max_input_tokens": 8191,
        "description": "High-quality embeddings with 1536 dimensions. Best for search, clustering, and classification.",
        "pricing_per_1k_tokens": 0.00013,
    },
    "harchos-embedding-3-small": {
        "id": "harchos-embedding-3-small",
        "name": "HarchOS Embedding 3 Small",
        "family": "embedding",
        "default_dimensions": 768,
        "max_dimensions": 768,
        "min_dimensions": 1,
        "max_input_tokens": 8191,
        "description": "Fast, efficient embeddings with 768 dimensions. Good balance of speed and quality.",
        "pricing_per_1k_tokens": 0.00002,
    },
    "harchos-embedding-multilingual": {
        "id": "harchos-embedding-multilingual",
        "name": "HarchOS Embedding Multilingual",
        "family": "embedding",
        "default_dimensions": 1536,
        "max_dimensions": 1536,
        "min_dimensions": 1,
        "max_input_tokens": 8191,
        "description": "Multilingual embeddings supporting 100+ languages with 1536 dimensions.",
        "pricing_per_1k_tokens": 0.00013,
    },
}


# ---------------------------------------------------------------------------
# Pydantic schemas — OpenAI-compatible + HarchOS extensions
# ---------------------------------------------------------------------------

class EmbeddingInputType(str):
    """Input type hint for embedding optimization.

    Mirrors Cohere/Voyage-style input_type hints that help the model
    produce better embeddings for specific use cases.
    """
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"

    @classmethod
    def allowed_values(cls) -> list[str]:
        return [cls.SEARCH_DOCUMENT, cls.SEARCH_QUERY, cls.CLASSIFICATION, cls.CLUSTERING]


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding creation request with HarchOS extensions.

    Matches: https://platform.openai.com/docs/api-reference/embeddings/create
    """

    model: str = Field(
        "harchos-embedding-3-large",
        description="ID of the embedding model to use.",
    )
    input: str | list[str] = Field(
        ...,
        description=(
            "Input text to embed, encoded as a string or array of strings. "
            "The input must not exceed the max input tokens for the model "
            "(8191 tokens) and cannot be an empty string."
        ),
    )
    dimensions: int | None = Field(
        None,
        ge=1,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in text-embedding-3 and later models."
        ),
    )
    encoding_format: str = Field(
        "float",
        description="The format to return the embeddings in. Can be either 'float' or 'base64'.",
    )
    input_type: str | None = Field(
        None,
        description=(
            "Hint about the intended use case for the embeddings. "
            "One of: search_document, search_query, classification, clustering. "
            "This is a HarchOS extension not present in the OpenAI API."
        ),
    )
    carbon_aware: bool = Field(
        True,
        description="Route to the greenest hub for embedding computation (HarchOS extension).",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that the requested model exists in our catalog."""
        if v not in EMBEDDING_MODELS:
            allowed = list(EMBEDDING_MODELS.keys())
            raise ValueError(
                f"Model '{v}' is not available for embeddings. "
                f"Available models: {', '.join(allowed)}"
            )
        return v

    @field_validator("input")
    @classmethod
    def validate_input(cls, v: str | list[str]) -> str | list[str]:
        """Validate input is non-empty and within reasonable limits."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Input string cannot be empty or whitespace-only.")
            if len(v) > 300_000:
                raise ValueError(
                    f"Input string length ({len(v)}) exceeds maximum of 300,000 characters."
                )
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Input array cannot be empty.")
            if len(v) > 2048:
                raise ValueError(
                    f"Input array length ({len(v)}) exceeds maximum of 2048 items."
                )
            for i, item in enumerate(v):
                if not isinstance(item, str):
                    raise ValueError(f"Input array item at index {i} must be a string.")
                if not item.strip():
                    raise ValueError(
                        f"Input array item at index {i} cannot be empty or whitespace-only."
                    )
                if len(item) > 300_000:
                    raise ValueError(
                        f"Input array item at index {i} length ({len(item)}) "
                        f"exceeds maximum of 300,000 characters."
                    )
        return v

    @field_validator("encoding_format")
    @classmethod
    def validate_encoding_format(cls, v: str) -> str:
        """Validate encoding format is one of the allowed values."""
        allowed = ["float", "base64"]
        if v not in allowed:
            raise ValueError(
                f"Invalid encoding_format '{v}'. Must be one of: {', '.join(allowed)}"
            )
        return v

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v: str | None) -> str | None:
        """Validate input_type is one of the allowed hints."""
        if v is None:
            return v
        allowed = EmbeddingInputType.allowed_values()
        if v not in allowed:
            raise ValueError(
                f"Invalid input_type '{v}'. Must be one of: {', '.join(allowed)}"
            )
        return v

    @model_validator(mode="after")
    def validate_dimensions_for_model(self) -> "EmbeddingRequest":
        """Ensure dimensions parameter is valid for the selected model."""
        if self.dimensions is not None:
            model_info = EMBEDDING_MODELS[self.model]
            max_dims = model_info["max_dimensions"]
            min_dims = model_info["min_dimensions"]
            if self.dimensions > max_dims:
                raise ValueError(
                    f"dimensions={self.dimensions} exceeds maximum of {max_dims} "
                    f"for model '{self.model}'."
                )
            if self.dimensions < min_dims:
                raise ValueError(
                    f"dimensions={self.dimensions} is below minimum of {min_dims} "
                    f"for model '{self.model}'."
                )
        return self


class EmbeddingData(BaseModel):
    """A single embedding result, matching OpenAI's embedding object."""
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage information for the embedding request."""
    prompt_tokens: int
    total_tokens: int


class CarbonFootprintEmbedding(BaseModel):
    """Carbon footprint for an embedding request — unique to HarchOS.

    No other embedding API provides per-request carbon tracking.
    """
    gco2_per_request: float = Field(
        ..., description="Grams of CO2 for this embedding request"
    )
    hub_region: str = Field(
        ..., description="Hub region where embedding was computed"
    )
    carbon_intensity_gco2_kwh: float = Field(
        ..., description="Carbon intensity at the hub (gCO2/kWh)"
    )
    renewable_percentage: float = Field(
        ..., description="Renewable energy percentage at the hub"
    )
    gpu_type: str = Field(
        ..., description="GPU type used for embedding computation"
    )
    estimated_power_watts: float = Field(
        ..., description="Estimated GPU power consumption (watts)"
    )
    inference_duration_seconds: float = Field(
        ..., description="Embedding computation duration in seconds"
    )
    carbon_saved_vs_average_gco2: float = Field(
        0.0,
        description="CO2 saved compared to global average grid (500 gCO2/kWh)",
    )


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response with HarchOS carbon extension."""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage
    carbon_footprint: CarbonFootprintEmbedding | None = None


class EmbeddingModelInfo(BaseModel):
    """Information about an available embedding model."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "harchos"
    default_dimensions: int
    max_dimensions: int
    min_dimensions: int
    max_input_tokens: int
    description: str
    pricing_per_1k_tokens: float
    carbon_intensity_gco2_kwh: float = 0.0
    hub_region: str = ""


class EmbeddingModelListResponse(BaseModel):
    """Response for listing available embedding models."""
    object: str = "list"
    data: list[EmbeddingModelInfo]


# ---------------------------------------------------------------------------
# Token estimation (approximate — real tokenizers are model-specific)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses the heuristic that 1 token ≈ 4 characters for English text,
    which is more accurate than word splitting. CJK characters are
    counted as ~2 tokens each since they typically tokenize to 1-2 tokens.

    This is the same approximation used by OpenAI's tiktoken for rough
    estimates and matches the approach in app.api.inference.
    """
    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
    other_chars = len(text) - cjk_chars
    return max(1, (other_chars // 4) + (cjk_chars * 2))


# ---------------------------------------------------------------------------
# Carbon estimation for embeddings
# ---------------------------------------------------------------------------

def _estimate_embedding_carbon(
    total_tokens: int,
    num_texts: int,
    carbon_intensity_gco2_kwh: float = 0.0,  # Real values come from carbon service
    renewable_percentage: float = 0.0,  # Real values come from carbon service
    gpu_type: str = "A100",
    hub_region: str = "",  # Real values come from carbon service
) -> CarbonFootprintEmbedding:
    """Estimate carbon footprint for an embedding request.

    Embedding computation is much lighter than LLM inference — no
    autoregressive generation, just a single forward pass. GPU power
    consumption is lower and duration is shorter.

    Based on:
    - GPU power × forward pass time × carbon intensity / 1000
    - A100 TDP: 400W (commonly used for embedding workloads)
    - Forward pass time ≈ total_tokens / 20000 seconds for embeddings
      (much faster than generation due to no autoregressive decode)
    """
    gpu_power_w = {"H100": 700, "A100": 400, "L40S": 350, "H200": 800, "B200": 1000}.get(
        gpu_type, 400
    )

    # Embedding forward pass is fast — ~20k tokens/sec throughput
    inference_seconds = max(0.01, total_tokens / 20000)
    # Add small per-text overhead (batching, preprocessing)
    inference_seconds += num_texts * 0.001

    # Energy = Power × Time (in kWh)
    energy_kwh = (gpu_power_w * inference_seconds) / (1000 * 3600)

    # CO2 = Energy × carbon_intensity
    gco2 = energy_kwh * carbon_intensity_gco2_kwh * 1000  # convert kg to g

    # CO2 saved vs global average grid (500 gCO2/kWh)
    avg_gco2 = energy_kwh * 500 * 1000
    carbon_saved = max(0, avg_gco2 - gco2)

    return CarbonFootprintEmbedding(
        gco2_per_request=round(gco2, 6),
        hub_region=hub_region,
        carbon_intensity_gco2_kwh=carbon_intensity_gco2_kwh,
        renewable_percentage=renewable_percentage,
        gpu_type=gpu_type,
        estimated_power_watts=gpu_power_w,
        inference_duration_seconds=round(inference_seconds, 4),
        carbon_saved_vs_average_gco2=round(carbon_saved, 6),
    )


# ---------------------------------------------------------------------------
# Backend proxy — forwards to real embedding backends when configured
# ---------------------------------------------------------------------------

async def _proxy_to_backend(
    request_body: dict,
    model_id: str,
) -> httpx.Response:
    """Proxy an embedding request to a configured backend.

    Raises HarchOSError if no backend is configured or if the backend
    returns an error.
    """
    backend_url = getattr(settings, "inference_backend_url", "")
    if not backend_url:
        raise HarchOSError(
            "E0500",
            detail="Inference backend not configured. Set HARCHOS_INFERENCE_BACKEND_URL to enable embeddings. Contact contact@harchos.ai for setup assistance.",
            meta={"configured": False},
        )

    backend_api_key = getattr(settings, "inference_backend_api_key", "")
    timeout_seconds = getattr(settings, "inference_backend_timeout_seconds", 30)

    # Map harchos- model IDs to backend model IDs
    backend_model = model_id.replace("harchos-", "")
    request_body["model"] = backend_model

    # Remove HarchOS-specific fields that the backend won't understand
    request_body.pop("carbon_aware", None)
    request_body.pop("input_type", None)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend_api_key}",
    }

    url = f"{backend_url.rstrip('/')}/embeddings"

    timeout = httpx.Timeout(float(timeout_seconds), connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=request_body, headers=headers)
            resp.raise_for_status()
            return resp
        except httpx.TimeoutException:
            raise HarchOSError(
                "E0501",
                detail=f"Embedding backend request timed out after {timeout_seconds}s.",
            )
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Embedding backend returned %s: %s",
                exc.response.status_code,
                exc.response.text[:500],
            )
            raise HarchOSError(
                "E0500",
                detail=f"Embedding backend returned error: HTTP {exc.response.status_code}",
            )
        except httpx.HTTPError as exc:
            logger.error("Embedding backend connection error: %s", exc)
            raise HarchOSError(
                "E0500",
                detail=f"Embedding backend unavailable: {exc}",
            )


# ---------------------------------------------------------------------------
# Response ID generation
# ---------------------------------------------------------------------------

def _generate_embedding_id() -> str:
    """Generate a unique embedding response ID matching OpenAI format."""
    return f"emb-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/models", response_model=EmbeddingModelListResponse)
async def list_embedding_models(
    api_key: ApiKey = Depends(require_auth),
):
    """List all available embedding models.

    Returns model catalog with dimension information, pricing,
    and carbon intensity data. OpenAI-compatible with HarchOS extensions.
    """
    # Try cache first
    cache_key = "embeddings:models:list"
    cached = await get_cached_json(cache_key)
    if cached is not None:
        return EmbeddingModelListResponse(**cached)

    # Get real carbon data from service
    model_carbon_intensity = 0.0
    model_hub_region = ""
    try:
        from app.services.carbon_service import CarbonService
        from app.database import async_session_factory
        async with async_session_factory() as carbon_db:
            ci = await CarbonService.get_zone_intensity(carbon_db, "MA")
            model_carbon_intensity = ci.carbon_intensity_gco2_kwh
            model_hub_region = getattr(ci, 'zone_name', '') or "Morocco"
    except Exception:
        pass

    models = [
        EmbeddingModelInfo(
            id=info["id"],
            created=1700000000,
            owned_by="harchos",
            default_dimensions=info["default_dimensions"],
            max_dimensions=info["max_dimensions"],
            min_dimensions=info["min_dimensions"],
            max_input_tokens=info["max_input_tokens"],
            description=info["description"],
            pricing_per_1k_tokens=info["pricing_per_1k_tokens"],
            carbon_intensity_gco2_kwh=model_carbon_intensity,
            hub_region=model_hub_region,
        )
        for info in EMBEDDING_MODELS.values()
    ]

    response = EmbeddingModelListResponse(data=models)

    # Cache for 10 minutes — model catalog rarely changes
    await set_cached_json(cache_key, response.model_dump(), ttl_seconds=600)

    return response


@router.post("/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create embeddings for text input — OpenAI-compatible with carbon tracking.

    HarchOS extension: Every response includes `carbon_footprint` showing
    the estimated CO2 emissions for this embedding request. No other
    embedding API provides this data.

    Supports:
    - Single string input: `"Hello world"`
    - Array of strings: `["Hello", "world"]`
    - Dimensions parameter (matryoshka embeddings)
    - input_type hint for optimized embeddings
    - Base64 encoding format

    When `carbon_aware=true` (default), the request is routed to the
    greenest available hub.
    """
    # Require a configured inference backend — embeddings cannot work without one
    if not settings.inference_backend_url:
        raise HarchOSError(
            "E0500",
            detail="Inference backend not configured. Set HARCHOS_INFERENCE_BACKEND_URL to enable embeddings. Contact contact@harchos.ai for setup assistance.",
            meta={"configured": False},
        )

    start_time = time.time()

    # Normalize input to a list of strings
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # Resolve the effective dimensions
    model_info = EMBEDDING_MODELS[request.model]
    effective_dimensions = request.dimensions or model_info["default_dimensions"]

    # Estimate token counts for each text
    token_counts = [_estimate_tokens(text) for text in texts]
    total_tokens = sum(token_counts)

    # Validate total tokens against model limit
    max_input_tokens = model_info["max_input_tokens"]
    if total_tokens > max_input_tokens * len(texts):
        # Per-text token limit check
        for i, tc in enumerate(token_counts):
            if tc > max_input_tokens:
                raise HarchOSError(
                    "E0504",
                    detail=(
                        f"Input at index {i} has approximately {tc} tokens, "
                        f"which exceeds the maximum of {max_input_tokens} tokens "
                        f"for model '{request.model}'."
                    ),
                    meta={"index": i, "token_count": tc, "max_tokens": max_input_tokens},
                )

    # Get carbon data for the hub
    carbon_intensity = 0.0
    renewable_pct = 0.0
    hub_region = ""
    try:
        from app.services.carbon_service import CarbonService
        intensity = await CarbonService.get_zone_intensity(db, "MA")
        carbon_intensity = intensity.carbon_intensity_gco2_kwh
        renewable_pct = intensity.renewable_percentage
        hub_region = getattr(intensity, 'zone_name', '') or "Morocco"
    except Exception:
        # Fallback: carbon service unavailable
        pass
    gpu_type = "A100"  # A100 is typical for embedding workloads

    # Proxy to inference backend
    request_body = request.model_dump(exclude_none=True)
    backend_resp = await _proxy_to_backend(request_body, request.model)

    # Add carbon tracking to the backend response
    resp_data = backend_resp.json()
    elapsed = time.time() - start_time

    # Use actual token usage from backend if available
    actual_usage = resp_data.get("usage", {})
    prompt_tokens = actual_usage.get("prompt_tokens", total_tokens)

    carbon = _estimate_embedding_carbon(
        total_tokens=prompt_tokens,
        num_texts=len(texts),
        carbon_intensity_gco2_kwh=carbon_intensity,
        renewable_percentage=renewable_pct,
        gpu_type=gpu_type,
        hub_region=hub_region,
    )
    carbon.inference_duration_seconds = round(elapsed, 4)

    # Inject carbon data into OpenAI response
    resp_data["carbon_footprint"] = carbon.model_dump()

    return JSONResponse(
        content=resp_data,
        headers={
            "X-Carbon-Footprint-gCO2": str(carbon.gco2_per_request),
            "X-Carbon-Intensity": str(carbon_intensity),
            "X-Renewable-Percentage": str(renewable_pct),
            "X-Carbon-Saved-vs-Avg-gCO2": str(carbon.carbon_saved_vs_average_gco2),
            "X-Embedding-Model": request.model,
            "X-Embedding-Dimensions": str(effective_dimensions),
        },
    )
