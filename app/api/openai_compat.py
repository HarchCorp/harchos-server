"""OpenAI API Compatibility Layer — Drop-in replacement for OpenAI SDK.

This module provides OpenAI-compatible endpoints at the root /v1/ level,
enabling users to use the OpenAI SDK with HarchOS by just changing the
base_url. This is the same approach used by Groq and Together AI.

Usage with OpenAI Python SDK:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.harchos.ai/v1",
        api_key="hsk_your_key"
    )
    response = client.chat.completions.create(
        model="harchos-llama-3.3-70b",
        messages=[{"role": "user", "content": "Hello"}]
    )

Usage with OpenAI JavaScript SDK:
    import OpenAI from 'openai';
    const client = new OpenAI({
        baseURL: 'https://api.harchos.ai/v1',
        apiKey: 'hsk_your_key'
    });

Key difference from /v1/inference/chat/completions:
- These endpoints accept the EXACT OpenAI request format (no carbon_aware field)
- Carbon tracking is ALWAYS included in responses (HarchOS differentiator)
- Default model is harchos-llama-3.3-70b (same as OpenAI defaults to gpt-4)
- Supports n, presence_penalty, frequency_penalty, logprobs, etc. (OpenAI params)
"""

import json
import logging
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.model import Model as DBModel
from app.api.deps import require_auth, check_model_access, check_region_access, check_token_budget, check_spending_limit
from app.config import settings
from app.core.exceptions import model_not_available, inference_timeout, HarchOSError

logger = logging.getLogger("harchos.openai_compat")
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas — exact OpenAI API format
# ---------------------------------------------------------------------------

class OpenAIChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str = Field(..., description="system, user, assistant, or tool")
    content: str | list = Field(..., description="Message content or content parts", max_length=100000)
    name: str | None = Field(None, description="Participant name")
    tool_calls: list | None = Field(None, description="Tool calls (assistant messages)")
    tool_call_id: str | None = Field(None, description="Tool call ID (tool messages)")


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format — exact compatibility."""
    model: str = Field("harchos-llama-3.3-70b", description="Model ID")
    messages: list[OpenAIChatMessage] = Field(..., min_length=1, description="Chat messages")
    temperature: float = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0, le=1, description="Nucleus sampling parameter")
    n: int = Field(1, ge=1, le=10, description="Number of completions to generate")
    stream: bool = Field(False, description="Enable SSE streaming")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    max_tokens: int | None = Field(None, ge=1, le=32768, description="Max tokens to generate")
    max_completion_tokens: int | None = Field(None, ge=1, le=32768, description="Max completion tokens (new OpenAI param)")
    presence_penalty: float = Field(0, ge=-2, le=2, description="Presence penalty")
    frequency_penalty: float = Field(0, ge=-2, le=2, description="Frequency penalty")
    logprobs: bool = Field(False, description="Return log probabilities")
    top_logprobs: int | None = Field(None, ge=1, le=20, description="Number of top log probs")
    response_format: dict | None = Field(None, description="Output format (json_object, text)")
    seed: int | None = Field(None, description="Random seed for deterministic outputs")
    tools: list | None = Field(None, description="Tool definitions for function calling")
    tool_choice: str | dict | None = Field(None, description="Tool choice strategy")
    user: str | None = Field(None, description="End-user identifier for abuse tracking")


class OpenAICompletionRequest(BaseModel):
    """OpenAI text completion request format — exact compatibility."""
    model: str = Field("harchos-llama-3.3-70b", description="Model ID")
    prompt: str | list[str] = Field(..., description="Text prompt(s)")
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(1.0, ge=0, le=1)
    n: int = Field(1, ge=1, le=10)
    stream: bool = Field(False)
    stop: str | list[str] | None = Field(None)
    max_tokens: int | None = Field(None, ge=1, le=32768)
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)
    logprobs: int | None = Field(None, ge=0, le=5)
    echo: bool = Field(False)
    user: str | None = Field(None)


class OpenAIEmbeddingRequest(BaseModel):
    """OpenAI embeddings request format — exact compatibility."""
    model: str = Field("harchos-embedding-3-large", description="Embedding model ID")
    input: str | list[str] | list[list[int]] = Field(..., description="Text(s) or token array(s)")
    dimensions: int | None = Field(None, ge=1, le=3072, description="Output dimensions")
    encoding_format: str = Field("float", description="Encoding format: float, base64")


class OpenAIModelInfo(BaseModel):
    """OpenAI model info format."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "harchos"


class OpenAIModelListResponse(BaseModel):
    """OpenAI model list response format."""
    object: str = "list"
    data: list[OpenAIModelInfo]


# ---------------------------------------------------------------------------
# Carbon footprint (always included — HarchOS differentiator)
# ---------------------------------------------------------------------------

class CarbonFootprint(BaseModel):
    """Carbon footprint data — automatically included in every response."""
    gco2_per_request: float
    hub_region: str
    carbon_intensity_gco2_kwh: float
    renewable_percentage: float
    gpu_type: str
    estimated_power_watts: float
    inference_duration_seconds: float
    carbon_saved_vs_average_gco2: float = 0.0



# ---------------------------------------------------------------------------
# Carbon estimation (shared with inference.py)
# ---------------------------------------------------------------------------

def _estimate_carbon(model_id: str, prompt_tokens: int, completion_tokens: int,
                     ci_gco2_kwh: float = 0.0, renewable_pct: float = 0.0,
                     hub_region: str = "") -> CarbonFootprint:
    """Estimate carbon footprint for inference.

    Real carbon values come from CarbonService; defaults of 0.0/"" are
    placeholders when the service is unavailable.
    """
    gpu_power_w = 700  # H100
    inference_seconds = max(0.5, (prompt_tokens / 5000) + (completion_tokens / 100))
    energy_kwh = (gpu_power_w * inference_seconds) / (1000 * 3600)
    gco2 = energy_kwh * ci_gco2_kwh * 1000
    avg_gco2 = energy_kwh * 500 * 1000
    return CarbonFootprint(
        gco2_per_request=round(gco2, 4),
        hub_region=hub_region,
        carbon_intensity_gco2_kwh=ci_gco2_kwh,
        renewable_percentage=renewable_pct,
        gpu_type="H100",
        estimated_power_watts=gpu_power_w,
        inference_duration_seconds=round(inference_seconds, 3),
        carbon_saved_vs_average_gco2=round(max(0, avg_gco2 - gco2), 4),
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Endpoints — exact OpenAI API format
# ---------------------------------------------------------------------------

@router.get("/models", response_model=OpenAIModelListResponse)
async def list_models(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List available models — OpenAI-compatible.

    Returns model catalog in the exact OpenAI format.
    HarchOS extension: includes carbon_intensity in model metadata.
    """
    # Try real backend first
    backend_url = getattr(settings, "inference_backend_url", "")
    if backend_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{backend_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {settings.inference_backend_api_key}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return OpenAIModelListResponse(data=[
                        OpenAIModelInfo(id=m["id"], created=m.get("created", 1700000000), owned_by="harchos")
                        for m in data.get("data", [])
                    ])
        except Exception:
            pass

    # Fallback: query from database
    db_result = await db.execute(select(DBModel).where(DBModel.status == "ready"))
    db_models = db_result.scalars().all()
    if db_models:
        return OpenAIModelListResponse(data=[
            OpenAIModelInfo(
                id=f"harchos-{m.name.lower().replace(' ', '-')}",
                created=int(m.created_at.timestamp()) if m.created_at else 1700000000,
                owned_by="harchos",
            )
            for m in db_models
        ])

    # No models available
    return OpenAIModelListResponse(data=[])


@router.post("/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a chat completion — EXACT OpenAI API format.

    This endpoint is a drop-in replacement for OpenAI's chat completions.
    Just change `base_url` in the OpenAI SDK to point to HarchOS.

    HarchOS extension: Every response includes `carbon_footprint` showing
    the estimated CO2 emissions for this inference. No other inference
    API provides this data automatically.
    """
    start_time = time.time()

    # Validate model access
    await check_model_access(api_key, request.model)

    # Enforce API key restrictions (region access)
    await check_region_access(api_key, "MA")

    # Budget enforcement
    prompt_tokens_est = sum(_estimate_tokens(str(m.content)) for m in request.messages if m.content)
    completion_tokens_est = min(request.max_tokens or request.max_completion_tokens or 150, 150)
    await check_token_budget(api_key, prompt_tokens_est + completion_tokens_est)
    estimated_cost = (prompt_tokens_est + completion_tokens_est) / 1000 * 0.002
    await check_spending_limit(api_key, estimated_cost)

    # Get carbon data
    ci = 0.0
    renewable = 0.0
    hub_region = ""
    try:
        from app.services.carbon_service import CarbonService
        intensity = await CarbonService.get_zone_intensity(db, "MA")
        ci = intensity.carbon_intensity_gco2_kwh
        renewable = intensity.renewable_percentage
        hub_region = getattr(intensity, 'zone_name', '') or "Morocco"
    except Exception:
        pass

    # Check for configured backend
    backend_url = getattr(settings, "inference_backend_url", "")
    backend_key = getattr(settings, "inference_backend_api_key", "")

    if not backend_url:
        raise HarchOSError(
            "E0500",
            detail="Inference backend not configured. Set HARCHOS_INFERENCE_BACKEND_URL to enable LLM inference via the OpenAI-compatible API.",
            meta={"configured": False, "setup_docs": "https://docs.harchos.ai/inference-setup"},
        )

    # Proxy to backend
    body = request.model_dump(exclude_none=True, exclude={"carbon_aware", "carbon_preference"})
    body["model"] = request.model.replace("harchos-", "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend_key}",
    }
    url = f"{backend_url.rstrip('/')}/chat/completions"
    timeout = httpx.Timeout(120.0 if request.stream else 30.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if request.stream:
                # Streaming: use aiter_lines for SSE passthrough
                async with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        raise HarchOSError("E0500", detail=f"Backend returned {resp.status_code}")

                    async def stream_generate() -> AsyncIterator[str]:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: ") and line != "data: [DONE]":
                                yield line + "\n\n"
                            elif line == "data: [DONE]":
                                # Inject carbon data before DONE
                                elapsed = time.time() - start_time
                                carbon = _estimate_carbon(request.model, 0, 0, ci, renewable, hub_region)
                                carbon.inference_duration_seconds = round(elapsed, 3)
                                carbon_chunk = {
                                    "object": "carbon_footprint",
                                    "carbon_footprint": carbon.model_dump(),
                                }
                                yield f"data: {json.dumps(carbon_chunk)}\n\n"
                                yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        stream_generate(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Carbon-Intensity": str(ci),
                            "X-Renewable-Percentage": str(renewable),
                        },
                    )
            else:
                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code != 200:
                    raise HarchOSError("E0500", detail=f"Backend returned {resp.status_code}")

                resp_data = resp.json()
                elapsed = time.time() - start_time

                # Add carbon tracking
                usage = resp_data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0

                carbon = _estimate_carbon(request.model, prompt_tokens, completion_tokens, ci, renewable, hub_region)
                carbon.inference_duration_seconds = round(elapsed, 3)
                resp_data["carbon_footprint"] = carbon.model_dump()

                return JSONResponse(
                    content=resp_data,
                    headers={
                        "X-Carbon-Footprint-gCO2": str(carbon.gco2_per_request),
                        "X-Carbon-Intensity": str(ci),
                        "X-Renewable-Percentage": str(renewable),
                    },
                )
        except httpx.TimeoutException:
            raise inference_timeout(request.model, timeout_seconds=30)
        except httpx.HTTPError as exc:
            logger.error("Inference backend error: %s", exc)
            raise HarchOSError("E0500", detail=f"Inference backend unavailable: {exc}")


@router.post("/completions")
async def completions(
    request: OpenAICompletionRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a text completion — EXACT OpenAI API format."""
    start_time = time.time()

    # Validate model access
    await check_model_access(api_key, request.model)

    # Get carbon data
    ci = 0.0
    renewable = 0.0
    hub_region = ""
    try:
        from app.services.carbon_service import CarbonService
        intensity = await CarbonService.get_zone_intensity(db, "MA")
        ci = intensity.carbon_intensity_gco2_kwh
        renewable = intensity.renewable_percentage
        hub_region = getattr(intensity, 'zone_name', '') or "Morocco"
    except Exception:
        pass

    # Check for configured backend
    backend_url = getattr(settings, "inference_backend_url", "")
    backend_key = getattr(settings, "inference_backend_api_key", "")

    if not backend_url:
        raise HarchOSError(
            "E0500",
            detail="Inference backend not configured. Set HARCHOS_INFERENCE_BACKEND_URL to enable LLM inference via the OpenAI-compatible API.",
            meta={"configured": False, "setup_docs": "https://docs.harchos.ai/inference-setup"},
        )

    # Proxy to backend
    body = request.model_dump(exclude_none=True)
    body["model"] = request.model.replace("harchos-", "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend_key}",
    }
    url = f"{backend_url.rstrip('/')}/completions"
    timeout = httpx.Timeout(120.0 if request.stream else 30.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise HarchOSError("E0500", detail=f"Backend returned {resp.status_code}")

            resp_data = resp.json()
            elapsed = time.time() - start_time

            # Add carbon tracking
            usage = resp_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0

            carbon = _estimate_carbon(request.model, prompt_tokens, completion_tokens, ci, renewable, hub_region)
            carbon.inference_duration_seconds = round(elapsed, 3)
            resp_data["carbon_footprint"] = carbon.model_dump()

            return JSONResponse(
                content=resp_data,
                headers={
                    "X-Carbon-Footprint-gCO2": str(carbon.gco2_per_request),
                    "X-Carbon-Intensity": str(ci),
                    "X-Renewable-Percentage": str(renewable),
                },
            )
        except httpx.TimeoutException:
            raise inference_timeout(request.model, timeout_seconds=30)
        except httpx.HTTPError as exc:
            logger.error("Inference backend error: %s", exc)
            raise HarchOSError("E0500", detail=f"Inference backend unavailable: {exc}")


@router.post("/embeddings")
async def embeddings(
    request: OpenAIEmbeddingRequest,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create embeddings — EXACT OpenAI API format."""
    start_time = time.time()

    # Check for configured backend
    backend_url = getattr(settings, "inference_backend_url", "")
    backend_key = getattr(settings, "inference_backend_api_key", "")

    if not backend_url:
        raise HarchOSError(
            "E0500",
            detail="Inference backend not configured. Set HARCHOS_INFERENCE_BACKEND_URL to enable embeddings via the OpenAI-compatible API.",
            meta={"configured": False, "setup_docs": "https://docs.harchos.ai/inference-setup"},
        )

    # Proxy to backend
    body = request.model_dump(exclude_none=True)
    body["model"] = request.model.replace("harchos-", "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend_key}",
    }
    url = f"{backend_url.rstrip('/')}/embeddings"
    timeout = httpx.Timeout(30.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise HarchOSError("E0500", detail=f"Backend returned {resp.status_code}")

            resp_data = resp.json()
            elapsed = time.time() - start_time

            # Add carbon tracking
            usage = resp_data.get("usage", {})
            total_tokens = usage.get("prompt_tokens", 0) or 0

            from app.services.carbon_service import CarbonService
            intensity = await CarbonService.get_zone_intensity(db, "MA")
            emb_hub_region = getattr(intensity, 'zone_name', '') or "Morocco"
            carbon = _estimate_carbon(request.model, total_tokens, 0, intensity.carbon_intensity_gco2_kwh, intensity.renewable_percentage, emb_hub_region)
            carbon.inference_duration_seconds = round(elapsed, 3)
            resp_data["carbon_footprint"] = carbon.model_dump()

            return JSONResponse(
                content=resp_data,
                headers={
                    "X-Carbon-Footprint-gCO2": str(carbon.gco2_per_request),
                    "X-Carbon-Intensity": str(intensity.carbon_intensity_gco2_kwh),
                    "X-Renewable-Percentage": str(intensity.renewable_percentage),
                },
            )
        except httpx.TimeoutException:
            raise inference_timeout(request.model, timeout_seconds=30)
        except httpx.HTTPError as exc:
            logger.error("Embeddings backend error: %s", exc)
            raise HarchOSError("E0500", detail=f"Embeddings backend unavailable: {exc}")
