"""Prometheus metrics middleware — pure ASGI for zero overhead.

Collects HTTP request duration, request counts, and carbon metrics.
Exposes /v1/metrics endpoint for Prometheus scraping.

v2: Converted from BaseHTTPMiddleware to pure ASGI to eliminate
body_iterator consumption and task spawning overhead.
"""

import time
import logging

from starlette.types import ASGIApp, Receive, Scope, Send
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger("harchos.metrics")

# ---------------------------------------------------------------------------
# Prometheus metrics definitions
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "harchos_http_requests_total",
    "Total count of HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "harchos_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUESTS_IN_PROGRESS = Gauge(
    "harchos_http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    ["method", "endpoint"],
)

# Carbon metrics (updated by carbon service)
CARBON_INTENSITY_GAUGE = Gauge(
    "harchos_carbon_intensity_gco2_kwh",
    "Current carbon intensity in gCO2/kWh by zone",
    ["zone"],
)

CARBON_SAVED_KG_TOTAL = Counter(
    "harchos_carbon_saved_kg_total",
    "Total CO2 saved in kg via carbon-aware scheduling",
)

WORKLOADS_OPTIMIZED_TOTAL = Counter(
    "harchos_workloads_optimized_total",
    "Total workloads optimized by carbon-aware scheduling",
    ["action"],
)

# Inference metrics
INFERENCE_REQUESTS_TOTAL = Counter(
    "harchos_inference_requests_total",
    "Total LLM inference requests",
    ["model", "stream"],
)

INFERENCE_DURATION = Histogram(
    "harchos_inference_duration_seconds",
    "LLM inference duration in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

INFERENCE_CARBON_GCO2_TOTAL = Counter(
    "harchos_inference_carbon_gco2_total",
    "Total gCO2 from LLM inference",
    ["model"],
)

INFERENCE_TOKENS_TOTAL = Counter(
    "harchos_inference_tokens_total",
    "Total tokens processed",
    ["model", "type"],
)

# Platform metrics
GPU_AVAILABLE = Gauge(
    "harchos_gpu_available",
    "Available GPUs by hub",
    ["hub_name", "gpu_type"],
)

GPU_TOTAL = Gauge(
    "harchos_gpu_total",
    "Total GPUs by hub",
    ["hub_name", "gpu_type"],
)

HUB_RENEWABLE_PERCENTAGE = Gauge(
    "harchos_hub_renewable_percentage",
    "Renewable energy percentage by hub",
    ["hub_name"],
)


class MetricsMiddleware:
    """Middleware that collects Prometheus metrics — pure ASGI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == "/v1/metrics":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        endpoint = self._normalize_endpoint(path)

        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        start_time = time.perf_counter()
        status_code = 0

        async def send_with_metrics(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_with_metrics)
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
            REQUEST_COUNT.labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    @staticmethod
    def _normalize_endpoint(path: str) -> str:
        """Normalize endpoint path to reduce cardinality."""
        parts = path.rstrip("/").split("/")
        normalized = []
        for part in parts:
            if len(part) == 36 and "-" in part and part.count("-") == 4:
                normalized.append(":id")
            elif part.isdigit():
                normalized.append(":num")
            else:
                normalized.append(part)
        return "/".join(normalized)


def get_metrics_response() -> StarletteResponse:
    """Generate Prometheus metrics response."""
    return StarletteResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
