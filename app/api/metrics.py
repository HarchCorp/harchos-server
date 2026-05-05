"""Prometheus metrics endpoint for monitoring."""

from fastapi import APIRouter, Depends
from starlette.responses import Response

from app.models.api_key import ApiKey
from app.api.deps import require_auth
from app.middleware.metrics import get_metrics_response

router = APIRouter()


@router.get("", response_class=Response)
async def prometheus_metrics(
    api_key: ApiKey = Depends(require_auth),
):
    """Expose Prometheus metrics for monitoring and alerting.
    
    Returns metrics in Prometheus exposition format including:
    - HTTP request counts and latencies
    - Carbon intensity by zone
    - Inference request counts and durations
    - GPU availability by hub
    - Carbon savings counter
    
    Requires authentication to prevent public exposure of metrics.
    """
    return get_metrics_response()
