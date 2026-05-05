"""Pydantic schemas package."""

from app.schemas.common import PaginationMeta, PaginatedResponse
from app.schemas.workload import (
    WorkloadCreate,
    WorkloadUpdate,
    WorkloadResponse,
    WorkloadCompute,
    WorkloadSovereignty,
    WorkloadCarbonMetrics,
    WorkloadMetadata,
)
from app.schemas.hub import (
    HubCreate,
    HubUpdate,
    HubResponse,
    HubCapacity,
    HubLocation,
    HubEnergy,
)
from app.schemas.model import (
    ModelCreate,
    ModelUpdate,
    ModelResponse,
)
from app.schemas.energy import (
    EnergyReportResponse,
    EnergySummaryResponse,
    GreenWindowResponse,
    EnergyConsumptionResponse,
)
from app.schemas.carbon import (
    CarbonIntensityZoneResponse,
    CarbonIntensityZoneListResponse,
    CarbonOptimalHubRequest,
    CarbonOptimalHubResponse,
    CarbonOptimizeRequest,
    CarbonOptimizeResponse,
    CarbonMetricsResponse,
    CarbonDashboardResponse,
    CarbonForecastResponse,
)
from app.schemas.auth import (
    ApiKeyCreate,
    ApiKeyResponse,
    TokenResponse,
    UserInfo,
)

__all__ = [
    "PaginationMeta",
    "PaginatedResponse",
    "WorkloadCreate",
    "WorkloadUpdate",
    "WorkloadResponse",
    "WorkloadCompute",
    "WorkloadSovereignty",
    "WorkloadCarbonMetrics",
    "WorkloadMetadata",
    "HubCreate",
    "HubUpdate",
    "HubResponse",
    "HubCapacity",
    "HubLocation",
    "HubEnergy",
    "ModelCreate",
    "ModelUpdate",
    "ModelResponse",
    "EnergyReportResponse",
    "EnergySummaryResponse",
    "GreenWindowResponse",
    "EnergyConsumptionResponse",
    "CarbonIntensityZoneResponse",
    "CarbonIntensityZoneListResponse",
    "CarbonOptimalHubRequest",
    "CarbonOptimalHubResponse",
    "CarbonOptimizeRequest",
    "CarbonOptimizeResponse",
    "CarbonMetricsResponse",
    "CarbonDashboardResponse",
    "CarbonForecastResponse",
    "ApiKeyCreate",
    "ApiKeyResponse",
    "TokenResponse",
    "UserInfo",
]
