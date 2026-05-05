"""Main API router aggregating all sub-routers."""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.auth import router as auth_router
from app.api.workloads import router as workloads_router
from app.api.hubs import router as hubs_router
from app.api.models import router as models_router
from app.api.energy import router as energy_router
from app.api.carbon import router as carbon_router
from app.api.pricing import router as pricing_router
from app.api.regions import router as regions_router
from app.api.monitoring import router as monitoring_router
from app.api.inference import router as inference_router
from app.api.metrics import router as metrics_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(auth_router, prefix="/auth", tags=["Auth"])
api_router.include_router(workloads_router, prefix="/workloads", tags=["Workloads"])
api_router.include_router(hubs_router, prefix="/hubs", tags=["Hubs"])
api_router.include_router(models_router, prefix="/models", tags=["Models"])
api_router.include_router(energy_router, prefix="/energy", tags=["Energy"])
api_router.include_router(carbon_router, prefix="/carbon", tags=["Carbon"])
api_router.include_router(pricing_router, prefix="/pricing", tags=["Pricing"])
api_router.include_router(regions_router, prefix="/regions", tags=["Regions"])
api_router.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
api_router.include_router(inference_router, prefix="/inference", tags=["Inference"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
