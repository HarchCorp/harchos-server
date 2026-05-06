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
from app.api.webhooks import router as webhooks_router

# New 10x endpoints — matching/beating all competitors
from app.api.batch import router as batch_router
from app.api.embeddings import router as embeddings_router
from app.api.fine_tuning import router as fine_tuning_router
from app.api.model_health import router as model_health_router

# Project-scoped API keys — Together AI-level security feature
from app.api.projects import router as projects_router

api_router = APIRouter()

# Core endpoints
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
api_router.include_router(metrics_router, tags=["Metrics"])
api_router.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])

# 10x endpoints — features no single competitor has all of
api_router.include_router(batch_router, prefix="/inference", tags=["Batch Inference"])
api_router.include_router(embeddings_router, prefix="/inference", tags=["Embeddings"])
api_router.include_router(fine_tuning_router, prefix="/fine-tuning", tags=["Fine-Tuning"])
api_router.include_router(model_health_router, tags=["Model Health"])

# Project-scoped API keys
api_router.include_router(projects_router, prefix="/v1/projects", tags=["Projects"])
