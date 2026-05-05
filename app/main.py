"""FastAPI application with CORS, lifespan, and router aggregation."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    await init_db()
    # Auto-seed on first run
    from app.seed import seed
    await seed()
    yield
    # Shutdown


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "HarchOS Server – the carbon-aware, sovereignty-first GPU orchestration "
        "platform by HarchCorp. Built on Morocco's renewable energy advantage "
        "to deliver the greenest AI compute on the planet.\n\n"
        "## Key Features\n"
        "- **Carbon-Aware Scheduling**: Automatically routes workloads to the "
        "greenest GPU hub based on real-time carbon intensity data\n"
        "- **Data Sovereignty**: Strict data residency controls with local-only "
        "storage policies and sovereign cloud compliance\n"
        "- **Multi-Region**: 5 Moroccan GPU hubs optimized for carbon intensity, "
        "with plans for Pan-African expansion\n"
        "- **Tiered Pricing**: Enterprise, Performance, and Standard tiers with "
        "transparent per-GPU-hour billing in USD, MAD, and EUR\n"
    ),
    lifespan=lifespan,
    contact={
        "name": "HarchCorp",
        "url": "https://harchcorp.vercel.app",
        "email": "contact@harchcorp.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    servers=[
        {
            "url": "https://api.harchos.ai",
            "description": "Production server",
        },
        {
            "url": "https://staging-api.harchos.ai",
            "description": "Staging server",
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
    ],
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API router under /v1
app.include_router(api_router, prefix="/v1")
