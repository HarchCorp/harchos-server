"""FastAPI application with CORS, lifespan, security headers, and router aggregation."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.database import init_db, close_db
from app.api.router import api_router


# ---------------------------------------------------------------------------
# Timing middleware — adds X-Process-Time header to every response
# ---------------------------------------------------------------------------

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Add X-Process-Time-ms header to all responses for latency monitoring."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        return response


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if settings.enable_security_headers:
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    await init_db()
    # Auto-seed on first run
    from app.seed import seed
    await seed()

    # Initialize Redis cache (optional)
    from app.cache import cache
    if cache.is_available():
        import logging
        logging.getLogger("harchos.main").info("Redis cache: CONNECTED")
    else:
        import logging
        logging.getLogger("harchos.main").info("Redis cache: not configured (using in-memory fallback)")

    yield

    # Shutdown
    await close_db()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# Timing middleware (must be first to capture full request time)
app.add_middleware(ProcessTimeMiddleware)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)

# CORS — restrictive by default, configure via HARCHOS_CORS_ORIGINS
cors_origins = settings.cors_origins
if not cors_origins and settings.is_development:
    # Dev fallback: allow common local origins
    cors_origins = ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/v1", include_in_schema=False)
async def api_root():
    """API info endpoint."""
    return JSONResponse({
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/v1/health",
    })


# ---------------------------------------------------------------------------
# Mount the API router under /v1
# ---------------------------------------------------------------------------

app.include_router(api_router, prefix="/v1")
