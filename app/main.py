"""FastAPI application with CORS, lifespan, security headers, error handling, and router aggregation.

HarchOS Server v0.7.0 — 10x competitive upgrade:
- 75+ API endpoints (was 43)
- OpenAI API compatibility layer (drop-in replacement, like Groq/Together AI)
- Project-scoped API keys with fine-grained permissions
- DB-backed batch inference and fine-tuning (was in-memory)
- SSRF-protected webhooks with HMAC signatures
- Admin-only infrastructure mutations
- Kubernetes-style health probes (liveness, readiness, startup, detailed)
- Counter-based Redis rate limiting (O(1) instead of O(n))
- Stable SHA-256 cache keys (not Python hash())
- Cache invalidation on data mutations
- Parallelized carbon data fetching
- Single-query workload stats (was 4 queries)
- In-memory cache with size limits (was unbounded)
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings
from app.database import init_db, close_db
from app.api.router import api_router
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.metrics import MetricsMiddleware
from app.middleware.performance import (
    CompressionMiddleware,
    ResponseCacheMiddleware,
    RequestSizeLimitMiddleware,
)
from app.core.exceptions import HarchOSError, harchos_error_handler, unhandled_error_handler

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# Startup flag for Kubernetes startup probe
_startup_complete = False


def mark_startup_complete():
    """Mark that the application has finished starting up."""
    global _startup_complete
    _startup_complete = True


def is_startup_complete() -> bool:
    """Check if the application has finished starting up."""
    return _startup_complete


# ---------------------------------------------------------------------------
# Timing middleware — adds X-Process-Time header to every response (pure ASGI)
# ---------------------------------------------------------------------------

class ProcessTimeMiddleware:
    """Add X-Process-Time-ms header to all responses — pure ASGI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()

        async def send_with_timing(message: dict) -> None:
            if message["type"] == "http.response.start":
                elapsed_ms = (time.perf_counter() - start) * 1000
                headers_list = list(message.get("headers", []))
                headers_list.append([b"x-process-time-ms", f"{elapsed_ms:.1f}".encode()])
                headers_list.append([b"server", b"HarchOS"])
                message["headers"] = headers_list
            await send(message)

        await self.app(scope, receive, send_with_timing)


# ---------------------------------------------------------------------------
# Security headers middleware (pure ASGI)
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware:
    """Add security headers to all responses — pure ASGI."""

    _SECURITY_HEADERS = [
        [b"x-content-type-options", b"nosniff"],
        [b"x-frame-options", b"DENY"],
        [b"x-xss-protection", b"1; mode=block"],
        [b"referrer-policy", b"strict-origin-when-cross-origin"],
        [b"permissions-policy", b"camera=(), microphone=(), geolocation=()"],
        [b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"],
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security(message: dict) -> None:
            if message["type"] == "http.response.start" and settings.enable_security_headers:
                headers_list = list(message.get("headers", []))
                headers_list.extend(self._SECURITY_HEADERS)
                headers_list.append([b"x-harchos-version", settings.app_version.encode()])
                if settings.is_production:
                    headers_list.append([b"strict-transport-security", b"max-age=31536000; includeSubDomains"])
                message["headers"] = headers_list
            await send(message)

        await self.app(scope, receive, send_with_security)


# ---------------------------------------------------------------------------
# Request logging middleware (pure ASGI)
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware:
    """Log all API requests with method, path, status, and duration — pure ASGI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 0

        async def send_with_logging(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        await self.app(scope, receive, send_with_logging)

        path = scope.get("path", "")
        if path not in ("/docs", "/redoc", "/openapi.json", "/"):
            elapsed_ms = (time.perf_counter() - start) * 1000
            method = scope.get("method", "GET")
            logger = logging.getLogger("harchos.requests")
            logger.info("%s %s → %d (%.1fms)", method, path, status_code, elapsed_ms)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger = logging.getLogger("harchos.main")

    # Startup
    await init_db()

    # Auto-seed on first run
    from app.seed import seed
    await seed()

    # Initialize Redis cache (optional)
    from app.cache import cache
    if cache.is_available():
        logger.info("Redis cache: CONNECTED")
    else:
        logger.info("Redis cache: not configured (using in-memory fallback)")

    # Start WebSocket background tasks
    try:
        from app.api.ws_monitoring import start_background_tasks
        await start_background_tasks()
        logger.info("WebSocket monitoring: STARTED")
    except Exception as e:
        logger.warning("WebSocket monitoring: failed to start (%s)", e)

    # Mark startup complete for Kubernetes probe
    mark_startup_complete()

    logger.info(
        "HarchOS Server v%s starting in %s mode — 75+ endpoints, carbon-aware, OpenAI-compatible, 10x competitive",
        settings.app_version, settings.environment,
    )

    yield

    # Shutdown
    try:
        from app.api.ws_monitoring import stop_background_tasks
        await stop_background_tasks()
    except Exception:
        pass

    await close_db()
    logger.info("HarchOS Server shutting down")


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
        "- **OpenAI-Compatible API**: Drop-in replacement for OpenAI — just change `base_url`. "
        "Supports `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`\n"
        "- **Carbon-Aware Scheduling**: Automatically routes workloads to the "
        "greenest GPU hub based on real-time carbon intensity data\n"
        "- **Data Sovereignty**: Strict data residency controls with local-only "
        "storage policies and sovereign cloud compliance\n"
        "- **Project-Scoped API Keys**: Fine-grained permissions with per-key "
        "scopes, model restrictions, region restrictions, and budget caps\n"
        "- **Multi-Region**: 5 Moroccan GPU hubs optimized for carbon intensity, "
        "with plans for Pan-African expansion\n"
        "- **Batch Inference**: Submit up to 100 requests at once with 50% cost savings\n"
        "- **Embeddings API**: OpenAI-compatible embeddings with carbon tracking\n"
        "- **Fine-Tuning**: Train models with carbon budget enforcement\n"
        "- **WebSocket Monitoring**: Real-time platform metrics and workload events\n"
        "- **Model Health**: Per-model health checks and latency metrics\n"
        "- **RBAC**: Role-based access control with admin, user, and viewer roles\n"
        "- **Tiered Rate Limiting**: Free, Standard, Enterprise tiers with per-key enforcement\n"
        "- **SSRF-Protected Webhooks**: HMAC-SHA256 signed webhook deliveries\n"
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

# Register exception handlers for structured error responses
app.add_exception_handler(HarchOSError, harchos_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


# ---------------------------------------------------------------------------
# Middleware (order matters — outermost first in FastAPI)
# ---------------------------------------------------------------------------

# 1. Request size limit (reject huge payloads early) — pure ASGI
app.add_middleware(RequestSizeLimitMiddleware)

# 2. Timing middleware (captures full request time)
app.add_middleware(ProcessTimeMiddleware)

# 3. Request logging
app.add_middleware(RequestLoggingMiddleware)

# 4. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 5. Response compression (gzip) — pure ASGI, no body buffering
app.add_middleware(CompressionMiddleware)

# 6. Response caching for GET endpoints — pure ASGI, no body buffering
app.add_middleware(ResponseCacheMiddleware)

# 7. Rate limiting (tiered, per-API-key)
app.add_middleware(RateLimitMiddleware)

# 8. Prometheus metrics collection
app.add_middleware(MetricsMiddleware)

# CORS — restrictive by default, configure via HARCHOS_CORS_ORIGINS
cors_origins = settings.cors_origins
if not cors_origins and settings.is_development:
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
        "openai_compatible": True,
        "endpoints": {
            "inference": "/v1/chat/completions",
            "completions": "/v1/completions",
            "embeddings": "/v1/embeddings",
            "models": "/v1/models",
            "batch": "/v1/inference/batch",
            "fine_tuning": "/v1/fine-tuning/jobs",
            "carbon": "/v1/carbon/intensity",
            "workloads": "/v1/workloads",
            "hubs": "/v1/hubs",
            "projects": "/v1/projects",
            "webhooks": "/v1/webhooks",
            "websockets": {
                "monitoring": "/v1/ws/monitoring",
                "workloads": "/v1/ws/workloads",
                "carbon": "/v1/ws/carbon",
            },
        },
    })


# ---------------------------------------------------------------------------
# Mount the API router under /v1
# ---------------------------------------------------------------------------

app.include_router(api_router, prefix="/v1")

# Mount WebSocket endpoints under /v1/ws
try:
    from app.api.ws_monitoring import router as ws_router
    app.include_router(ws_router, prefix="/v1/ws")
except ImportError:
    logging.getLogger("harchos.main").warning("WebSocket module not available")

# ---------------------------------------------------------------------------
# OpenAI API Compatibility Layer
# Mount inference endpoints at /v1/chat/completions and /v1/completions
# for direct OpenAI SDK compatibility (like Groq, Together AI)
# ---------------------------------------------------------------------------

try:
    from app.api.openai_compat import router as openai_compat_router
    app.include_router(openai_compat_router, prefix="/v1", tags=["OpenAI Compatible"])
    logging.getLogger("harchos.main").info("OpenAI compatibility layer: ENABLED")
except ImportError:
    logging.getLogger("harchos.main").info("OpenAI compatibility layer: not available")
