"""Shared test fixtures for HarchOS Server test suite.

Provides:
- Test database (SQLite in-memory) with schema creation/drop per test
- Test app without lifespan (no seed, no WS background tasks)
- Auth fixtures: test user, API key, JWT token, auth headers
- Async test client via httpx AsyncClient
"""

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Override settings BEFORE importing app modules
# ---------------------------------------------------------------------------
os.environ.setdefault("HARCHOS_ENVIRONMENT", "dev")
os.environ.setdefault("HARCHOS_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("HARCHOS_DEFAULT_API_KEY", "hsk_test_default_key_for_pytest_1234")
os.environ.setdefault("HARCHOS_DATABASE_URL", "sqlite+aiosqlite://")
os.environ.setdefault("HARCHOS_INFERENCE_BACKEND_URL", "")
os.environ.setdefault("HARCHOS_ELECTRICITY_MAPS_API_KEY", "")

from app.config import settings  # noqa: E402
from app.database import Base, get_db  # noqa: E402
from app.models.user import User  # noqa: E402
from app.models.api_key import ApiKey  # noqa: E402
from app.models.hub import Hub  # noqa: E402
from app.models.workload import Workload  # noqa: E402
from app.models.carbon import CarbonIntensityRecord, CarbonOptimizationLog  # noqa: E402
from app.models.pricing import Pricing, BillingRecord  # noqa: E402
from app.models.model import Model  # noqa: E402
from app.models.energy import EnergyReport, EnergyConsumption  # noqa: E402
from app.services.auth_service import AuthService  # noqa: E402

# ---------------------------------------------------------------------------
# Test database engine (SQLite in-memory, shared per session via file URI)
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite+aiosqlite://"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    future=True,
)

TestSessionFactory = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Override get_db dependency to use test database
# ---------------------------------------------------------------------------
async def _get_test_db() -> AsyncSession:
    """Yield a test database session with automatic rollback on error."""
    async with TestSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Build a test FastAPI app that skips lifespan (no seed, no WS tasks)
# ---------------------------------------------------------------------------
def _create_test_app():
    """Create a FastAPI app instance for testing without lifespan side-effects.

    We include routers individually so we can give the metrics router
    a proper prefix (its endpoint uses an empty path which clashes with
    FastAPI's prefix+path validation when mounted without one).
    """
    from fastapi import FastAPI, APIRouter
    from fastapi.middleware.cors import CORSMiddleware

    from app.core.exceptions import HarchOSError, harchos_error_handler, unhandled_error_handler

    # Import all sub-routers
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
    from app.api.batch import router as batch_router
    from app.api.embeddings import router as embeddings_router
    from app.api.fine_tuning import router as fine_tuning_router
    from app.api.model_health import router as model_health_router

    @asynccontextmanager
    async def _test_lifespan(app: FastAPI):
        # Just create tables – no seed, no background tasks
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Mark startup as complete so /health/startup returns 200
        from app.api.health import mark_startup_complete
        mark_startup_complete()
        yield
        # Cleanup
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    test_app = FastAPI(
        title="HarchOS Test",
        version="0.6.0",
        lifespan=_test_lifespan,
    )

    # Register error handlers
    test_app.add_exception_handler(HarchOSError, harchos_error_handler)
    test_app.add_exception_handler(Exception, unhandled_error_handler)

    # Override DB dependency
    test_app.dependency_overrides[get_db] = _get_test_db

    # Build the API router ourselves to avoid the metrics empty-path issue
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
    api_router.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])
    api_router.include_router(batch_router, prefix="/inference", tags=["Batch Inference"])
    api_router.include_router(embeddings_router, prefix="/inference", tags=["Embeddings"])
    api_router.include_router(fine_tuning_router, prefix="/fine-tuning", tags=["Fine-Tuning"])
    api_router.include_router(model_health_router, tags=["Model Health"])

    test_app.include_router(api_router, prefix="/v1")

    # Allow CORS for test
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount WS router (may fail import, that's ok)
    try:
        from app.api.ws_monitoring import router as ws_router
        test_app.include_router(ws_router, prefix="/v1/ws")
    except ImportError:
        pass

    return test_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def _setup_tables():
    """Create all tables before each test, drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provide a test database session."""
    async with TestSessionFactory() as session:
        yield session


@pytest_asyncio.fixture
async def app():
    """Provide the test FastAPI app."""
    test_app = _create_test_app()
    yield test_app


@pytest_asyncio.fixture
async def client(app) -> AsyncClient:
    """Provide an async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create and return a test user."""
    user = User(
        email="testuser@harchos.ai",
        name="Test User",
        is_active=True,
        role="user",
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest_asyncio.fixture
async def test_admin(db_session: AsyncSession) -> User:
    """Create and return a test admin user."""
    admin = User(
        email="admin@harchos.ai",
        name="Admin User",
        is_active=True,
        role="admin",
    )
    db_session.add(admin)
    await db_session.flush()
    return admin


@pytest_asyncio.fixture
async def test_viewer(db_session: AsyncSession) -> User:
    """Create and return a test viewer user."""
    viewer = User(
        email="viewer@harchos.ai",
        name="Viewer User",
        is_active=True,
        role="viewer",
    )
    db_session.add(viewer)
    await db_session.flush()
    return viewer


@pytest_asyncio.fixture
async def test_api_key(db_session: AsyncSession, test_user: User) -> dict:
    """Create a test API key and return both the key object and raw key string."""
    raw_key = AuthService.generate_api_key()
    key_hash = AuthService.hash_key(raw_key)
    key_prefix = raw_key[:8]

    api_key = ApiKey(
        user_id=test_user.id,
        name="Test API Key",
        key_hash=key_hash,
        key_prefix=key_prefix,
        is_active=True,
    )
    db_session.add(api_key)
    await db_session.flush()
    return {"api_key_obj": api_key, "raw_key": raw_key, "user": test_user}


@pytest_asyncio.fixture
async def test_admin_api_key(db_session: AsyncSession, test_admin: User) -> dict:
    """Create a test API key for admin user."""
    raw_key = AuthService.generate_api_key()
    key_hash = AuthService.hash_key(raw_key)
    key_prefix = raw_key[:8]

    api_key = ApiKey(
        user_id=test_admin.id,
        name="Admin API Key",
        key_hash=key_hash,
        key_prefix=key_prefix,
        is_active=True,
    )
    db_session.add(api_key)
    await db_session.flush()
    return {"api_key_obj": api_key, "raw_key": raw_key, "user": test_admin}


@pytest_asyncio.fixture
async def test_viewer_api_key(db_session: AsyncSession, test_viewer: User) -> dict:
    """Create a test API key for viewer user."""
    raw_key = AuthService.generate_api_key()
    key_hash = AuthService.hash_key(raw_key)
    key_prefix = raw_key[:8]

    api_key = ApiKey(
        user_id=test_viewer.id,
        name="Viewer API Key",
        key_hash=key_hash,
        key_prefix=key_prefix,
        is_active=True,
    )
    db_session.add(api_key)
    await db_session.flush()
    return {"api_key_obj": api_key, "raw_key": raw_key, "user": test_viewer}


@pytest_asyncio.fixture
async def auth_headers(test_api_key: dict) -> dict:
    """Provide authentication headers with a valid API key."""
    return {"X-API-Key": test_api_key["raw_key"]}


@pytest_asyncio.fixture
async def admin_auth_headers(test_admin_api_key: dict) -> dict:
    """Provide authentication headers with an admin API key."""
    return {"X-API-Key": test_admin_api_key["raw_key"]}


@pytest_asyncio.fixture
async def viewer_auth_headers(test_viewer_api_key: dict) -> dict:
    """Provide authentication headers with a viewer API key."""
    return {"X-API-Key": test_viewer_api_key["raw_key"]}


@pytest_asyncio.fixture
async def test_jwt_token(test_api_key: dict) -> str:
    """Create a JWT token for the test user."""
    token_resp = AuthService.create_jwt_token(
        api_key_id=test_api_key["api_key_obj"].id,
        user_id=test_api_key["user"].id,
    )
    return token_resp.access_token


@pytest_asyncio.fixture
async def bearer_auth_headers(test_jwt_token: str) -> dict:
    """Provide authentication headers with a Bearer JWT token."""
    return {"Authorization": f"Bearer {test_jwt_token}"}


@pytest_asyncio.fixture
async def test_hub(db_session: AsyncSession) -> Hub:
    """Create a test hub."""
    hub = Hub(
        name="Test Hub Ouarzazate",
        region="Draa-Tafilalet",
        status="ready",
        tier="enterprise",
        total_gpus=100,
        available_gpus=80,
        total_cpu_cores=800,
        available_cpu_cores=640,
        total_memory_gb=3200.0,
        available_memory_gb=2560.0,
        total_storage_gb=50000.0,
        available_storage_gb=35000.0,
        latitude=30.9189,
        longitude=-6.9000,
        city="Ouarzazate",
        country="Morocco",
        renewable_percentage=97.2,
        grid_carbon_intensity=18.0,
        pue=1.04,
        sovereignty_level="strict",
        data_residency_policy="local_only",
    )
    db_session.add(hub)
    await db_session.flush()
    return hub
