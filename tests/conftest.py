"""Test configuration and fixtures."""

import hashlib
import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.database import init_db, async_session_factory, Base, engine
from app.main import app
from app.models.user import User
from app.models.api_key import ApiKey
from app.config import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create tables before each test and drop after."""
    await init_db()
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def seeded_db():
    """Seed the database with a test user and API key."""
    async with async_session_factory() as session:
        # Create test user
        user = User(
            email="test@harchos.ai",
            name="Test User",
            is_active=True,
        )
        session.add(user)
        await session.flush()

        # Create test API key
        test_key = settings.default_api_key
        key_hash = hashlib.sha256(test_key.encode()).hexdigest()
        key_prefix = test_key[:8]

        api_key = ApiKey(
            user_id=user.id,
            name="Test Key",
            key_hash=key_hash,
            key_prefix=key_prefix,
            is_active=True,
        )
        session.add(api_key)
        await session.commit()

        yield {"user": user, "api_key": api_key}


@pytest_asyncio.fixture
async def client():
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(seeded_db):
    """Headers with the test API key (requires seeded database)."""
    return {"X-API-Key": settings.default_api_key}
