"""Tests for health check endpoints.

Covers:
- GET /v1/health — lightweight liveness check
- GET /v1/health/detailed — comprehensive readiness check with component health
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_healthy(client: AsyncClient):
    """GET /v1/health returns status healthy."""
    response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


@pytest.mark.asyncio
async def test_health_has_version(client: AsyncClient):
    """GET /v1/health includes the app version."""
    response = await client.get("/v1/health")
    data = response.json()
    assert data["version"] == "0.6.0"


@pytest.mark.asyncio
async def test_health_has_environment(client: AsyncClient):
    """GET /v1/health includes the environment name."""
    response = await client.get("/v1/health")
    data = response.json()
    assert data["environment"] in ("dev", "staging", "production")


@pytest.mark.asyncio
async def test_health_no_auth_required(client: AsyncClient):
    """GET /v1/health does not require authentication."""
    response = await client.get("/v1/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_detailed_health_returns_status(client: AsyncClient):
    """GET /v1/health/detailed returns a top-level status."""
    response = await client.get("/v1/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "version" in data
    assert "environment" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_detailed_health_has_components(client: AsyncClient):
    """GET /v1/health/detailed includes component health data."""
    response = await client.get("/v1/health/detailed")
    data = response.json()
    assert "components" in data
    components = data["components"]
    assert isinstance(components, dict)


@pytest.mark.asyncio
async def test_detailed_health_database_component(client: AsyncClient):
    """GET /v1/health/detailed includes database component status."""
    response = await client.get("/v1/health/detailed")
    data = response.json()
    components = data["components"]
    assert "database" in components
    db = components["database"]
    assert "status" in db
    assert db["status"] in ("healthy", "degraded", "unhealthy")


@pytest.mark.asyncio
async def test_detailed_health_cache_component(client: AsyncClient):
    """GET /v1/health/detailed includes cache component status."""
    response = await client.get("/v1/health/detailed")
    data = response.json()
    components = data["components"]
    assert "cache" in components
    cache = components["cache"]
    assert cache["status"] in ("healthy", "degraded", "unhealthy")


@pytest.mark.asyncio
async def test_detailed_health_carbon_api_component(client: AsyncClient):
    """GET /v1/health/detailed includes carbon_api component status."""
    response = await client.get("/v1/health/detailed")
    data = response.json()
    components = data["components"]
    assert "carbon_api" in components
    carbon = components["carbon_api"]
    assert carbon["status"] in ("healthy", "degraded", "unhealthy")


@pytest.mark.asyncio
async def test_detailed_health_uptime_positive(client: AsyncClient):
    """GET /v1/health/detailed returns positive uptime_seconds."""
    response = await client.get("/v1/health/detailed")
    data = response.json()
    assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_detailed_health_healthy_db_has_latency(client: AsyncClient):
    """When database is healthy, it reports latency_ms."""
    response = await client.get("/v1/health/detailed")
    data = response.json()
    db = data["components"].get("database", {})
    if db.get("status") == "healthy":
        assert "latency_ms" in db
        assert db["latency_ms"] is not None
        assert db["latency_ms"] >= 0
