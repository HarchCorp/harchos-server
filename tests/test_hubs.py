"""Tests for hub endpoints.

Covers:
- List hubs (public, no auth required)
- Create hub (requires auth)
- Hub capacity endpoint
- Get specific hub
- Hub filtering by status, tier, region
"""

import pytest
from httpx import AsyncClient


VALID_HUB = {
    "name": "Test Hub",
    "region": "Draa-Tafilalet",
    "tier": "enterprise",
    "sovereignty_level": "strict",
    "gpu_types": ["h100"],
    "auto_scale": True,
    "min_gpu_count": 0,
    "max_gpu_count": 16,
    "carbon_aware_scheduling": True,
}


@pytest.mark.asyncio
async def test_list_hubs_public(client: AsyncClient, test_hub):
    """GET /v1/hubs is accessible without authentication."""
    response = await client.get("/v1/hubs")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "pagination" in data
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_list_hubs_includes_test_hub(client: AsyncClient, test_hub):
    """Listed hubs include the test hub."""
    response = await client.get("/v1/hubs")
    data = response.json()
    hub_names = [h["metadata"]["name"] for h in data["items"]]
    assert "Test Hub Ouarzazate" in hub_names


@pytest.mark.asyncio
async def test_list_hubs_pagination(client: AsyncClient, test_hub):
    """GET /v1/hubs supports pagination."""
    response = await client.get(
        "/v1/hubs",
        params={"page": 1, "per_page": 10},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["pagination"]["per_page"] == 10


@pytest.mark.asyncio
async def test_list_hubs_filter_by_status(client: AsyncClient, test_hub):
    """GET /v1/hubs supports status filter."""
    response = await client.get(
        "/v1/hubs",
        params={"status": "ready"},
    )
    assert response.status_code == 200
    data = response.json()
    for hub in data["items"]:
        assert hub["status"] == "ready"


@pytest.mark.asyncio
async def test_list_hubs_filter_by_tier(client: AsyncClient, test_hub):
    """GET /v1/hubs supports tier filter."""
    response = await client.get(
        "/v1/hubs",
        params={"tier": "enterprise"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_hubs_filter_by_region(client: AsyncClient, test_hub):
    """GET /v1/hubs supports region filter."""
    response = await client.get(
        "/v1/hubs",
        params={"region": "Draa"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_create_hub(client: AsyncClient, auth_headers: dict):
    """POST /v1/hubs creates a new hub (requires auth)."""
    response = await client.post(
        "/v1/hubs",
        json=VALID_HUB,
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert "metadata" in data
    assert data["metadata"]["name"] == "Test Hub"
    assert data["spec"]["region"] == "Draa-Tafilalet"
    assert data["status"] == "creating"


@pytest.mark.asyncio
async def test_create_hub_requires_auth(client: AsyncClient):
    """POST /v1/hubs requires authentication."""
    response = await client.post("/v1/hubs", json=VALID_HUB)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_hub_with_capacity(client: AsyncClient, auth_headers: dict):
    """POST /v1/hubs with capacity info."""
    payload = {
        **VALID_HUB,
        "capacity": {
            "total_gpus": 50,
            "available_gpus": 40,
            "total_cpu_cores": 400,
            "available_cpu_cores": 320,
            "total_memory_gb": 1600.0,
            "available_memory_gb": 1280.0,
            "total_storage_gb": 25000.0,
            "available_storage_gb": 20000.0,
        },
    }
    response = await client.post(
        "/v1/hubs",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["capacity"]["total_gpus"] == 50


@pytest.mark.asyncio
async def test_get_hub(client: AsyncClient, test_hub):
    """GET /v1/hubs/{hub_id} returns a specific hub."""
    response = await client.get(f"/v1/hubs/{test_hub.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["id"] == test_hub.id
    assert data["metadata"]["name"] == "Test Hub Ouarzazate"


@pytest.mark.asyncio
async def test_get_hub_public(client: AsyncClient, test_hub):
    """GET /v1/hubs/{hub_id} does not require authentication."""
    response = await client.get(f"/v1/hubs/{test_hub.id}")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_nonexistent_hub(client: AsyncClient):
    """GET /v1/hubs/{hub_id} with invalid ID returns 404."""
    response = await client.get("/v1/hubs/nonexistent-hub-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_hub_capacity_endpoint(client: AsyncClient, test_hub):
    """GET /v1/hubs/{hub_id}/capacity returns capacity info."""
    response = await client.get(f"/v1/hubs/{test_hub.id}/capacity")
    assert response.status_code == 200
    data = response.json()
    assert "total_gpus" in data
    assert "available_gpus" in data
    assert "total_cpu_cores" in data
    assert "total_memory_gb" in data
    assert "total_storage_gb" in data
    assert data["total_gpus"] == 100
    assert data["available_gpus"] == 80


@pytest.mark.asyncio
async def test_hub_capacity_public(client: AsyncClient, test_hub):
    """Hub capacity endpoint is accessible without auth."""
    response = await client.get(f"/v1/hubs/{test_hub.id}/capacity")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_hub_capacity_not_found(client: AsyncClient):
    """Hub capacity for nonexistent hub returns 404."""
    response = await client.get("/v1/hubs/nonexistent/capacity")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_hub_response_includes_carbon_metrics(client: AsyncClient, test_hub):
    """Hub response includes carbon metrics."""
    response = await client.get(f"/v1/hubs/{test_hub.id}")
    data = response.json()
    assert "carbon_metrics" in data
    cm = data["carbon_metrics"]
    assert "co2_grams" in cm
    assert "energy_kwh" in cm
    assert "pue" in cm
    assert "region_grid_intensity" in cm
    assert "renewable_percentage" in cm


@pytest.mark.asyncio
async def test_hub_response_includes_spec(client: AsyncClient, test_hub):
    """Hub response includes spec with sovereignty info."""
    response = await client.get(f"/v1/hubs/{test_hub.id}")
    data = response.json()
    assert "spec" in data
    spec = data["spec"]
    assert spec["region"] == "Draa-Tafilalet"
    assert spec["tier"] == "enterprise"


@pytest.mark.asyncio
async def test_hub_response_includes_endpoint(client: AsyncClient, test_hub):
    """Hub response includes an endpoint URL."""
    response = await client.get(f"/v1/hubs/{test_hub.id}")
    data = response.json()
    assert "endpoint" in data
    assert data["endpoint"].startswith("https://")
