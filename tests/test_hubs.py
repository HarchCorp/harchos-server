"""Tests for hub endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_hubs_public(client: AsyncClient):
    """Test that listing hubs is public (no auth required)."""
    response = await client.get("/v1/hubs")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "pagination" in data


@pytest.mark.asyncio
async def test_create_hub(client: AsyncClient, auth_headers: dict):
    """Test creating a hub."""
    payload = {
        "name": "Test Hub",
        "region": "Test Region",
        "tier": "standard",
        "capacity": {"total_gpus": 32, "available_gpus": 16, "total_cpu_cores": 256, "available_cpu_cores": 128, "total_memory_gb": 1024.0, "available_memory_gb": 512.0},
        "location": {"latitude": 33.5, "longitude": -7.5, "city": "Test City", "country": "Morocco"},
        "energy": {"renewable_percentage": 80.0, "grid_carbon_intensity": 50.0, "pue": 1.1},
    }
    response = await client.post("/v1/hubs", json=payload, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    # Response uses Kubernetes-style nested structure
    assert data["spec"]["name"] == "Test Hub"
    assert data["status"] == "creating"
    assert data["capacity"]["total_gpus"] == 32


@pytest.mark.asyncio
async def test_get_hub_capacity(client: AsyncClient, auth_headers: dict):
    """Test getting hub capacity."""
    payload = {"name": "Capacity Test Hub", "region": "Test"}
    create_resp = await client.post("/v1/hubs", json=payload, headers=auth_headers)
    hub_id = create_resp.json()["metadata"]["id"]

    response = await client.get(f"/v1/hubs/{hub_id}/capacity", headers=auth_headers)
    assert response.status_code == 200
    assert "total_gpus" in response.json()
