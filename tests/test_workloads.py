"""Tests for workload endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_workloads_unauthorized(client: AsyncClient):
    """Test that listing workloads requires auth."""
    response = await client.get("/v1/workloads")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_workloads_authorized(client: AsyncClient, auth_headers: dict):
    """Test listing workloads with valid auth."""
    response = await client.get("/v1/workloads", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "pagination" in data
    assert data["items"] == []


@pytest.mark.asyncio
async def test_create_workload(client: AsyncClient, auth_headers: dict):
    """Test creating a workload."""
    payload = {
        "name": "Test Training Job",
        "type": "training",
        "compute": {"gpu_count": 4, "gpu_type": "A100", "cpu_cores": 32, "memory_gb": 128.0, "storage_gb": 500.0},
        "priority": "high",
    }
    response = await client.post("/v1/workloads", json=payload, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Training Job"
    assert data["type"] == "training"
    assert data["status"] == "pending"
    assert data["priority"] == "high"
    assert data["compute"]["gpu_count"] == 4
    assert "id" in data
    assert "metadata" in data


@pytest.mark.asyncio
async def test_get_workload(client: AsyncClient, auth_headers: dict):
    """Test getting a workload by ID."""
    # Create first
    payload = {"name": "Get Test", "type": "inference"}
    create_resp = await client.post("/v1/workloads", json=payload, headers=auth_headers)
    workload_id = create_resp.json()["id"]

    # Get it
    response = await client.get(f"/v1/workloads/{workload_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"] == workload_id


@pytest.mark.asyncio
async def test_update_workload(client: AsyncClient, auth_headers: dict):
    """Test updating a workload."""
    payload = {"name": "Update Test", "type": "fine_tuning"}
    create_resp = await client.post("/v1/workloads", json=payload, headers=auth_headers)
    workload_id = create_resp.json()["id"]

    response = await client.patch(
        f"/v1/workloads/{workload_id}",
        json={"status": "running"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "running"


@pytest.mark.asyncio
async def test_delete_workload(client: AsyncClient, auth_headers: dict):
    """Test deleting a workload."""
    payload = {"name": "Delete Test", "type": "batch"}
    create_resp = await client.post("/v1/workloads", json=payload, headers=auth_headers)
    workload_id = create_resp.json()["id"]

    response = await client.delete(f"/v1/workloads/{workload_id}", headers=auth_headers)
    assert response.status_code == 204
