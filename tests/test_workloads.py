"""Tests for workload endpoints.

Covers:
- CRUD operations (create, read, update, delete)
- Pagination
- Validation errors
- RBAC enforcement (user can only see own, admin sees all)
"""

import pytest
from httpx import AsyncClient


VALID_WORKLOAD = {
    "name": "Test Workload",
    "type": "training",
    "compute": {
        "gpu_count": 2,
        "gpu_type": "h100",
        "cpu_cores": 8,
        "memory_gb": 32.0,
        "storage_gb": 100.0,
    },
    "priority": "normal",
}


@pytest.mark.asyncio
async def test_create_workload(client: AsyncClient, auth_headers: dict):
    """POST /v1/workloads creates a new workload."""
    response = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert "metadata" in data
    assert data["metadata"]["name"] == "Test Workload"
    assert data["spec"]["type"] == "training"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_create_workload_with_hub(client: AsyncClient, auth_headers: dict, test_hub):
    """POST /v1/workloads with hub_id specified."""
    payload = {**VALID_WORKLOAD, "hub_id": test_hub.id}
    response = await client.post(
        "/v1/workloads",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["hub_id"] == test_hub.id


@pytest.mark.asyncio
async def test_create_workload_requires_auth(client: AsyncClient):
    """POST /v1/workloads requires authentication."""
    response = await client.post("/v1/workloads", json=VALID_WORKLOAD)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_create_workload_viewer_denied(client: AsyncClient, viewer_auth_headers: dict):
    """Viewer users cannot create workloads (requires write access)."""
    response = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=viewer_auth_headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_workloads(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads returns paginated list."""
    # Create a workload first
    await client.post("/v1/workloads", json=VALID_WORKLOAD, headers=auth_headers)

    response = await client.get("/v1/workloads", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "pagination" in data
    assert len(data["items"]) >= 1


@pytest.mark.asyncio
async def test_list_workloads_pagination(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads supports pagination params."""
    # Create workloads
    for i in range(3):
        payload = {**VALID_WORKLOAD, "name": f"Workload {i}"}
        await client.post("/v1/workloads", json=payload, headers=auth_headers)

    response = await client.get(
        "/v1/workloads",
        params={"page": 1, "per_page": 2},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) <= 2
    assert data["pagination"]["per_page"] == 2


@pytest.mark.asyncio
async def test_list_workloads_filter_by_status(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads supports status filter."""
    await client.post("/v1/workloads", json=VALID_WORKLOAD, headers=auth_headers)

    response = await client.get(
        "/v1/workloads",
        params={"status": "pending"},
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_workloads_invalid_status_filter(client: AsyncClient, auth_headers: dict):
    """Invalid status filter returns validation error."""
    response = await client.get(
        "/v1/workloads",
        params={"status": "invalid_status"},
        headers=auth_headers,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_workload(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads/{id} returns a specific workload."""
    create_resp = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=auth_headers,
    )
    workload_id = create_resp.json()["metadata"]["id"]

    response = await client.get(f"/v1/workloads/{workload_id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["id"] == workload_id


@pytest.mark.asyncio
async def test_get_nonexistent_workload(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads/{id} with invalid ID returns 404."""
    response = await client.get(
        "/v1/workloads/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_workload(client: AsyncClient, auth_headers: dict):
    """PATCH /v1/workloads/{id} updates a workload."""
    create_resp = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=auth_headers,
    )
    workload_id = create_resp.json()["metadata"]["id"]

    response = await client.patch(
        f"/v1/workloads/{workload_id}",
        json={"status": "running"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_update_workload_name(client: AsyncClient, auth_headers: dict):
    """PATCH /v1/workloads/{id} can update the name."""
    create_resp = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=auth_headers,
    )
    workload_id = create_resp.json()["metadata"]["id"]

    response = await client.patch(
        f"/v1/workloads/{workload_id}",
        json={"name": "Updated Workload Name"},
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_delete_workload(client: AsyncClient, auth_headers: dict):
    """DELETE /v1/workloads/{id} deletes a workload."""
    create_resp = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=auth_headers,
    )
    workload_id = create_resp.json()["metadata"]["id"]

    response = await client.delete(
        f"/v1/workloads/{workload_id}",
        headers=auth_headers,
    )
    assert response.status_code == 204

    # Verify deletion
    get_resp = await client.get(f"/v1/workloads/{workload_id}", headers=auth_headers)
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_workload(client: AsyncClient, auth_headers: dict):
    """DELETE /v1/workloads/{id} with invalid ID returns 404."""
    response = await client.delete(
        "/v1/workloads/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_workload_stats(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads/stats returns workload statistics."""
    await client.post("/v1/workloads", json=VALID_WORKLOAD, headers=auth_headers)

    response = await client.get("/v1/workloads/stats", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "active" in data
    assert "completed" in data
    assert "failed" in data


@pytest.mark.asyncio
async def test_create_workload_invalid_type(client: AsyncClient, auth_headers: dict):
    """Invalid workload type is rejected."""
    payload = {**VALID_WORKLOAD, "type": "invalid_type"}
    response = await client.post(
        "/v1/workloads",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_create_workload_invalid_priority(client: AsyncClient, auth_headers: dict):
    """Invalid priority is rejected."""
    payload = {**VALID_WORKLOAD, "priority": "super_urgent"}
    response = await client.post(
        "/v1/workloads",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_create_workload_invalid_gpu_type(client: AsyncClient, auth_headers: dict):
    """Invalid GPU type is rejected."""
    payload = {
        **VALID_WORKLOAD,
        "compute": {
            "gpu_count": 2,
            "gpu_type": "FAKE_GPU_9000",
            "cpu_cores": 8,
            "memory_gb": 32.0,
            "storage_gb": 100.0,
        },
    }
    response = await client.post(
        "/v1/workloads",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_create_workload_empty_name(client: AsyncClient, auth_headers: dict):
    """Empty workload name is rejected."""
    payload = {**VALID_WORKLOAD, "name": ""}
    response = await client.post(
        "/v1/workloads",
        json=payload,
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_user_sees_own_workloads_only(
    client: AsyncClient, auth_headers: dict, viewer_auth_headers: dict
):
    """Regular users only see their own workloads."""
    # User A creates a workload
    await client.post("/v1/workloads", json=VALID_WORKLOAD, headers=auth_headers)

    # User B lists workloads — should not see User A's
    response = await client.get("/v1/workloads", headers=viewer_auth_headers)
    data = response.json()
    user_b_workloads = [w for w in data["items"] if w["metadata"].get("name") == "Test Workload"]
    assert len(user_b_workloads) == 0


@pytest.mark.asyncio
async def test_admin_sees_all_workloads(
    client: AsyncClient, auth_headers: dict, admin_auth_headers: dict
):
    """Admin users can see all workloads."""
    # Regular user creates a workload
    await client.post("/v1/workloads", json=VALID_WORKLOAD, headers=auth_headers)

    # Admin lists workloads
    response = await client.get("/v1/workloads", headers=admin_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["pagination"]["total"] >= 1


@pytest.mark.asyncio
async def test_list_active_workloads(client: AsyncClient, auth_headers: dict):
    """GET /v1/workloads/active returns running/scheduled workloads."""
    await client.post("/v1/workloads", json=VALID_WORKLOAD, headers=auth_headers)

    response = await client.get("/v1/workloads/active", headers=auth_headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_update_workload_status_transitions(client: AsyncClient, auth_headers: dict):
    """Workload status can transition from pending to running to completed."""
    create_resp = await client.post(
        "/v1/workloads",
        json=VALID_WORKLOAD,
        headers=auth_headers,
    )
    workload_id = create_resp.json()["metadata"]["id"]

    # Pending -> Running
    resp = await client.patch(
        f"/v1/workloads/{workload_id}",
        json={"status": "running"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"

    # Running -> Completed
    resp = await client.patch(
        f"/v1/workloads/{workload_id}",
        json={"status": "completed"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"
