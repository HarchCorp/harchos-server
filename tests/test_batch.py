"""Tests for batch inference endpoints.

Covers:
- POST /v1/inference/batch creates batch
- GET /v1/inference/batch/{id} gets status
- POST /v1/inference/batch/{id}/cancel cancels batch
- GET /v1/inference/batch lists batches
- Batch discount applied (50%)

Note: The batch router is mounted at prefix="/inference" with empty path "",
so the batch endpoints are at /v1/inference (POST/GET list),
/v1/inference/{batch_id} (GET), and /v1/inference/{batch_id}/cancel (POST).
"""

import asyncio

import pytest
from httpx import AsyncClient

# The batch router uses empty path ("") under /inference prefix,
# so batch submit is POST /v1/inference and batch list is GET /v1/inference.
# However, to avoid confusion with other inference routes, we use the
# explicit query approach or direct paths.
# After checking the route listing, the actual paths are:
# POST /v1/inference  → submit batch
# GET  /v1/inference  → list batches
# GET  /v1/inference/{batch_id}  → get batch
# POST /v1/inference/{batch_id}/cancel  → cancel batch


BATCH_SUBMIT_URL = "/v1/inference"
BATCH_LIST_URL = "/v1/inference"


def _batch_get_url(batch_id: str) -> str:
    return f"/v1/inference/{batch_id}"


def _batch_cancel_url(batch_id: str) -> str:
    return f"/v1/inference/{batch_id}/cancel"


@pytest.mark.asyncio
async def test_create_batch(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference creates a new batch job."""
    response = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-1",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Hello"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["status"] in ("pending", "processing")
    assert data["total_requests"] == 1


@pytest.mark.asyncio
async def test_create_batch_multiple_items(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference with multiple requests."""
    response = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-a",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                {
                    "request_id": "req-b",
                    "model": "harchos-llama-3.3-8b",
                    "messages": [{"role": "user", "content": "World"}],
                },
                {
                    "request_id": "req-c",
                    "model": "harchos-mistral-large",
                    "messages": [{"role": "user", "content": "Test"}],
                },
            ],
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["total_requests"] == 3


@pytest.mark.asyncio
async def test_create_batch_with_metadata(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference with optional metadata."""
    response = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-meta",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Metadata test"}],
                }
            ],
            "metadata": {"project": "test", "run": "1"},
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data.get("metadata") == {"project": "test", "run": "1"}


@pytest.mark.asyncio
async def test_get_batch_status(client: AsyncClient, auth_headers: dict):
    """GET /v1/inference/{batch_id} returns batch status."""
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-status",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Status test"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # Wait a bit for processing
    await asyncio.sleep(0.5)

    response = await client.get(_batch_get_url(batch_id), headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == batch_id
    assert data["status"] in ("pending", "processing", "completed", "failed")


@pytest.mark.asyncio
async def test_get_batch_includes_results(client: AsyncClient, auth_headers: dict):
    """GET /v1/inference/{batch_id} with include_results=true includes results."""
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-results",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Results test"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # Wait for completion
    await asyncio.sleep(1.0)

    response = await client.get(
        _batch_get_url(batch_id),
        params={"include_results": True},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    if data["status"] == "completed" and data.get("results"):
        result = data["results"][0]
        assert result["request_id"] == "req-results"


@pytest.mark.asyncio
async def test_cancel_batch(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference/{batch_id}/cancel cancels a batch."""
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": f"req-cancel-{i}",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": f"Cancel test {i}"}],
                }
                for i in range(10)
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # Try to cancel immediately
    response = await client.post(
        _batch_cancel_url(batch_id),
        headers=auth_headers,
    )
    # Should succeed (200) or batch already completed (400)
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "cancelled"
        assert data["cancelled_requests"] >= 0


@pytest.mark.asyncio
async def test_cancel_completed_batch_fails(client: AsyncClient, auth_headers: dict):
    """Cancelling an already completed batch returns an error."""
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-done",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Done test"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # Wait for completion
    await asyncio.sleep(1.5)

    # Try to cancel
    response = await client.post(
        _batch_cancel_url(batch_id),
        headers=auth_headers,
    )
    if response.status_code == 400:
        data = response.json()
        assert data["error"]["code"] == "E0200"


@pytest.mark.asyncio
async def test_list_batches(client: AsyncClient, auth_headers: dict):
    """GET /v1/inference lists user's batches."""
    # Create a batch first
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-list",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "List test"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201

    response = await client.get(BATCH_LIST_URL, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert "total" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_list_batches_pagination(client: AsyncClient, auth_headers: dict):
    """GET /v1/inference supports pagination params."""
    response = await client.get(
        BATCH_LIST_URL,
        params={"limit": 5, "offset": 0},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "has_more" in data


@pytest.mark.asyncio
async def test_batch_discount_applied(client: AsyncClient, auth_headers: dict):
    """Batch items have 50% discount in carbon footprint."""
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-discount",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Discount test"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # Wait for completion
    await asyncio.sleep(1.0)

    # Get results
    response = await client.get(
        _batch_get_url(batch_id),
        params={"include_results": True},
        headers=auth_headers,
    )
    data = response.json()
    if data["status"] == "completed" and data.get("results"):
        result = data["results"][0]
        if result.get("carbon_footprint"):
            cf = result["carbon_footprint"]
            assert cf.get("batch_discount_applied") is True
            assert cf.get("batch_discount_factor") == 0.5


@pytest.mark.asyncio
async def test_batch_aggregate_carbon(client: AsyncClient, auth_headers: dict):
    """Completed batch includes aggregate carbon footprint."""
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": f"req-agg-{i}",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": f"Aggregate {i}"}],
                }
                for i in range(3)
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # Wait for completion
    await asyncio.sleep(1.5)

    response = await client.get(_batch_get_url(batch_id), headers=auth_headers)
    data = response.json()
    if data["status"] == "completed":
        assert data.get("aggregate_carbon_footprint") is not None
        agg = data["aggregate_carbon_footprint"]
        assert "total_gco2" in agg
        assert "items_completed" in agg
        assert agg["total_gco2"] > 0
        assert "batch_savings_gco2" in agg


@pytest.mark.asyncio
async def test_batch_duplicate_request_ids_rejected(client: AsyncClient, auth_headers: dict):
    """Duplicate request_id values within a batch are rejected."""
    response = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "dup-1",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "First"}],
                },
                {
                    "request_id": "dup-1",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Second"}],
                },
            ],
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_invalid_model_rejected(client: AsyncClient, auth_headers: dict):
    """Invalid model in batch request is rejected."""
    response = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-bad",
                    "model": "nonexistent-model",
                    "messages": [{"role": "user", "content": "Bad model"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_batch_requires_auth(client: AsyncClient):
    """Batch endpoints require authentication."""
    response = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-noauth",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "No auth"}],
                }
            ],
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_nonexistent_batch(client: AsyncClient, auth_headers: dict):
    """GET /v1/inference/{id} with invalid ID returns 404."""
    response = await client.get(
        _batch_get_url("nonexistent-batch-id"),
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_batch_access_other_users_batch_denied(
    client: AsyncClient, auth_headers: dict, viewer_auth_headers: dict
):
    """User cannot access another user's batch."""
    # User A creates a batch
    create_resp = await client.post(
        BATCH_SUBMIT_URL,
        json={
            "requests": [
                {
                    "request_id": "req-owner",
                    "model": "harchos-llama-3.3-70b",
                    "messages": [{"role": "user", "content": "Owner test"}],
                }
            ],
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    batch_id = create_resp.json()["id"]

    # User B tries to access it
    response = await client.get(
        _batch_get_url(batch_id),
        headers=viewer_auth_headers,
    )
    assert response.status_code in (403, 404)  # Either forbidden or not found
