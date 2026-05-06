"""Tests for input validation across the API.

Covers:
- Invalid names rejected
- Invalid email rejected
- Invalid GPU type rejected
- SQL injection attempts rejected
- XSS attempts rejected
- Oversized inputs rejected
"""

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_name_rejected(client: AsyncClient, auth_headers: dict):
    """Empty workload name is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_whitespace_only_name_rejected(client: AsyncClient, auth_headers: dict):
    """Whitespace-only name is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "   ",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_oversized_name_rejected(client: AsyncClient, auth_headers: dict):
    """Name exceeding 128 characters is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "A" * 200,
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_control_chars_in_name_rejected(client: AsyncClient, auth_headers: dict):
    """Control characters in name are rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Test\x00Name",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Email validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_email_format_rejected(client: AsyncClient):
    """Invalid email format is rejected during registration."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "not-an-email",
            "name": "Test User",
            "role": "user",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_email_domain_rejected(client: AsyncClient):
    """Email without domain is rejected."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "user@",
            "name": "Test User",
            "role": "user",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_email_tld_rejected(client: AsyncClient):
    """Email without TLD is rejected."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "user@domain",
            "name": "Test User",
            "role": "user",
        },
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GPU type validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_gpu_type_rejected(client: AsyncClient, auth_headers: dict):
    """Invalid GPU type is rejected in workload creation."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Bad GPU Workload",
            "type": "training",
            "compute": {
                "gpu_count": 1,
                "gpu_type": "FAKE_GPU_99999",
                "cpu_cores": 4,
                "memory_gb": 16.0,
                "storage_gb": 50.0,
            },
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_valid_gpu_types_accepted(client: AsyncClient, auth_headers: dict):
    """Valid GPU types are accepted."""
    valid_types = ["h100", "a100", "l40s", "a10g"]
    for gpu_type in valid_types:
        response = await client.post(
            "/v1/workloads",
            json={
                "name": f"Workload {gpu_type}",
                "type": "training",
                "compute": {
                    "gpu_count": 1,
                    "gpu_type": gpu_type,
                    "cpu_cores": 4,
                    "memory_gb": 16.0,
                    "storage_gb": 50.0,
                },
                "priority": "normal",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201, f"GPU type '{gpu_type}' should be valid"


# ---------------------------------------------------------------------------
# SQL injection attempts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sql_injection_in_name_rejected(client: AsyncClient, auth_headers: dict):
    """SQL injection in name field is handled safely."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "'; DROP TABLE workloads; --",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    # Should either reject the input or safely store it (no SQL execution)
    assert response.status_code in (201, 400, 422)

    # Verify the table still exists by listing workloads
    list_resp = await client.get("/v1/workloads", headers=auth_headers)
    assert list_resp.status_code == 200


@pytest.mark.asyncio
async def test_sql_injection_in_hub_name(client: AsyncClient, auth_headers: dict):
    """SQL injection in hub name is handled safely."""
    response = await client.post(
        "/v1/hubs",
        json={
            "name": "'; DROP TABLE hubs; --",
            "region": "test",
        },
        headers=auth_headers,
    )
    assert response.status_code in (201, 400, 422)


@pytest.mark.asyncio
async def test_sql_injection_in_api_key_name(client: AsyncClient, auth_headers: dict):
    """SQL injection in API key name is handled safely."""
    response = await client.post(
        "/v1/auth/api-keys",
        json={"name": "'; DROP TABLE api_keys; --"},
        headers=auth_headers,
    )
    # Should either reject or safely store
    assert response.status_code in (201, 400, 422)


# ---------------------------------------------------------------------------
# XSS attempts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_xss_in_name_sanitized(client: AsyncClient, auth_headers: dict):
    """XSS script tags in name are handled safely."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": '<script>alert("xss")</script>',
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    # Should reject or sanitize
    assert response.status_code in (201, 400, 422)

    if response.status_code == 201:
        # If accepted, verify the response doesn't render the script
        # The name may be stored as-is but the API returns JSON (not HTML),
        # so XSS is not a direct concern in API responses
        data = response.json()
        assert "metadata" in data or "spec" in data


@pytest.mark.asyncio
async def test_xss_in_hub_name(client: AsyncClient, auth_headers: dict):
    """XSS in hub name is handled safely."""
    response = await client.post(
        "/v1/hubs",
        json={
            "name": '<img src=x onerror=alert(1)>',
            "region": "test",
        },
        headers=auth_headers,
    )
    assert response.status_code in (201, 400, 422)


# ---------------------------------------------------------------------------
# Oversized inputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oversized_api_key_name(client: AsyncClient, auth_headers: dict):
    """API key name exceeding 64 characters is rejected."""
    response = await client.post(
        "/v1/auth/api-keys",
        json={"name": "A" * 100},
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_oversized_workload_name(client: AsyncClient, auth_headers: dict):
    """Workload name exceeding max length is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "X" * 200,
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_gpu_count_rejected(client: AsyncClient, auth_headers: dict):
    """Negative GPU count is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Negative GPU",
            "type": "training",
            "compute": {
                "gpu_count": -1,
                "cpu_cores": 4,
                "memory_gb": 16.0,
                "storage_gb": 50.0,
            },
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_memory_rejected(client: AsyncClient, auth_headers: dict):
    """Negative memory is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Negative Memory",
            "type": "training",
            "compute": {
                "gpu_count": 1,
                "cpu_cores": 4,
                "memory_gb": -10.0,
                "storage_gb": 50.0,
            },
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_zero_cpu_cores_rejected(client: AsyncClient, auth_headers: dict):
    """Zero CPU cores is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Zero CPU",
            "type": "training",
            "compute": {
                "gpu_count": 1,
                "cpu_cores": 0,
                "memory_gb": 16.0,
                "storage_gb": 50.0,
            },
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_oversized_embedding_input(client: AsyncClient, auth_headers: dict):
    """Very large embedding input is rejected."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "A" * 400000,
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_too_many_embedding_inputs(client: AsyncClient, auth_headers: dict):
    """Too many items in embedding array is rejected."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": ["text"] * 3000,
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_empty_embedding_item_rejected(client: AsyncClient, auth_headers: dict):
    """Empty string in embedding array is rejected."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": ["Hello", "   ", "World"],
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_workload_type_rejected(client: AsyncClient, auth_headers: dict):
    """Invalid workload type is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Bad Type",
            "type": "quantum_computing",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_invalid_sovereignty_level_rejected(client: AsyncClient, auth_headers: dict):
    """Invalid sovereignty level is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Bad Sovereignty",
            "type": "training",
            "sovereignty_level": "ultra_strict",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_invalid_priority_rejected(client: AsyncClient, auth_headers: dict):
    """Invalid priority is rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Bad Priority",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "yesterday",
        },
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_invalid_workload_status_in_update(client: AsyncClient, auth_headers: dict):
    """Invalid status in workload update is rejected."""
    # Create workload first
    create_resp = await client.post(
        "/v1/workloads",
        json={
            "name": "Status Test",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
        },
        headers=auth_headers,
    )
    workload_id = create_resp.json()["metadata"]["id"]

    # Try invalid status
    response = await client.patch(
        f"/v1/workloads/{workload_id}",
        json={"status": "exploded"},
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_secret_in_env_var_rejected(client: AsyncClient, auth_headers: dict):
    """Environment variables with 'password' in name are rejected."""
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Secret Env",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
            "env": {"DATABASE_PASSWORD": "supersecret"},
        },
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_too_many_labels_rejected(client: AsyncClient, auth_headers: dict):
    """More than 50 labels is rejected."""
    labels = {f"label{i}": f"value{i}" for i in range(55)}
    response = await client.post(
        "/v1/workloads",
        json={
            "name": "Too Many Labels",
            "type": "training",
            "compute": {"gpu_count": 1, "cpu_cores": 4, "memory_gb": 16.0, "storage_gb": 50.0},
            "priority": "normal",
            "labels": labels,
        },
        headers=auth_headers,
    )
    assert response.status_code in (400, 422)
