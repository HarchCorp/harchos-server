"""Tests for auth endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_invalid_api_key(client: AsyncClient):
    """Test that invalid API keys are rejected."""
    response = await client.get("/v1/workloads", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_exchange_api_key_for_token(client: AsyncClient, auth_headers: dict):
    """Test exchanging an API key for a JWT token."""
    response = await client.post("/v1/auth/token", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["access_token"].startswith("hst_")


@pytest.mark.asyncio
async def test_use_jwt_token(client: AsyncClient, auth_headers: dict):
    """Test using a JWT token to access protected endpoints."""
    # Get token
    token_resp = await client.post("/v1/auth/token", headers=auth_headers)
    token = token_resp.json()["access_token"]

    # Use token
    response = await client.get(
        "/v1/workloads",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_current_user(client: AsyncClient, auth_headers: dict):
    """Test getting current user info."""
    response = await client.get("/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "email" in data
    assert "name" in data
