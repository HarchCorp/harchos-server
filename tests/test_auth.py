"""Tests for authentication endpoints.

Covers:
- POST /v1/auth/api-keys — create API key
- POST /v1/auth/token — exchange API key for JWT
- GET /v1/auth/me — get current user info
- DELETE /v1/auth/api-keys/{id} — revoke API key
- POST /v1/auth/login — login with email + API key
- POST /v1/auth/register — register new user (dev mode)
- Invalid key returns E0101
- Expired token returns E0102
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_api_key(client: AsyncClient, auth_headers: dict):
    """POST /v1/auth/api-keys creates a new API key."""
    response = await client.post(
        "/v1/auth/api-keys",
        json={"name": "My Test Key"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert "key" in data
    assert data["key"].startswith("hsk_")
    assert data["name"] == "My Test Key"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_create_api_key_requires_auth(client: AsyncClient):
    """POST /v1/auth/api-keys requires authentication."""
    response = await client.post(
        "/v1/auth/api-keys",
        json={"name": "Should Fail"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_exchange_api_key_for_token(client: AsyncClient, auth_headers: dict):
    """POST /v1/auth/token exchanges API key for JWT."""
    response = await client.post("/v1/auth/token", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["access_token"].startswith("hst_")
    assert "expires_in" in data
    assert data["expires_in"] > 0


@pytest.mark.asyncio
async def test_token_via_bearer_header(client: AsyncClient, test_api_key: dict):
    """POST /v1/auth/token works with Authorization: Bearer hsk_... header."""
    response = await client.post(
        "/v1/auth/token",
        headers={"Authorization": f"Bearer {test_api_key['raw_key']}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"].startswith("hst_")


@pytest.mark.asyncio
async def test_get_current_user_info(client: AsyncClient, auth_headers: dict, test_user):
    """GET /v1/auth/me returns the authenticated user's info."""
    response = await client.get("/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email
    assert data["name"] == test_user.name
    assert data["is_active"] is True
    assert "id" in data


@pytest.mark.asyncio
async def test_get_me_with_bearer_token(client: AsyncClient, bearer_auth_headers: dict, test_user):
    """GET /v1/auth/me works with Bearer JWT token."""
    response = await client.get("/v1/auth/me", headers=bearer_auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email


@pytest.mark.asyncio
async def test_revoke_api_key(client: AsyncClient, auth_headers: dict, test_api_key: dict):
    """DELETE /v1/auth/api-keys/{id} revokes the key."""
    key_id = test_api_key["api_key_obj"].id
    response = await client.delete(
        f"/v1/auth/api-keys/{key_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["revoked"] is True
    assert data["id"] == key_id


@pytest.mark.asyncio
async def test_revoke_nonexistent_key(client: AsyncClient, auth_headers: dict):
    """DELETE /v1/auth/api-keys/{id} with invalid ID returns 404."""
    response = await client.delete(
        "/v1/auth/api-keys/nonexistent-id",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_api_key_returns_e0101(client: AsyncClient):
    """Invalid API key returns error code E0101."""
    response = await client.get(
        "/v1/auth/me",
        headers={"X-API-Key": "hsk_invalid_key_that_does_not_exist"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "E0101"


@pytest.mark.asyncio
async def test_missing_auth_returns_e0100(client: AsyncClient):
    """Missing authentication returns error code E0100."""
    response = await client.get("/v1/auth/me")
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "E0100"


@pytest.mark.asyncio
async def test_invalid_token_returns_e0102(client: AsyncClient):
    """Invalid JWT token returns error code E0102."""
    response = await client.get(
        "/v1/auth/me",
        headers={"Authorization": "Bearer hst_invalidtoken12345"},
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "E0102"


@pytest.mark.asyncio
async def test_non_hsk_prefix_key_rejected(client: AsyncClient):
    """API key without hsk_ prefix is rejected."""
    response = await client.get(
        "/v1/auth/me",
        headers={"X-API-Key": "sk-1234567890abcdef"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_with_email_and_key(client: AsyncClient, test_api_key: dict, test_user):
    """POST /v1/auth/login exchanges email + API key for JWT."""
    response = await client.post(
        "/v1/auth/login",
        json={
            "email": test_user.email,
            "api_key": test_api_key["raw_key"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"].startswith("hst_")


@pytest.mark.asyncio
async def test_login_wrong_email(client: AsyncClient, test_api_key: dict):
    """POST /v1/auth/login with wrong email returns E0103."""
    response = await client.post(
        "/v1/auth/login",
        json={
            "email": "wrong@example.com",
            "api_key": test_api_key["raw_key"],
        },
    )
    assert response.status_code == 401
    data = response.json()
    assert data["error"]["code"] == "E0103"


@pytest.mark.asyncio
async def test_login_invalid_key(client: AsyncClient):
    """POST /v1/auth/login with invalid key returns E0101."""
    response = await client.post(
        "/v1/auth/login",
        json={
            "email": "test@harchos.ai",
            "api_key": "hsk_nonexistent_key_12345678",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_register_new_user(client: AsyncClient):
    """POST /v1/auth/register creates a new user in dev mode."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "newuser@harchos.ai",
            "name": "New User",
            "role": "user",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "user" in data
    assert "api_key" in data
    assert "token" in data
    assert data["user"]["email"] == "newuser@harchos.ai"
    assert data["api_key"]["key"].startswith("hsk_")


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, test_user):
    """POST /v1/auth/register with duplicate email returns error."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": test_user.email,
            "name": "Duplicate",
            "role": "user",
        },
    )
    assert response.status_code == 409
    data = response.json()
    assert data["error"]["code"] in ("E0308", "E0309")


@pytest.mark.asyncio
async def test_register_duplicate_email_case_insensitive(client: AsyncClient, test_user):
    """POST /v1/auth/register rejects duplicate email regardless of casing.

    If a user registered as 'testuser@harchos.ai', attempting to register
    again as 'TESTUSER@HARCHOS.AI' must still return 409.
    """
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": test_user.email.upper(),
            "name": "Duplicate Upper",
            "role": "user",
        },
    )
    assert response.status_code == 409
    data = response.json()
    assert data["error"]["code"] in ("E0308", "E0309")


@pytest.mark.asyncio
async def test_register_invalid_role(client: AsyncClient):
    """POST /v1/auth/register with invalid role returns error."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": "roleuser@harchos.ai",
            "name": "Role User",
            "role": "superadmin",
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_revoked_key_cannot_authenticate(client: AsyncClient, auth_headers: dict, test_api_key: dict):
    """After revoking an API key, it can no longer authenticate."""
    key_id = test_api_key["api_key_obj"].id

    # Revoke the key
    await client.delete(f"/v1/auth/api-keys/{key_id}", headers=auth_headers)

    # Try to use the revoked key
    response = await client.get(
        "/v1/auth/me",
        headers={"X-API-Key": test_api_key["raw_key"]},
    )
    assert response.status_code == 401
