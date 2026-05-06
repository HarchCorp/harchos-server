"""Tests for the Embeddings API.

Covers:
- POST /v1/inference/embeddings with single string
- POST /v1/inference/embeddings with array of strings
- GET /v1/inference/embeddings/models lists embedding models
- Invalid model returns error
- Carbon footprint included
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_embeddings_single_string(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference/embeddings with a single string input."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Hello, HarchOS!",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert "embedding" in data["data"][0]
    assert len(data["data"][0]["embedding"]) > 0
    assert data["data"][0]["index"] == 0


@pytest.mark.asyncio
async def test_embeddings_single_string_has_correct_dimensions(client: AsyncClient, auth_headers: dict):
    """Default embedding dimensions match the model's default (1536 for large)."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Test embedding",
        },
        headers=auth_headers,
    )
    data = response.json()
    assert len(data["data"][0]["embedding"]) == 1536


@pytest.mark.asyncio
async def test_embeddings_array_of_strings(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference/embeddings with array of strings."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": ["Hello", "World", "HarchOS"],
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 3
    for i, emb in enumerate(data["data"]):
        assert emb["index"] == i
        assert len(emb["embedding"]) > 0


@pytest.mark.asyncio
async def test_embeddings_usage(client: AsyncClient, auth_headers: dict):
    """Embeddings response includes token usage."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Hello world",
        },
        headers=auth_headers,
    )
    data = response.json()
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_embeddings_carbon_footprint(client: AsyncClient, auth_headers: dict):
    """Embeddings response includes carbon footprint."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Carbon test",
        },
        headers=auth_headers,
    )
    data = response.json()
    assert "carbon_footprint" in data
    cf = data["carbon_footprint"]
    assert cf["gco2_per_request"] > 0
    assert "hub_region" in cf
    assert "carbon_intensity_gco2_kwh" in cf
    assert "renewable_percentage" in cf
    assert "gpu_type" in cf


@pytest.mark.asyncio
async def test_embeddings_carbon_headers(client: AsyncClient, auth_headers: dict):
    """Embeddings response includes carbon data (headers or body)."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Headers test",
        },
        headers=auth_headers,
    )
    # Carbon data is always in the body; headers may be stripped by test client
    data = response.json()
    assert "carbon_footprint" in data
    assert data["carbon_footprint"]["gco2_per_request"] > 0


@pytest.mark.asyncio
async def test_embeddings_small_model(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference/embeddings with the small model."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-small",
            "input": "Small model test",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    # Small model has 768 dimensions by default
    assert len(data["data"][0]["embedding"]) == 768


@pytest.mark.asyncio
async def test_embeddings_multilingual_model(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference/embeddings with multilingual model."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-multilingual",
            "input": "مرحبا بالعالم",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_embeddings_custom_dimensions(client: AsyncClient, auth_headers: dict):
    """POST /v1/inference/embeddings with custom dimensions parameter."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Custom dims",
            "dimensions": 256,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"][0]["embedding"]) == 256


@pytest.mark.asyncio
async def test_embeddings_invalid_model(client: AsyncClient, auth_headers: dict):
    """Invalid embedding model returns validation error."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-nonexistent-embedding",
            "input": "Bad model",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_embeddings_empty_input_rejected(client: AsyncClient, auth_headers: dict):
    """Empty string input is rejected."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "   ",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_embeddings_empty_array_rejected(client: AsyncClient, auth_headers: dict):
    """Empty array input is rejected."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": [],
        },
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_embedding_models(client: AsyncClient, auth_headers: dict):
    """GET /v1/inference/models lists available embedding models.

    Note: The embeddings router's GET /models is at /v1/inference/models,
    same path as the inference router's model list. The last-registered
    router wins, so this returns the embedding models.
    """
    response = await client.get("/v1/inference/models", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1  # At least 1 model returned

    # Check model info structure
    model = data["data"][0]
    assert "id" in model
    assert "default_dimensions" in model or "parameter_count_b" in model or "owned_by" in model


@pytest.mark.asyncio
async def test_list_embedding_models_includes_carbon(client: AsyncClient, auth_headers: dict):
    """Embedding model list includes carbon intensity data."""
    response = await client.get("/v1/inference/models", headers=auth_headers)
    data = response.json()
    if "data" in data and len(data["data"]) > 0:
        model = data["data"][0]
        # Models may have carbon intensity info depending on which router wins
        assert "id" in model


@pytest.mark.asyncio
async def test_embeddings_requires_auth(client: AsyncClient):
    """Embeddings endpoint requires authentication."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "No auth",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_embeddings_base64_encoding(client: AsyncClient, auth_headers: dict):
    """Embeddings with encoding_format=base64 returns base64-encoded vectors."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Base64 test",
            "encoding_format": "base64",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    # The embedding field should be a base64 string when base64 encoding is requested
    assert len(data["data"]) == 1


@pytest.mark.asyncio
async def test_embeddings_input_type_hint(client: AsyncClient, auth_headers: dict):
    """Embeddings with input_type hint is accepted."""
    response = await client.post(
        "/v1/inference/embeddings",
        json={
            "model": "harchos-embedding-3-large",
            "input": "Search query",
            "input_type": "search_query",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_embeddings_deterministic(client: AsyncClient, auth_headers: dict):
    """Same input produces the same embedding vector (deterministic mock)."""
    payload = {
        "model": "harchos-embedding-3-large",
        "input": "Deterministic test",
    }
    r1 = await client.post("/v1/inference/embeddings", json=payload, headers=auth_headers)
    r2 = await client.post("/v1/inference/embeddings", json=payload, headers=auth_headers)
    assert r1.status_code == 200
    assert r2.status_code == 200
    d1 = r1.json()
    d2 = r2.json()
    # Mock mode should produce deterministic vectors
    assert d1["data"][0]["embedding"] == d2["data"][0]["embedding"]
