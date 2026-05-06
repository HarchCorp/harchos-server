"""Tests for carbon-aware scheduling endpoints.

Covers:
- GET /v1/carbon/intensity/MA returns Morocco data
- GET /v1/carbon/intensity returns all zones
- POST /v1/carbon/optimal-hub finds greenest hub
- Carbon data includes gCO2/kWh
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_morocco_carbon_intensity(client: AsyncClient, test_hub):
    """GET /v1/carbon/intensity/MA returns carbon data for Morocco."""
    response = await client.get("/v1/carbon/intensity/MA")
    assert response.status_code == 200
    data = response.json()
    assert data["zone"] == "MA"
    assert "carbon_intensity_gco2_kwh" in data
    assert data["carbon_intensity_gco2_kwh"] > 0
    assert "renewable_percentage" in data
    assert data["renewable_percentage"] >= 0
    assert "fossil_percentage" in data
    assert "source" in data


@pytest.mark.asyncio
async def test_morocco_carbon_data_includes_gco2(client: AsyncClient, test_hub):
    """Morocco carbon data includes gCO2/kWh measurement."""
    response = await client.get("/v1/carbon/intensity/MA")
    data = response.json()
    # Morocco static fallback: 47 gCO2/kWh
    assert data["carbon_intensity_gco2_kwh"] > 0
    assert data["carbon_intensity_gco2_kwh"] < 1000  # Reasonable range


@pytest.mark.asyncio
async def test_morocco_carbon_renewable_percentage(client: AsyncClient, test_hub):
    """Morocco carbon data includes renewable percentage."""
    response = await client.get("/v1/carbon/intensity/MA")
    data = response.json()
    # Morocco static fallback: 81.5% renewable
    assert data["renewable_percentage"] >= 0
    assert data["renewable_percentage"] <= 100


@pytest.mark.asyncio
async def test_morocco_carbon_data_source(client: AsyncClient, test_hub):
    """Morocco carbon data reports its source (static fallback or API)."""
    response = await client.get("/v1/carbon/intensity/MA")
    data = response.json()
    assert data["source"] in ("static", "electricity_maps", "redis_cache", "carbon_intensity_uk")


@pytest.mark.asyncio
async def test_carbon_intensity_zone_not_in_static(client: AsyncClient, test_hub):
    """Unknown zone returns fallback data with default high intensity."""
    response = await client.get("/v1/carbon/intensity/XX")
    assert response.status_code == 200
    data = response.json()
    assert data["zone"] == "XX"
    # Unknown zones use a high default carbon intensity
    assert data["carbon_intensity_gco2_kwh"] > 0


@pytest.mark.asyncio
async def test_carbon_intensity_case_insensitive(client: AsyncClient, test_hub):
    """Zone codes are case-insensitive (ma → MA)."""
    response = await client.get("/v1/carbon/intensity/ma")
    assert response.status_code == 200
    data = response.json()
    assert data["zone"] == "MA"


@pytest.mark.asyncio
async def test_get_all_zone_intensities(client: AsyncClient, test_hub):
    """GET /v1/carbon/intensity returns all zones."""
    response = await client.get("/v1/carbon/intensity")
    assert response.status_code == 200
    data = response.json()
    assert "zones" in data
    assert "total" in data
    assert data["total"] > 0
    assert len(data["zones"]) > 0


@pytest.mark.asyncio
async def test_all_zones_have_gco2(client: AsyncClient, test_hub):
    """All zone entries include carbon intensity in gCO2/kWh."""
    response = await client.get("/v1/carbon/intensity")
    data = response.json()
    for zone in data["zones"]:
        assert "carbon_intensity_gco2_kwh" in zone
        assert zone["carbon_intensity_gco2_kwh"] >= 0


@pytest.mark.asyncio
async def test_all_zones_have_zone_code(client: AsyncClient, test_hub):
    """All zone entries include the zone code."""
    response = await client.get("/v1/carbon/intensity")
    data = response.json()
    for zone in data["zones"]:
        assert "zone" in zone
        assert len(zone["zone"]) > 0


@pytest.mark.asyncio
async def test_find_optimal_hub(client: AsyncClient, auth_headers: dict, test_hub):
    """POST /v1/carbon/optimal-hub finds the greenest hub."""
    response = await client.post(
        "/v1/carbon/optimal-hub",
        json={
            "gpu_count": 2,
            "defer_ok": True,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert data["action"] in ("schedule_now", "defer", "no_suitable_hub")
    assert "carbon_intensity_gco2_kwh" in data
    assert "renewable_percentage" in data
    assert "analyzed_at" in data


@pytest.mark.asyncio
async def test_optimal_hub_with_region(client: AsyncClient, auth_headers: dict, test_hub):
    """POST /v1/carbon/optimal-hub with region filter."""
    response = await client.post(
        "/v1/carbon/optimal-hub",
        json={
            "region": "africa",
            "gpu_count": 1,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_optimal_hub_with_carbon_threshold(client: AsyncClient, auth_headers: dict, test_hub):
    """POST /v1/carbon/optimal-hub with carbon_max_gco2 threshold."""
    response = await client.post(
        "/v1/carbon/optimal-hub",
        json={
            "gpu_count": 1,
            "carbon_max_gco2": 200,
            "defer_ok": True,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in ("schedule_now", "defer", "no_suitable_hub")


@pytest.mark.asyncio
async def test_optimal_hub_requires_auth(client: AsyncClient):
    """POST /v1/carbon/optimal-hub requires authentication."""
    response = await client.post(
        "/v1/carbon/optimal-hub",
        json={"gpu_count": 1},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_optimal_hub_includes_alternatives(client: AsyncClient, auth_headers: dict, test_hub):
    """Optimal hub response includes alternative hubs."""
    response = await client.post(
        "/v1/carbon/optimal-hub",
        json={"gpu_count": 1},
        headers=auth_headers,
    )
    data = response.json()
    assert "alternative_hubs" in data
    assert isinstance(data["alternative_hubs"], list)


@pytest.mark.asyncio
async def test_optimal_hub_carbon_saved(client: AsyncClient, auth_headers: dict, test_hub):
    """Optimal hub response includes estimated carbon savings."""
    response = await client.post(
        "/v1/carbon/optimal-hub",
        json={"gpu_count": 1},
        headers=auth_headers,
    )
    data = response.json()
    assert "estimated_carbon_saved_kg" in data
    assert data["estimated_carbon_saved_kg"] >= 0


@pytest.mark.asyncio
async def test_carbon_intensity_public_access(client: AsyncClient, test_hub):
    """GET /v1/carbon/intensity/{zone} is accessible without auth."""
    response = await client.get("/v1/carbon/intensity/MA")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_carbon_all_zones_public_access(client: AsyncClient, test_hub):
    """GET /v1/carbon/intensity is accessible without auth."""
    response = await client.get("/v1/carbon/intensity")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_carbon_intensity_france(client: AsyncClient, test_hub):
    """GET /v1/carbon/intensity/FR returns France data."""
    response = await client.get("/v1/carbon/intensity/FR")
    assert response.status_code == 200
    data = response.json()
    assert data["zone"] == "FR"
    # France static: 58 gCO2/kWh
    assert data["carbon_intensity_gco2_kwh"] > 0


@pytest.mark.asyncio
async def test_carbon_intensity_poland_high(client: AsyncClient, test_hub):
    """Poland has high carbon intensity (660 gCO2/kWh static)."""
    response = await client.get("/v1/carbon/intensity/PL")
    assert response.status_code == 200
    data = response.json()
    # Poland static fallback is 660 — very high
    assert data["carbon_intensity_gco2_kwh"] > 100


@pytest.mark.asyncio
async def test_carbon_intensity_iceland_low(client: AsyncClient, test_hub):
    """Iceland has very low carbon intensity (7 gCO2/kWh static)."""
    response = await client.get("/v1/carbon/intensity/IS")
    assert response.status_code == 200
    data = response.json()
    # Iceland static fallback is 7 — very low
    assert data["carbon_intensity_gco2_kwh"] < 50
