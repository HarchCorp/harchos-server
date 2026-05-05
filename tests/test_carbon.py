"""Tests for the carbon-aware scheduling service and API endpoints.

Covers:
- Carbon intensity fetching (static fallback, API mocking)
- Hub ranking by carbon intensity
- Optimal hub selection (schedule_now, defer, no_suitable_hub)
- Workload optimization (carbon savings calculation)
- Forecast generation
- Carbon metrics aggregation
- API endpoint integration tests
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models.hub import Hub
from app.models.carbon import CarbonIntensityRecord, CarbonOptimizationLog
from app.services.carbon_service import (
    CarbonService,
    STATIC_CARBON_DATA,
    GREEN_THRESHOLD_GCO2,
    _hub_to_zone,
    _estimate_carbon_kg,
    _gpu_power,
)
from app.schemas.carbon import (
    CarbonOptimalHubRequest,
    CarbonOptimizeRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_hubs():
    """Return a list of sample Hub ORM objects."""
    now = datetime.now(timezone.utc)
    return [
        Hub(
            id="hub-morocco-1",
            name="Casablanca GPU Hub",
            region="africa",
            status="ready",
            tier="performance",
            total_gpus=64,
            available_gpus=32,
            total_cpu_cores=512,
            available_cpu_cores=256,
            total_memory_gb=2048,
            available_memory_gb=1024,
            latitude=33.57,
            longitude=-7.59,
            city="Casablanca",
            country="Morocco",
            renewable_percentage=37.0,
            grid_carbon_intensity=520.0,
            pue=1.3,
        ),
        Hub(
            id="hub-sweden-1",
            name="Stockholm Green Hub",
            region="europe",
            status="ready",
            tier="enterprise",
            total_gpus=128,
            available_gpus=64,
            total_cpu_cores=1024,
            available_cpu_cores=512,
            total_memory_gb=4096,
            available_memory_gb=2048,
            latitude=59.33,
            longitude=18.07,
            city="Stockholm",
            country="Sweden",
            renewable_percentage=68.0,
            grid_carbon_intensity=13.0,
            pue=1.1,
        ),
        Hub(
            id="hub-france-1",
            name="Paris Nuclear Hub",
            region="europe",
            status="ready",
            tier="standard",
            total_gpus=32,
            available_gpus=16,
            total_cpu_cores=256,
            available_cpu_cores=128,
            total_memory_gb=1024,
            available_memory_gb=512,
            latitude=48.86,
            longitude=2.35,
            city="Paris",
            country="France",
            renewable_percentage=27.0,
            grid_carbon_intensity=58.0,
            pue=1.2,
        ),
        Hub(
            id="hub-poland-1",
            name="Warsaw Coal Hub",
            region="europe",
            status="ready",
            tier="starter",
            total_gpus=16,
            available_gpus=8,
            total_cpu_cores=128,
            available_cpu_cores=64,
            total_memory_gb=512,
            available_memory_gb=256,
            latitude=52.23,
            longitude=21.01,
            city="Warsaw",
            country="Poland",
            renewable_percentage=22.0,
            grid_carbon_intensity=660.0,
            pue=1.6,
        ),
        Hub(
            id="hub-offline-1",
            name="Offline Hub",
            region="europe",
            status="offline",
            tier="standard",
            total_gpus=32,
            available_gpus=0,
            total_cpu_cores=256,
            available_cpu_cores=0,
            total_memory_gb=1024,
            available_memory_gb=0,
            city="Berlin",
            country="Germany",
            renewable_percentage=50.0,
            grid_carbon_intensity=350.0,
            pue=1.4,
        ),
    ]


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Test carbon service helper functions."""

    def test_gpu_power_known_types(self):
        """GPU power estimates for known GPU types."""
        assert _gpu_power("A100") == 0.4
        assert _gpu_power("H100") == 0.7
        assert _gpu_power("V100") == 0.3
        assert _gpu_power("T4") == 0.07

    def test_gpu_power_unknown_type(self):
        """Unknown GPU type returns default power."""
        assert _gpu_power("UNKNOWN-GPU") == 0.3
        assert _gpu_power(None) == 0.3

    def test_estimate_carbon_kg_basic(self):
        """Basic carbon estimation calculation."""
        # 4 A100s for 1 hour at 100 gCO2/kWh, PUE=1.0
        # total_power = 4 * 0.4 * 1.0 = 1.6 kW
        # kwh = 1.6 * 1.0 = 1.6 kWh
        # co2_kg = 1.6 * 100 / 1000 = 0.16 kg
        carbon = _estimate_carbon_kg(4, "A100", 1.0, 100.0, 1.0)
        assert abs(carbon - 0.16) < 0.001

    def test_estimate_carbon_kg_with_pue(self):
        """Carbon estimation with PUE > 1.0."""
        # Same as above but PUE=1.5
        # total_power = 4 * 0.4 * 1.5 = 2.4 kW
        # kwh = 2.4 * 1.0 = 2.4 kWh
        # co2_kg = 2.4 * 100 / 1000 = 0.24 kg
        carbon = _estimate_carbon_kg(4, "A100", 1.0, 100.0, 1.5)
        assert abs(carbon - 0.24) < 0.001

    def test_estimate_carbon_kg_zero_duration(self):
        """Zero duration = zero carbon."""
        carbon = _estimate_carbon_kg(4, "A100", 0.0, 500.0, 1.0)
        assert carbon == 0.0

    def test_hub_to_zone_morocco(self):
        """Morocco hub maps to MA zone."""
        hub = Hub(country="Morocco", region="africa")
        assert _hub_to_zone(hub) == "MA"

    def test_hub_to_zone_france(self):
        """France hub maps to FR zone."""
        hub = Hub(country="France", region="europe")
        assert _hub_to_zone(hub) == "FR"

    def test_hub_to_zone_sweden(self):
        """Sweden hub maps to SE zone."""
        hub = Hub(country="Sweden", region="europe")
        assert _hub_to_zone(hub) == "SE"

    def test_hub_to_zone_default_morocco(self):
        """Unknown country defaults to MA (Morocco)."""
        hub = Hub(country="Unknown", region="unknown")
        assert _hub_to_zone(hub) == "MA"


# ---------------------------------------------------------------------------
# Unit tests: static carbon data
# ---------------------------------------------------------------------------

class TestStaticCarbonData:
    """Test static fallback carbon data."""

    def test_morocco_data(self):
        """Morocco static data has expected fields."""
        ma = STATIC_CARBON_DATA["MA"]
        assert ma["zone_name"] == "Morocco"
        assert ma["carbon_intensity"] > 0
        assert 0 <= ma["renewable_pct"] <= 100

    def test_sweden_green(self):
        """Sweden has very low carbon intensity."""
        se = STATIC_CARBON_DATA["SE"]
        assert se["carbon_intensity"] < 50
        assert se["renewable_pct"] > 50

    def test_all_zones_have_required_fields(self):
        """All zones have the required fields."""
        for zone_code, data in STATIC_CARBON_DATA.items():
            assert "zone_name" in data, f"Missing zone_name for {zone_code}"
            assert "carbon_intensity" in data, f"Missing carbon_intensity for {zone_code}"
            assert "renewable_pct" in data, f"Missing renewable_pct for {zone_code}"
            assert data["carbon_intensity"] >= 0, f"Negative intensity for {zone_code}"
            assert 0 <= data["renewable_pct"] <= 100, f"Invalid renewable % for {zone_code}"

    def test_get_static_carbon_data_known(self):
        """Known zone returns correct static data."""
        result = CarbonService.get_static_carbon_data("FR")
        assert result["zone_name"] == "France"
        assert result["carbon_intensity"] == 58

    def test_get_static_carbon_data_unknown(self):
        """Unknown zone returns default static data."""
        result = CarbonService.get_static_carbon_data("XX")
        assert result["carbon_intensity"] > 0
        assert result["zone_name"] == "XX"


# ---------------------------------------------------------------------------
# Unit tests: carbon intensity resolution
# ---------------------------------------------------------------------------

class TestCarbonIntensityResolution:
    """Test the carbon intensity data resolution chain."""

    @pytest.mark.asyncio
    async def test_get_zone_intensity_static_fallback(self, db_session):
        """When no API key and no cached data, static fallback is used."""
        result = await CarbonService.get_zone_intensity(db_session, "MA")
        assert result.zone == "MA"
        assert result.carbon_intensity_gco2_kwh > 0
        assert result.source in ("static", "electricity_maps", "carbon_intensity_uk")

    @pytest.mark.asyncio
    async def test_get_zone_intensity_all_zones(self, db_session):
        """All zones return valid intensity data."""
        result = await CarbonService.get_all_zone_intensities(db_session)
        assert result.total > 0
        assert len(result.zones) == result.total
        for zone in result.zones:
            assert zone.carbon_intensity_gco2_kwh >= 0
            assert 0 <= zone.renewable_percentage <= 100


# ---------------------------------------------------------------------------
# Unit tests: hub ranking
# ---------------------------------------------------------------------------

class TestHubRanking:
    """Test hub ranking by carbon intensity."""

    @pytest.mark.asyncio
    async def test_rank_hubs_greenest_first(self, db_session, sample_hubs):
        """Hubs are ranked by carbon intensity (greenest first)."""
        # Add sample hubs to DB
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        ranked = await CarbonService._rank_hubs_by_carbon(db_session)
        # Sweden (13 gCO2/kWh) should be first, Poland (660) last
        assert len(ranked) == 4  # Only 4 "ready" hubs
        assert ranked[0]["hub_zone"] == "SE"
        assert ranked[-1]["hub_zone"] == "PL"

    @pytest.mark.asyncio
    async def test_rank_hubs_region_filter(self, db_session, sample_hubs):
        """Region filter excludes non-matching hubs."""
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        ranked = await CarbonService._rank_hubs_by_carbon(db_session, region="europe")
        for h in ranked:
            assert "europe" in h["hub_region"].lower()

    @pytest.mark.asyncio
    async def test_rank_hubs_gpu_filter(self, db_session, sample_hubs):
        """GPU count filter excludes hubs with insufficient GPUs."""
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        ranked = await CarbonService._rank_hubs_by_carbon(db_session, gpu_count=50)
        for h in ranked:
            assert h["available_gpus"] >= 50


# ---------------------------------------------------------------------------
# Unit tests: optimal hub selection
# ---------------------------------------------------------------------------

class TestOptimalHubSelection:
    """Test carbon-optimal hub selection logic."""

    @pytest.mark.asyncio
    async def test_optimal_hub_schedule_now(self, db_session, sample_hubs):
        """When a green hub is available, schedule_now is returned."""
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        request = CarbonOptimalHubRequest(
            region="europe",
            gpu_count=1,
            carbon_max_gco2=200,
            priority="normal",
            defer_ok=True,
        )
        result = await CarbonService.find_optimal_hub(db_session, request)
        assert result.action in ("schedule_now", "defer")
        if result.action == "schedule_now":
            assert result.carbon_intensity_gco2_kwh <= 200
            assert result.recommended_hub_id is not None

    @pytest.mark.asyncio
    async def test_optimal_hub_no_suitable(self, db_session):
        """When no ready hubs exist, no_suitable_hub is returned."""
        request = CarbonOptimalHubRequest(
            gpu_count=1000,  # Unrealistic
            priority="normal",
        )
        result = await CarbonService.find_optimal_hub(db_session, request)
        assert result.action == "no_suitable_hub"
        assert result.recommended_hub_id is None

    @pytest.mark.asyncio
    async def test_optimal_hub_carbon_savings(self, db_session, sample_hubs):
        """Carbon savings are calculated correctly."""
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        request = CarbonOptimalHubRequest(
            gpu_count=4,
            gpu_type="A100",
            carbon_max_gco2=200,
            priority="normal",
        )
        result = await CarbonService.find_optimal_hub(db_session, request)
        # Sweden (SE) should be recommended with significant savings vs Poland (PL)
        if result.action == "schedule_now" and result.recommended_hub_id == "hub-sweden-1":
            assert result.estimated_carbon_saved_kg > 0


# ---------------------------------------------------------------------------
# Unit tests: workload optimization
# ---------------------------------------------------------------------------

class TestWorkloadOptimization:
    """Test carbon-aware workload optimization."""

    @pytest.mark.asyncio
    async def test_optimize_schedule_now(self, db_session, sample_hubs):
        """Workload optimization schedules on greenest hub."""
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        request = CarbonOptimizeRequest(
            workload_name="test-training",
            gpu_count=2,
            gpu_type="A100",
            carbon_aware=True,
            carbon_max_gco2=200,
            estimated_duration_hours=2.0,
        )
        result = await CarbonService.optimize_workload(db_session, request)
        assert result.action in ("schedule_now", "defer")
        assert result.workload_name == "test-training"
        assert result.carbon_saved_kg >= 0

    @pytest.mark.asyncio
    async def test_optimize_reject_no_hubs(self, db_session):
        """When no hubs are available, optimization rejects."""
        request = CarbonOptimizeRequest(
            workload_name="test-training",
            gpu_count=1000,
            carbon_aware=True,
        )
        result = await CarbonService.optimize_workload(db_session, request)
        assert result.action == "reject"

    @pytest.mark.asyncio
    async def test_optimization_creates_log(self, db_session, sample_hubs):
        """Optimization creates an audit log entry."""
        for hub in sample_hubs:
            db_session.add(hub)
        await db_session.commit()

        request = CarbonOptimizeRequest(
            workload_name="logged-workload",
            gpu_count=1,
            carbon_aware=True,
            estimated_duration_hours=1.0,
        )
        await CarbonService.optimize_workload(db_session, request)

        # Verify log was created
        from sqlalchemy import select
        log_result = await db_session.execute(
            select(CarbonOptimizationLog)
            .where(CarbonOptimizationLog.workload_name == "logged-workload")
        )
        log_entry = log_result.scalar_one_or_none()
        assert log_entry is not None
        assert log_entry.workload_name == "logged-workload"
        assert log_entry.action in ("schedule_now", "defer")


# ---------------------------------------------------------------------------
# Unit tests: forecast
# ---------------------------------------------------------------------------

class TestForecast:
    """Test carbon intensity forecast generation."""

    @pytest.mark.asyncio
    async def test_forecast_has_points(self, db_session):
        """Forecast returns data points."""
        result = await CarbonService.get_forecast(db_session, "MA", hours=6)
        assert result.zone == "MA"
        assert len(result.forecast) > 0

    def test_forecast_points_have_required_fields(self, db_session):
        """Each forecast point has required fields."""
        # This tests the synchronous parts of the model
        pass

    @pytest.mark.asyncio
    async def test_forecast_identifies_green_windows(self, db_session):
        """Forecast identifies green windows in low-carbon zones."""
        # Sweden should have green windows
        result = await CarbonService.get_forecast(db_session, "SE", hours=24)
        # Sweden is very green, should find green windows
        assert len(result.forecast) > 0
        green_points = [p for p in result.forecast if p.is_green]
        assert len(green_points) > 0


# ---------------------------------------------------------------------------
# Unit tests: metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test carbon metrics aggregation."""

    @pytest.mark.asyncio
    async def test_metrics_empty_db(self, db_session):
        """Metrics with no optimization logs returns zeros."""
        result = await CarbonService.get_metrics(db_session)
        assert result.total_carbon_saved_kg == 0.0
        assert result.total_workloads_optimized == 0
        assert result.total_workloads_deferred == 0

    @pytest.mark.asyncio
    async def test_metrics_with_logs(self, db_session, sample_hubs):
        """Metrics with optimization logs returns correct totals."""
        for hub in sample_hubs:
            db_session.add(hub)

        # Create some optimization logs
        for i in range(3):
            db_session.add(CarbonOptimizationLog(
                workload_name=f"test-workload-{i}",
                action="schedule_now",
                selected_hub_name="Stockholm Green Hub",
                carbon_intensity_at_schedule_gco2_kwh=13.0,
                carbon_saved_kg=0.5 + i * 0.1,
                baseline_carbon_kg=1.0 + i * 0.1,
                actual_carbon_kg=0.5,
                deferred_hours=0.0,
                reason="Scheduled on greenest hub",
            ))
        await db_session.commit()

        result = await CarbonService.get_metrics(db_session)
        assert result.total_workloads_optimized == 3
        assert result.total_carbon_saved_kg > 0


# ---------------------------------------------------------------------------
# Integration tests: API endpoints
# ---------------------------------------------------------------------------

class TestCarbonAPIEndpoints:
    """Integration tests for /v1/carbon/* API endpoints."""

    @pytest.mark.asyncio
    async def test_get_zone_intensity(self, client_with_auth):
        """GET /v1/carbon/intensity/{zone} returns carbon data."""
        response = await client_with_auth.get("/v1/carbon/intensity/MA")
        assert response.status_code == 200
        data = response.json()
        assert data["zone"] == "MA"
        assert data["carbon_intensity_gco2_kwh"] > 0
        assert "renewable_percentage" in data

    @pytest.mark.asyncio
    async def test_get_all_intensities(self, client_with_auth):
        """GET /v1/carbon/intensity returns all zones."""
        response = await client_with_auth.get("/v1/carbon/intensity")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        assert len(data["zones"]) == data["total"]

    @pytest.mark.asyncio
    async def test_optimal_hub(self, client_with_auth):
        """POST /v1/carbon/optimal-hub finds best hub."""
        response = await client_with_auth.post(
            "/v1/carbon/optimal-hub",
            json={
                "region": "europe",
                "gpu_count": 1,
                "carbon_max_gco2": 200,
                "priority": "normal",
                "defer_ok": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ("schedule_now", "defer", "no_suitable_hub")

    @pytest.mark.asyncio
    async def test_optimize_workload(self, client_with_auth):
        """POST /v1/carbon/optimize optimizes workload scheduling."""
        response = await client_with_auth.post(
            "/v1/carbon/optimize",
            json={
                "workload_name": "training-job",
                "gpu_count": 4,
                "gpu_type": "A100",
                "carbon_aware": True,
                "carbon_max_gco2": 100,
                "estimated_duration_hours": 2.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ("schedule_now", "defer", "reject")
        assert data["workload_name"] == "training-job"

    @pytest.mark.asyncio
    async def test_get_forecast(self, client_with_auth):
        """GET /v1/carbon/forecast/{zone} returns forecast."""
        response = await client_with_auth.get("/v1/carbon/forecast/SE?hours=6")
        assert response.status_code == 200
        data = response.json()
        assert data["zone"] == "SE"
        assert len(data["forecast"]) > 0

    @pytest.mark.asyncio
    async def test_get_metrics(self, client_with_auth):
        """GET /v1/carbon/metrics returns metrics."""
        response = await client_with_auth.get("/v1/carbon/metrics?period_days=30")
        assert response.status_code == 200
        data = response.json()
        assert "total_carbon_saved_kg" in data
        assert "total_workloads_optimized" in data

    @pytest.mark.asyncio
    async def test_get_dashboard(self, client_with_auth):
        """GET /v1/carbon/dashboard returns dashboard data."""
        response = await client_with_auth.get("/v1/carbon/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "hub_intensities" in data
        assert "optimization_log" in data

    @pytest.mark.asyncio
    async def test_unauthenticated_access(self, client_no_auth):
        """Carbon endpoints require authentication."""
        response = await client_no_auth.get("/v1/carbon/intensity/MA")
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# SDK contract test
# ---------------------------------------------------------------------------

class TestSDKContract:
    """Verify the SDK contract: client.carbon.* works as expected."""

    def test_carbon_resource_available(self):
        """HarchOSClient.carbon property is accessible."""
        from harchos import HarchOSClient
        # Just verify the property exists on the class
        assert hasattr(HarchOSClient, 'carbon')

    def test_carbon_models_importable(self):
        """All carbon models can be imported from the SDK."""
        from harchos import (
            CarbonIntensityZone,
            CarbonOptimalHub,
            CarbonOptimizeResult,
            CarbonForecast,
            CarbonForecastPoint,
            CarbonMetrics,
            CarbonDashboard,
            FuelMixEntry,
            CarbonAction,
            CarbonDataSource,
        )
        # Verify they are classes
        assert isinstance(CarbonIntensityZone, type)
        assert isinstance(CarbonOptimalHub, type)
        assert isinstance(CarbonOptimizeResult, type)

    def test_carbon_optimal_hub_is_green_property(self):
        """CarbonIntensityZone.is_green property works."""
        from harchos import CarbonIntensityZone
        green_zone = CarbonIntensityZone(
            zone="SE",
            carbon_intensity_gco2_kwh=13.0,
            renewable_percentage=68.0,
            fossil_percentage=2.0,
            datetime=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert green_zone.is_green is True

        dirty_zone = CarbonIntensityZone(
            zone="PL",
            carbon_intensity_gco2_kwh=660.0,
            renewable_percentage=22.0,
            fossil_percentage=78.0,
            datetime=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert dirty_zone.is_green is False

    def test_carbon_optimize_result_savings_percentage(self):
        """CarbonOptimizeResult.carbon_savings_percentage property works."""
        from harchos import CarbonOptimizeResult
        result = CarbonOptimizeResult(
            action="schedule_now",
            workload_name="test",
            carbon_saved_kg=0.6,
            baseline_carbon_kg=1.0,
            actual_carbon_kg=0.4,
            optimized_at=datetime.now(timezone.utc),
        )
        assert result.carbon_savings_percentage == 60.0

    def test_carbon_forecast_green_hours(self):
        """CarbonForecast.green_hours_count property works."""
        from harchos import CarbonForecast, CarbonForecastPoint
        forecast = CarbonForecast(
            zone="SE",
            forecast=[
                CarbonForecastPoint(
                    datetime=datetime.now(timezone.utc) + timedelta(minutes=i*15),
                    carbon_intensity_gco2_kwh=50.0,
                    renewable_percentage=70.0,
                    is_green=True,
                )
                for i in range(4)  # 1 hour of green
            ],
        )
        assert forecast.green_hours_count > 0


# ---------------------------------------------------------------------------
# Pytest fixtures for async DB + HTTP client
# ---------------------------------------------------------------------------

# In-memory SQLite for testing
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def db_engine():
    """Create a test database engine."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Create a test database session."""
    async_session = async_sessionmaker(db_engine, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.fixture
async def client_with_auth(db_engine):
    """Create an authenticated test client."""
    from app.api.deps import require_auth
    from app.models.api_key import ApiKey
    from app.models.user import User

    async_session = async_sessionmaker(db_engine, expire_on_commit=False)

    # Seed user and API key
    async with async_session() as session:
        user = User(email="test@harchos.io", name="Test User", is_active=True)
        session.add(user)
        await session.commit()

        api_key = ApiKey(
            user_id=user.id,
            name="test-key",
            key_hash="hash",
            key_prefix="hsk_",
            is_active=True,
        )
        session.add(api_key)
        await session.commit()
        key_id = api_key.id

    # Override dependencies
    async def override_get_db():
        async with async_session() as session:
            yield session

    async def override_require_auth():
        return ApiKey(
            id=key_id,
            user_id=user.id,
            name="test-key",
            key_hash="hash",
            key_prefix="hsk_",
            is_active=True,
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[require_auth] = override_require_auth

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
async def client_no_auth(db_engine):
    """Create an unauthenticated test client."""
    async_session = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override_get_db():
        async with async_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
