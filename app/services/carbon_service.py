"""Carbon-aware scheduling service.

This service is the core of HarchOS's competitive differentiator: native
carbon-aware GPU orchestration.  It:

1. Fetches real-time carbon intensity data from **Electricity Maps** and
   **Carbon Intensity (UK)** APIs, with static fallback data when APIs
   are unavailable.
2. Ranks hubs by current carbon intensity and renewable percentage.
3. Decides whether to schedule a workload immediately, defer it to a
   green window, or (in extreme cases) reject it.
4. Logs every optimization decision for auditing and dashboards.
5. Provides aggregate carbon metrics (total saved, per-hub breakdowns).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import httpx
from sqlalchemy import select, func, and_, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.carbon import CarbonIntensityRecord, CarbonOptimizationLog
from app.models.hub import Hub
from app.schemas.carbon import (
    CarbonDashboardResponse,
    CarbonForecastPoint,
    CarbonForecastResponse,
    CarbonIntensityZoneResponse,
    CarbonIntensityZoneListResponse,
    CarbonMetricsResponse,
    CarbonOptimalHubResponse,
    CarbonOptimalHubRequest,
    CarbonOptimizeRequest,
    CarbonOptimizeResponse,
    FuelMixEntry,
)

logger = logging.getLogger("harchos.carbon")

# ---------------------------------------------------------------------------
# Static fallback data — used when no API key is configured or APIs are down
# ---------------------------------------------------------------------------

# Carbon intensity in gCO2/kWh and renewable % by Electricity Maps zone code
STATIC_CARBON_DATA: dict[str, dict[str, Any]] = {
    "MA":  {"zone_name": "Morocco",        "carbon_intensity": 520,  "renewable_pct": 37.0,  "fossil_pct": 63.0},
    "FR":  {"zone_name": "France",         "carbon_intensity": 58,   "renewable_pct": 27.0,  "fossil_pct": 8.0},
    "DE":  {"zone_name": "Germany",        "carbon_intensity": 350,  "renewable_pct": 50.0,  "fossil_pct": 36.0},
    "GB":  {"zone_name": "Great Britain",  "carbon_intensity": 230,  "renewable_pct": 43.0,  "fossil_pct": 40.0},
    "NL":  {"zone_name": "Netherlands",    "carbon_intensity": 340,  "renewable_pct": 33.0,  "fossil_pct": 53.0},
    "IE":  {"zone_name": "Ireland",        "carbon_intensity": 280,  "renewable_pct": 40.0,  "fossil_pct": 52.0},
    "SE":  {"zone_name": "Sweden",         "carbon_intensity": 13,   "renewable_pct": 68.0,  "fossil_pct": 2.0},
    "NO":  {"zone_name": "Norway",         "carbon_intensity": 9,    "renewable_pct": 98.0,  "fossil_pct": 1.5},
    "IS":  {"zone_name": "Iceland",        "carbon_intensity": 7,    "renewable_pct": 99.9,  "fossil_pct": 0.1},
    "PT":  {"zone_name": "Portugal",       "carbon_intensity": 180,  "renewable_pct": 60.0,  "fossil_pct": 30.0},
    "ES":  {"zone_name": "Spain",          "carbon_intensity": 170,  "renewable_pct": 50.0,  "fossil_pct": 30.0},
    "IT":  {"zone_name": "Italy",          "carbon_intensity": 310,  "renewable_pct": 36.0,  "fossil_pct": 52.0},
    "PL":  {"zone_name": "Poland",         "carbon_intensity": 660,  "renewable_pct": 22.0,  "fossil_pct": 78.0},
    "DK":  {"zone_name": "Denmark",        "carbon_intensity": 65,   "renewable_pct": 80.0,  "fossil_pct": 12.0},
    "FI":  {"zone_name": "Finland",        "carbon_intensity": 55,   "renewable_pct": 52.0,  "fossil_pct": 15.0},
    "US-CAL-CISO": {"zone_name": "California (CAISO)", "carbon_intensity": 200, "renewable_pct": 47.0, "fossil_pct": 40.0},
    "US-TEX-ERCO": {"zone_name": "Texas (ERCOT)",       "carbon_intensity": 380, "renewable_pct": 31.0, "fossil_pct": 60.0},
    "CA-ON":       {"zone_name": "Ontario",              "carbon_intensity": 30,  "renewable_pct": 94.0, "fossil_pct": 5.0},
    "SG":  {"zone_name": "Singapore",      "carbon_intensity": 490,  "renewable_pct": 3.0,   "fossil_pct": 96.0},
    "JP-TK": {"zone_name": "Tokyo",        "carbon_intensity": 470,  "renewable_pct": 22.0,  "fossil_pct": 68.0},
    "AU-NSW": {"zone_name": "NSW Australia", "carbon_intensity": 580, "renewable_pct": 27.0, "fossil_pct": 72.0},
}

# Default green threshold: below 200 gCO2/kWh is considered "green"
GREEN_THRESHOLD_GCO2 = 200.0

# Average GPU power consumption in kW (used for carbon savings estimation)
GPU_POWER_KW = {
    "A100": 0.4,
    "H100": 0.7,
    "V100": 0.3,
    "A10G": 0.15,
    "T4": 0.07,
    "L4": 0.12,
    "default": 0.3,
}


def _gpu_power(gpu_type: str | None) -> float:
    """Return estimated power consumption in kW for a GPU type."""
    if gpu_type:
        return GPU_POWER_KW.get(gpu_type.upper(), GPU_POWER_KW["default"])
    return GPU_POWER_KW["default"]


def _estimate_carbon_kg(
    gpu_count: int,
    gpu_type: str | None,
    duration_hours: float,
    carbon_intensity_gco2_kwh: float,
    pue: float = 1.0,
) -> float:
    """Estimate CO2 emissions in kg for a workload.

    Formula: (GPU_count * GPU_power_kW * PUE * duration_h) * carbon_intensity / 1000
    """
    total_power_kw = gpu_count * _gpu_power(gpu_type) * pue
    kwh = total_power_kw * duration_hours
    co2_kg = kwh * carbon_intensity_gco2_kwh / 1000.0
    return co2_kg


class CarbonService:
    """Service for carbon-aware scheduling decisions."""

    # ------------------------------------------------------------------
    # 1. Carbon intensity data fetching
    # ------------------------------------------------------------------

    @staticmethod
    async def fetch_carbon_intensity_electricity_maps(
        zone: str,
    ) -> dict[str, Any] | None:
        """Fetch real-time carbon intensity from Electricity Maps API.

        Docs: https://api.electricitymap.org/
        Requires HARCHOS_ELECTRICITY_MAPS_API_KEY to be set.
        """
        api_key = getattr(settings, "electricity_maps_api_key", "")
        if not api_key:
            logger.debug("No Electricity Maps API key configured, skipping")
            return None

        url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
        headers = {"auth-token": api_key}

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(url, params={"zone": zone}, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "zone": zone,
                        "carbon_intensity": data.get("carbonIntensity", 0),
                        "datetime": data.get("datetime", ""),
                        "updated_at": data.get("updatedAt", ""),
                    }
                logger.warning("Electricity Maps API returned %d for zone %s", resp.status_code, zone)
            except httpx.HTTPError as exc:
                logger.warning("Electricity Maps API error for zone %s: %s", zone, exc)
        return None

    @staticmethod
    async def fetch_carbon_intensity_uk(
        region: str = "",
    ) -> dict[str, Any] | None:
        """Fetch real-time carbon intensity from Carbon Intensity API (UK).

        Docs: https://carbonintensity.org.uk/
        Free, no API key required for basic queries.
        """
        url = "https://api.carbonintensity.org.uk/intensity"
        if region:
            url = f"https://api.carbonintensity.org.uk/regional/regionid/{region}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    intensity = data.get("data", [{}])[0] if "data" in data else {}
                    return {
                        "zone": "GB",
                        "carbon_intensity": intensity.get("intensity", {}).get("actual", 0)
                            or intensity.get("intensity", {}).get("forecast", 0),
                        "datetime": intensity.get("from", ""),
                        "updated_at": intensity.get("to", ""),
                    }
                logger.warning("Carbon Intensity UK API returned %d", resp.status_code)
            except httpx.HTTPError as exc:
                logger.warning("Carbon Intensity UK API error: %s", exc)
        return None

    @staticmethod
    def get_static_carbon_data(zone: str) -> dict[str, Any]:
        """Return static fallback carbon data for a zone."""
        return STATIC_CARBON_DATA.get(
            zone,
            {"zone_name": zone, "carbon_intensity": 400.0, "renewable_pct": 20.0, "fossil_pct": 80.0},
        )

    @staticmethod
    async def get_zone_intensity(
        db: AsyncSession, zone: str, *, use_cache: bool = True,
    ) -> CarbonIntensityZoneResponse:
        """Get carbon intensity for a zone, using live API → DB cache → static fallback.

        Resolution order:
        1. Recent DB record (< 30 min old)
        2. Live API (Electricity Maps or Carbon Intensity UK)
        3. Static fallback data
        """
        now = datetime.now(timezone.utc)

        # 1. Check DB for recent data (< 30 min)
        if use_cache:
            recent = await db.execute(
                select(CarbonIntensityRecord)
                .where(
                    CarbonIntensityRecord.zone == zone,
                    CarbonIntensityRecord.is_forecast.is_(False),
                    CarbonIntensityRecord.datetime >= now - timedelta(minutes=30),
                )
                .order_by(CarbonIntensityRecord.datetime.desc())
                .limit(1)
            )
            record = recent.scalar_one_or_none()
            if record:
                fuel_mix = []
                if record.fuel_mix_json:
                    try:
                        fuel_mix = [FuelMixEntry(**e) for e in json.loads(record.fuel_mix_json)]
                    except (json.JSONDecodeError, TypeError):
                        pass
                return CarbonIntensityZoneResponse(
                    zone=record.zone,
                    zone_name=STATIC_CARBON_DATA.get(zone, {}).get("zone_name", zone),
                    carbon_intensity_gco2_kwh=record.carbon_intensity_gco2_kwh,
                    renewable_percentage=record.renewable_percentage,
                    fossil_percentage=record.fossil_percentage,
                    fuel_mix=fuel_mix,
                    source=record.source,
                    is_forecast=record.is_forecast,
                    datetime=record.datetime,
                    updated_at=record.created_at,
                )

        # 2. Try live APIs
        live_data = await CarbonService.fetch_carbon_intensity_electricity_maps(zone)
        source = "electricity_maps"

        if live_data is None and zone in ("GB", "GB-ENG", "GB-SCT", "GB-WLS", "GB-NIR"):
            live_data = await CarbonService.fetch_carbon_intensity_uk()
            source = "carbon_intensity_uk"

        if live_data and live_data.get("carbon_intensity") is not None:
            ci = live_data["carbon_intensity"]
            static = CarbonService.get_static_carbon_data(zone)
            # Store in DB for caching
            db_record = CarbonIntensityRecord(
                zone=zone,
                carbon_intensity_gco2_kwh=float(ci),
                renewable_percentage=static.get("renewable_pct", 0.0),
                fossil_percentage=static.get("fossil_pct", 0.0),
                source=source,
                is_forecast=False,
                datetime=now,
            )
            db.add(db_record)
            await db.commit()

            return CarbonIntensityZoneResponse(
                zone=zone,
                zone_name=static.get("zone_name", zone),
                carbon_intensity_gco2_kwh=float(ci),
                renewable_percentage=static.get("renewable_pct", 0.0),
                fossil_percentage=static.get("fossil_pct", 0.0),
                fuel_mix=[],
                source=source,
                is_forecast=False,
                datetime=now,
                updated_at=now,
            )

        # 3. Static fallback
        static = CarbonService.get_static_carbon_data(zone)
        return CarbonIntensityZoneResponse(
            zone=zone,
            zone_name=static["zone_name"],
            carbon_intensity_gco2_kwh=static["carbon_intensity"],
            renewable_percentage=static["renewable_pct"],
            fossil_percentage=static["fossil_pct"],
            fuel_mix=[],
            source="static",
            is_forecast=False,
            datetime=now,
            updated_at=now,
        )

    @staticmethod
    async def get_all_zone_intensities(
        db: AsyncSession,
    ) -> CarbonIntensityZoneListResponse:
        """Get carbon intensity for all known zones."""
        zones_data = []
        for zone_code in STATIC_CARBON_DATA:
            intensity = await CarbonService.get_zone_intensity(db, zone_code)
            zones_data.append(intensity)
        return CarbonIntensityZoneListResponse(zones=zones_data, total=len(zones_data))

    # ------------------------------------------------------------------
    # 2. Hub ranking & optimal hub selection
    # ------------------------------------------------------------------

    @staticmethod
    async def _rank_hubs_by_carbon(
        db: AsyncSession,
        *,
        region: str | None = None,
        gpu_count: int | None = None,
        gpu_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank all hubs by carbon intensity (lowest first).

        Returns a list of dicts with hub info + current carbon data.
        """
        query = select(Hub).where(Hub.status == "ready")
        if region:
            query = query.where(Hub.region.ilike(f"%{region}%"))
        if gpu_count:
            query = query.where(Hub.available_gpus >= gpu_count)

        result = await db.execute(query)
        hubs = result.scalars().all()

        ranked = []
        for hub in hubs:
            zone = _hub_to_zone(hub)
            intensity = await CarbonService.get_zone_intensity(db, zone)
            ranked.append({
                "hub_id": hub.id,
                "hub_name": hub.name,
                "hub_region": hub.region,
                "hub_zone": zone,
                "city": hub.city,
                "country": hub.country,
                "available_gpus": hub.available_gpus,
                "total_gpus": hub.total_gpus,
                "carbon_intensity_gco2_kwh": intensity.carbon_intensity_gco2_kwh,
                "renewable_percentage": intensity.renewable_percentage,
                "pue": hub.pue,
                "renewable_hub_pct": hub.renewable_percentage,
                "source": intensity.source,
            })

        # Sort by carbon intensity ascending (greenest first)
        ranked.sort(key=lambda h: h["carbon_intensity_gco2_kwh"])
        return ranked

    @staticmethod
    async def find_optimal_hub(
        db: AsyncSession, request: CarbonOptimalHubRequest,
    ) -> CarbonOptimalHubResponse:
        """Find the carbon-optimal hub for a workload.

        Decision logic:
        1. Rank all eligible hubs by carbon intensity.
        2. If the best hub's carbon intensity <= carbon_max_gco2 (or default threshold),
           schedule now.
        3. If carbon_max_gco2 is set and no hub meets it, check if deferral is ok.
        4. If defer_ok, find the next green window and recommend deferral.
        5. Otherwise, return the best available hub anyway.
        """
        now = datetime.now(timezone.utc)
        max_gco2 = request.carbon_max_gco2 or GREEN_THRESHOLD_GCO2

        ranked = await CarbonService._rank_hubs_by_carbon(
            db,
            region=request.region,
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
        )

        if not ranked:
            return CarbonOptimalHubResponse(
                recommended_hub_id=None,
                recommended_hub_name="",
                hub_region=request.region or "",
                hub_zone="",
                carbon_intensity_gco2_kwh=0.0,
                renewable_percentage=0.0,
                available_gpus=0,
                action="no_suitable_hub",
                defer_hours=0.0,
                defer_reason="No ready hubs found matching requirements",
                estimated_carbon_saved_kg=0.0,
                alternative_hubs=[],
                analyzed_at=now,
            )

        best = ranked[0]
        worst = ranked[-1]

        # Calculate carbon savings vs worst hub
        duration_h = 1.0  # assume 1h for comparison
        gpu_ct = request.gpu_count or 1
        baseline_carbon = _estimate_carbon_kg(
            gpu_ct, request.gpu_type, duration_h, worst["carbon_intensity_gco2_kwh"], worst["pue"],
        )
        best_carbon = _estimate_carbon_kg(
            gpu_ct, request.gpu_type, duration_h, best["carbon_intensity_gco2_kwh"], best["pue"],
        )
        carbon_saved = baseline_carbon - best_carbon

        # Decision: schedule now or defer?
        if best["carbon_intensity_gco2_kwh"] <= max_gco2:
            action = "schedule_now"
            defer_hours = 0.0
            defer_reason = ""
        elif request.defer_ok and request.priority not in ("high", "critical"):
            # Recommend deferral — find next green window
            forecast = await CarbonService.get_forecast(db, best["hub_zone"])
            green_window = _find_next_green_window(forecast, max_gco2)
            if green_window:
                action = "defer"
                defer_hours = max(0.0, (green_window["start"] - now).total_seconds() / 3600)
                defer_reason = (
                    f"Current carbon intensity at best hub ({best['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh) "
                    f"exceeds threshold ({max_gco2:.0f} gCO2/kWh). "
                    f"Green window starts in {defer_hours:.1f}h."
                )
            else:
                # No green window found — schedule anyway but flag it
                action = "schedule_now"
                defer_hours = 0.0
                defer_reason = (
                    f"No green window found in forecast. Scheduling on best available hub "
                    f"({best['carbon_intensity_gco2_kwh']:.0f} gCO2/kWh)."
                )
        else:
            # High/critical priority or defer not ok — schedule now
            action = "schedule_now"
            defer_hours = 0.0
            defer_reason = (
                f"Workload priority={request.priority}, scheduling immediately on best hub."
            )

        # Build alternative hubs (top 5)
        alternatives = [
            {
                "hub_id": h["hub_id"],
                "hub_name": h["hub_name"],
                "carbon_intensity_gco2_kwh": h["carbon_intensity_gco2_kwh"],
                "renewable_percentage": h["renewable_percentage"],
                "available_gpus": h["available_gpus"],
            }
            for h in ranked[1:6]
        ]

        return CarbonOptimalHubResponse(
            recommended_hub_id=best["hub_id"],
            recommended_hub_name=best["hub_name"],
            hub_region=best["hub_region"],
            hub_zone=best["hub_zone"],
            carbon_intensity_gco2_kwh=best["carbon_intensity_gco2_kwh"],
            renewable_percentage=best["renewable_percentage"],
            available_gpus=best["available_gpus"],
            action=action,
            defer_hours=round(defer_hours, 2),
            defer_reason=defer_reason,
            estimated_carbon_saved_kg=round(carbon_saved, 4),
            alternative_hubs=alternatives,
            analyzed_at=now,
        )

    # ------------------------------------------------------------------
    # 3. Workload carbon optimization
    # ------------------------------------------------------------------

    @staticmethod
    async def optimize_workload(
        db: AsyncSession, request: CarbonOptimizeRequest,
    ) -> CarbonOptimizeResponse:
        """Optimize a workload's scheduling based on carbon intensity.

        This is the main entry point that the ``/v1/carbon/optimize``
        endpoint calls.  It combines hub ranking, carbon calculation,
        and deferral logic into a single decision.
        """
        now = datetime.now(timezone.utc)
        max_gco2 = request.carbon_max_gco2 or GREEN_THRESHOLD_GCO2

        # Find optimal hub
        hub_request = CarbonOptimalHubRequest(
            region=request.region,
            gpu_count=request.gpu_count,
            gpu_type=request.gpu_type,
            carbon_max_gco2=request.carbon_max_gco2,
            priority=request.priority,
            defer_ok=request.carbon_aware,
        )
        optimal = await CarbonService.find_optimal_hub(db, hub_request)

        if optimal.action == "no_suitable_hub":
            return CarbonOptimizeResponse(
                action="reject",
                workload_name=request.workload_name,
                selected_hub_id=None,
                selected_hub_name="",
                carbon_intensity_at_schedule_gco2_kwh=0.0,
                carbon_saved_kg=0.0,
                baseline_carbon_kg=0.0,
                actual_carbon_kg=0.0,
                deferred_hours=0.0,
                reason="No suitable hub found for workload requirements",
                estimated_green_window=None,
                optimized_at=now,
            )

        # Calculate carbon estimates
        ranked = await CarbonService._rank_hubs_by_carbon(
            db, region=request.region, gpu_count=request.gpu_count, gpu_type=request.gpu_type,
        )
        if ranked:
            worst_ci = ranked[-1]["carbon_intensity_gco2_kwh"]
            worst_pue = ranked[-1]["pue"]
        else:
            worst_ci = 500.0
            worst_pue = 1.5

        baseline_carbon = _estimate_carbon_kg(
            request.gpu_count, request.gpu_type, request.estimated_duration_hours,
            worst_ci, worst_pue,
        )

        hub_pue = 1.0
        if optimal.recommended_hub_id:
            hub_result = await db.execute(select(Hub).where(Hub.id == optimal.recommended_hub_id))
            hub_obj = hub_result.scalar_one_or_none()
            if hub_obj:
                hub_pue = hub_obj.pue

        actual_carbon = _estimate_carbon_kg(
            request.gpu_count, request.gpu_type, request.estimated_duration_hours,
            optimal.carbon_intensity_gco2_kwh, hub_pue,
        )
        carbon_saved = baseline_carbon - actual_carbon

        action = optimal.action
        if action == "no_suitable_hub":
            action = "reject"

        # Log the optimization decision
        log_entry = CarbonOptimizationLog(
            workload_name=request.workload_name,
            action=action,
            selected_hub_id=optimal.recommended_hub_id,
            selected_hub_name=optimal.recommended_hub_name,
            carbon_intensity_at_schedule_gco2_kwh=optimal.carbon_intensity_gco2_kwh,
            carbon_saved_kg=round(carbon_saved, 6),
            baseline_carbon_kg=round(baseline_carbon, 6),
            actual_carbon_kg=round(actual_carbon, 6),
            deferred_hours=optimal.defer_hours,
            reason=optimal.defer_reason or f"Scheduled on {optimal.recommended_hub_name}",
        )
        db.add(log_entry)
        await db.commit()

        # Green window info for deferred workloads
        green_window = None
        if action == "defer":
            zone = optimal.hub_zone
            forecast = await CarbonService.get_forecast(db, zone)
            gw = _find_next_green_window(forecast, max_gco2)
            if gw:
                green_window = gw

        return CarbonOptimizeResponse(
            action=action,
            workload_name=request.workload_name,
            selected_hub_id=optimal.recommended_hub_id,
            selected_hub_name=optimal.recommended_hub_name,
            carbon_intensity_at_schedule_gco2_kwh=optimal.carbon_intensity_gco2_kwh,
            carbon_saved_kg=round(carbon_saved, 6),
            baseline_carbon_kg=round(baseline_carbon, 6),
            actual_carbon_kg=round(actual_carbon, 6),
            deferred_hours=optimal.defer_hours,
            reason=optimal.defer_reason or f"Scheduled on {optimal.recommended_hub_name}",
            estimated_green_window=green_window,
            optimized_at=now,
        )

    # ------------------------------------------------------------------
    # 4. Forecast
    # ------------------------------------------------------------------

    @staticmethod
    async def get_forecast(
        db: AsyncSession, zone: str, hours: int = 24,
    ) -> CarbonForecastResponse:
        """Get a carbon intensity forecast for a zone.

        Tries Electricity Maps forecast API first, then generates a
        synthetic forecast from historical patterns.
        """
        now = datetime.now(timezone.utc)
        static = CarbonService.get_static_carbon_data(zone)
        base_intensity = static["carbon_intensity"]
        base_renewable = static["renewable_pct"]

        # Try live forecast from Electricity Maps
        api_key = getattr(settings, "electricity_maps_api_key", "")
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "https://api.electricitymap.org/v3/carbon-intensity/forecast",
                        params={"zone": zone},
                        headers={"auth-token": api_key},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        forecast_points = data.get("forecast", [])
                        if forecast_points:
                            points = []
                            green_windows = []
                            current_green_start = None

                            for fp in forecast_points[: hours * 4]:  # 15-min intervals
                                ci = fp.get("carbonIntensity", base_intensity)
                                dt_str = fp.get("datetime", "")
                                try:
                                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                                except (ValueError, AttributeError):
                                    dt = now + timedelta(minutes=15 * len(points))
                                is_green = ci <= GREEN_THRESHOLD_GCO2

                                points.append(CarbonForecastPoint(
                                    datetime=dt,
                                    carbon_intensity_gco2_kwh=ci,
                                    renewable_percentage=min(100, base_renewable + (base_intensity - ci) / 5),
                                    is_green=is_green,
                                ))

                                # Track green windows
                                if is_green and current_green_start is None:
                                    current_green_start = dt
                                elif not is_green and current_green_start is not None:
                                    green_windows.append({
                                        "start": current_green_start.isoformat(),
                                        "end": dt.isoformat(),
                                        "estimated_carbon_intensity_gco2_kwh": ci,
                                    })
                                    current_green_start = None

                            if current_green_start is not None and points:
                                green_windows.append({
                                    "start": current_green_start.isoformat(),
                                    "end": points[-1].reading_datetime.isoformat(),
                                    "estimated_carbon_intensity_gco2_kwh": GREEN_THRESHOLD_GCO2,
                                })

                            return CarbonForecastResponse(
                                zone=zone,
                                zone_name=static["zone_name"],
                                forecast=points,
                                green_windows=green_windows,
                            )
            except httpx.HTTPError as exc:
                logger.warning("Electricity Maps forecast API error: %s", exc)

        # Synthetic forecast based on daily patterns
        points = []
        green_windows = []
        current_green_start = None

        for i in range(hours * 4):  # 15-min intervals
            dt = now + timedelta(minutes=15 * i)
            hour = dt.hour

            # Solar peaks around noon, wind peaks at night
            solar_factor = max(0, math.cos((hour - 12) * math.pi / 12)) if 6 <= hour <= 18 else 0
            wind_factor = 0.3 + 0.2 * math.sin(hour * math.pi / 6)

            # Renewable contribution reduces carbon intensity
            renewable_boost = solar_factor * 25 + wind_factor * 15
            ci = max(0, base_intensity - renewable_boost * (base_renewable / 100))
            renewable = min(100, base_renewable + renewable_boost)
            is_green = ci <= GREEN_THRESHOLD_GCO2

            points.append(CarbonForecastPoint(
                datetime=dt,
                carbon_intensity_gco2_kwh=round(ci, 1),
                renewable_percentage=round(renewable, 1),
                is_green=is_green,
            ))

            if is_green and current_green_start is None:
                current_green_start = dt
            elif not is_green and current_green_start is not None:
                green_windows.append({
                    "start": current_green_start.isoformat(),
                    "end": dt.isoformat(),
                    "estimated_carbon_intensity_gco2_kwh": round(ci, 1),
                })
                current_green_start = None

        if current_green_start is not None and points:
            green_windows.append({
                "start": current_green_start.isoformat(),
                "end": points[-1].reading_datetime.isoformat(),
                "estimated_carbon_intensity_gco2_kwh": GREEN_THRESHOLD_GCO2,
            })

        return CarbonForecastResponse(
            zone=zone,
            zone_name=static["zone_name"],
            forecast=points,
            green_windows=green_windows,
        )

    # ------------------------------------------------------------------
    # 5. Metrics & Dashboard
    # ------------------------------------------------------------------

    @staticmethod
    async def get_metrics(
        db: AsyncSession,
        period_days: int = 30,
    ) -> CarbonMetricsResponse:
        """Get aggregate carbon metrics for the platform."""
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(days=period_days)

        # Aggregate optimization logs
        result = await db.execute(
            select(
                func.count(CarbonOptimizationLog.id).label("total_optimized"),
                func.sum(CarbonOptimizationLog.carbon_saved_kg).label("total_saved"),
                func.sum(
                    func.cast(CarbonOptimizationLog.action == "defer", Integer)
                ).label("total_deferred"),
                func.avg(CarbonOptimizationLog.carbon_intensity_at_schedule_gco2_kwh).label(
                    "avg_carbon_intensity"
                ),
            ).where(CarbonOptimizationLog.created_at >= period_start)
        )
        row = result.one()

        total_optimized = row.total_optimized or 0
        total_saved = float(row.total_saved or 0)
        total_deferred = int(row.total_deferred or 0)
        avg_ci = float(row.avg_carbon_intensity or 0)

        # Best and worst hubs
        ranked = await CarbonService._rank_hubs_by_carbon(db)
        best_hub = ranked[0] if ranked else {}
        worst_ci = ranked[-1]["carbon_intensity_gco2_kwh"] if ranked else 0.0

        return CarbonMetricsResponse(
            total_carbon_saved_kg=round(total_saved, 4),
            total_workloads_optimized=total_optimized,
            total_workloads_deferred=total_deferred,
            average_carbon_intensity_gco2_kwh=round(avg_ci, 2),
            best_hub_id=best_hub.get("hub_id"),
            best_hub_name=best_hub.get("hub_name", ""),
            best_hub_carbon_intensity=best_hub.get("carbon_intensity_gco2_kwh", 0.0),
            worst_hub_carbon_intensity=round(worst_ci, 2),
            period_start=period_start,
            period_end=now,
        )

    @staticmethod
    async def get_dashboard(db: AsyncSession) -> CarbonDashboardResponse:
        """Get full carbon dashboard data."""
        metrics = await CarbonService.get_metrics(db)

        # Hub intensities
        zones = await CarbonService.get_all_zone_intensities(db)

        # Recent optimization logs
        log_result = await db.execute(
            select(CarbonOptimizationLog)
            .order_by(CarbonOptimizationLog.created_at.desc())
            .limit(20)
        )
        logs = log_result.scalars().all()
        log_data = [
            {
                "id": str(l.id),
                "workload_name": l.workload_name,
                "action": l.action,
                "selected_hub_name": l.selected_hub_name,
                "carbon_saved_kg": l.carbon_saved_kg,
                "actual_carbon_kg": l.actual_carbon_kg,
                "deferred_hours": l.deferred_hours,
                "reason": l.reason,
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in logs
        ]

        # Green windows
        green_windows = []
        for zone_resp in zones.zones[:5]:  # top 5 zones
            forecast = await CarbonService.get_forecast(db, zone_resp.zone, hours=12)
            for gw in forecast.green_windows:
                gw["zone"] = zone_resp.zone
                gw["zone_name"] = zone_resp.zone_name
                green_windows.append(gw)

        return CarbonDashboardResponse(
            metrics=metrics,
            hub_intensities=zones.zones,
            optimization_log=log_data,
            green_windows=green_windows,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Mapping from Hub country/region to Electricity Maps zone code
_COUNTRY_TO_ZONE: dict[str, str] = {
    "morocco": "MA",
    "maroc": "MA",
    "morocco": "MA",
    "france": "FR",
    "germany": "DE",
    "deutschland": "DE",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "netherlands": "NL",
    "ireland": "IE",
    "sweden": "SE",
    "norway": "NO",
    "iceland": "IS",
    "portugal": "PT",
    "spain": "ES",
    "italy": "IT",
    "poland": "PL",
    "denmark": "DK",
    "finland": "FI",
    "singapore": "SG",
}


def _hub_to_zone(hub: Hub) -> str:
    """Map a Hub ORM object to an Electricity Maps zone code.

    Uses the hub's country field first, then region, then defaults to
    Morocco (since HarchOS is Moroccan-first).
    """
    country = (hub.country or "").lower().strip()
    if country in _COUNTRY_TO_ZONE:
        return _COUNTRY_TO_ZONE[country]

    region = (hub.region or "").lower().strip()
    for key, zone in _COUNTRY_TO_ZONE.items():
        if key in region or region in key:
            return zone

    # Default: Morocco (HarchOS is a Moroccan platform)
    return "MA"


def _find_next_green_window(
    forecast: CarbonForecastResponse, max_gco2: float,
) -> dict | None:
    """Find the next green window from a forecast."""
    for gw in forecast.green_windows:
        try:
            from datetime import datetime as _dt
            start = _dt.fromisoformat(gw["start"].replace("Z", "+00:00")) if isinstance(gw["start"], str) else gw["start"]
        except (ValueError, TypeError, KeyError):
            continue
        return gw  # Return first green window found
    return None
