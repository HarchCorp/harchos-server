"""Energy service – energy reports, summary, green windows, consumption."""

from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.hub import Hub
from app.models.energy import EnergyReport, EnergyConsumption
from app.schemas.energy import (
    EnergyReportResponse,
    EnergySummaryResponse,
    GreenWindowResponse,
    EnergyConsumptionResponse,
)

class EnergyService:
    """Service for energy-related operations."""

    @staticmethod
    async def get_energy_report(
        db: AsyncSession, resource_id: str
    ) -> EnergyReportResponse | None:
        """Get the latest energy report for a resource.

        Returns real DB data only. No mock/fallback data is generated.
        If no report exists in the database, returns None (caller returns 404).
        """
        result = await db.execute(
            select(EnergyReport)
            .where(EnergyReport.resource_id == resource_id)
            .order_by(EnergyReport.created_at.desc())
            .limit(1)
        )
        report = result.scalar_one_or_none()
        if not report:
            return None
        return EnergyReportResponse(
            id=report.id,
            resource_id=report.resource_id,
            resource_type=report.resource_type,
            total_consumption_kwh=report.total_consumption_kwh,
            renewable_percentage=report.renewable_percentage,
            carbon_emissions_kg=report.carbon_emissions_kg,
            pue=report.pue,
            efficiency_score=report.efficiency_score,
            period_start=report.period_start,
            period_end=report.period_end,
            created_at=report.created_at,
        )

    @staticmethod
    async def get_energy_summary(db: AsyncSession) -> EnergySummaryResponse:
        """Get energy summary across all hubs – matches SDK EnergySummary."""
        # Get hub count
        hub_count_result = await db.execute(select(func.count()).select_from(Hub))
        hub_count = hub_count_result.scalar() or 0

        # Aggregate hub energy data
        hubs_result = await db.execute(select(Hub))
        hubs = hubs_result.scalars().all()

        total_kwh = 0.0
        total_co2_grams = 0.0
        total_pue = 0.0
        total_renewable_fraction = 0.0

        for hub in hubs:
            consumption = hub.total_gpus * 12.5  # rough estimate
            total_kwh += consumption
            co2 = consumption * (1 - hub.renewable_percentage / 100) * hub.grid_carbon_intensity
            total_co2_grams += co2
            total_pue += hub.pue
            total_renewable_fraction += hub.renewable_percentage / 100.0

        average_pue = total_pue / hub_count if hub_count > 0 else 1.0
        average_renewable = total_renewable_fraction / hub_count if hub_count > 0 else 0.0

        now = datetime.now(timezone.utc)

        return EnergySummaryResponse(
            total_kwh=round(total_kwh, 2),
            total_co2_grams=round(total_co2_grams, 2),
            average_pue=round(average_pue, 3),
            average_renewable_fraction=round(min(average_renewable, 1.0), 4),
            resource_count=hub_count,
            period_start=now - timedelta(days=30),
            period_end=now,
        )

    @staticmethod
    async def get_green_windows(db: AsyncSession) -> list[GreenWindowResponse]:
        """Get green energy windows for scheduling.

        Uses real carbon intensity forecast data from CarbonService when available.
        Identifies time windows where carbon intensity is below the green threshold,
        ensuring data is based on actual forecasts rather than synthetic schedules.
        """
        from app.services.carbon_service import CarbonService, GREEN_THRESHOLD_GCO2

        hubs_result = await db.execute(select(Hub))
        hubs = hubs_result.scalars().all()

        windows = []
        now = datetime.now(timezone.utc)

        for hub in hubs:
            # Map hub to electricity zone
            from app.services.carbon_service import _hub_to_zone
            zone = _hub_to_zone(hub)

            try:
                # Get real forecast data
                forecast = await CarbonService.get_forecast(db, zone, hours=24)

                # Extract green windows from forecast
                for gw in forecast.green_windows:
                    try:
                        start = gw.get("start", "")
                        end = gw.get("end", "")
                        ci = gw.get("estimated_carbon_intensity_gco2_kwh", 0)

                        # Parse ISO datetime strings
                        if isinstance(start, str):
                            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        else:
                            start_dt = start
                        if isinstance(end, str):
                            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                        else:
                            end_dt = end

                        # Only include future windows
                        if end_dt > now:
                            windows.append(GreenWindowResponse(
                                hub_id=hub.id,
                                hub_name=hub.name,
                                start=start_dt,
                                end=end_dt,
                                renewable_percentage=hub.renewable_percentage,
                                estimated_co2_grams_per_kwh=ci if ci > 0 else hub.grid_carbon_intensity,
                                recommended=ci <= GREEN_THRESHOLD_GCO2 if ci > 0 else hub.renewable_percentage > 70,
                            ))
                    except (ValueError, TypeError, KeyError):
                        continue
            except Exception as exc:
                import logging
                logging.getLogger("harchos.energy").warning(
                    "Could not get forecast for hub %s zone %s: %s", hub.name, zone, exc
                )
                continue

        # Sort by start time
        windows.sort(key=lambda w: w.start)
        return windows

    @staticmethod
    async def get_consumption(
        db: AsyncSession, resource_id: str
    ) -> list[EnergyConsumptionResponse]:
        """Get energy consumption data for a resource.

        Returns real DB data only. No mock/fallback data is generated.
        If no consumption data exists in the database, returns an empty list.
        """
        result = await db.execute(
            select(EnergyConsumption)
            .where(EnergyConsumption.resource_id == resource_id)
            .order_by(EnergyConsumption.timestamp.desc())
            .limit(24)
        )
        consumptions = result.scalars().all()

        if not consumptions:
            return []

        return [
            EnergyConsumptionResponse(
                id=c.id,
                resource_id=c.resource_id,
                resource_type=c.resource_type,
                consumption_kwh=c.consumption_kwh,
                carbon_emissions_kg=c.carbon_emissions_kg,
                renewable_percentage=c.renewable_percentage,
                timestamp=c.timestamp,
            )
            for c in consumptions
        ]
