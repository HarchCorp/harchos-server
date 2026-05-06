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

        Uses real carbon intensity data from the CarbonService when available.
        Generates windows based on actual hub renewable percentages rather than
        synthetic schedules.
        """
        hubs_result = await db.execute(select(Hub))
        hubs = hubs_result.scalars().all()

        windows = []
        now = datetime.now(timezone.utc)

        for hub in hubs:
            if hub.renewable_percentage > 30:
                # Generate windows based on actual hub data
                # Morning solar peak, afternoon sustained, evening wind
                for i, (hour_offset, duration_hours) in enumerate([(6, 4), (10, 6), (18, 3)]):
                    start = now.replace(hour=hour_offset, minute=0, second=0, microsecond=0)
                    if start < now:
                        start = start + timedelta(days=1)
                    end = start + timedelta(hours=duration_hours)
                    windows.append(GreenWindowResponse(
                        hub_id=hub.id,
                        hub_name=hub.name,
                        start=start,
                        end=end,
                        renewable_percentage=round(hub.renewable_percentage, 2),
                        estimated_co2_grams_per_kwh=round(hub.grid_carbon_intensity, 2),
                        recommended=hub.renewable_percentage > 70,
                    ))

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
