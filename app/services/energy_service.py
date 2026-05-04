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
        """Get the latest energy report for a resource."""
        result = await db.execute(
            select(EnergyReport)
            .where(EnergyReport.resource_id == resource_id)
            .order_by(EnergyReport.created_at.desc())
            .limit(1)
        )
        report = result.scalar_one_or_none()
        if not report:
            # Generate a mock report based on hub data
            hub_result = await db.execute(select(Hub).where(Hub.id == resource_id))
            hub = hub_result.scalar_one_or_none()
            if hub:
                now = datetime.now(timezone.utc)
                return EnergyReportResponse(
                    id="mock-report",
                    resource_id=resource_id,
                    resource_type="hub",
                    total_consumption_kwh=hub.total_gpus * 12.5,
                    renewable_percentage=hub.renewable_percentage,
                    carbon_emissions_kg=hub.total_gpus * 12.5 * (1 - hub.renewable_percentage / 100) * hub.grid_carbon_intensity / 1000,
                    pue=hub.pue,
                    efficiency_score=hub.renewable_percentage / 100 * 0.5 + (1.0 / hub.pue) * 0.5,
                    period_start=now - timedelta(days=30),
                    period_end=now,
                    created_at=now,
                )
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
        """Get green energy windows for scheduling."""
        hubs_result = await db.execute(select(Hub))
        hubs = hubs_result.scalars().all()

        windows = []
        now = datetime.now(timezone.utc)

        for hub in hubs:
            # Generate 3 green windows per hub based on renewable percentage
            if hub.renewable_percentage > 30:
                for i in range(3):
                    start = now + timedelta(hours=i * 8)
                    end = start + timedelta(hours=6)
                    # Simulate varying renewable availability throughout the day
                    renewable_pct = min(100, hub.renewable_percentage + (i * 5) - 10)
                    carbon_intensity = max(0, hub.grid_carbon_intensity - (renewable_pct - hub.renewable_percentage) * 2)
                    windows.append(GreenWindowResponse(
                        hub_id=hub.id,
                        hub_name=hub.name,
                        start=start,
                        end=end,
                        renewable_percentage=round(renewable_pct, 2),
                        estimated_co2_grams_per_kwh=round(carbon_intensity, 2),
                        recommended=renewable_pct > 70,
                    ))

        return windows

    @staticmethod
    async def get_consumption(
        db: AsyncSession, resource_id: str
    ) -> list[EnergyConsumptionResponse]:
        """Get energy consumption data for a resource."""
        # Check for stored consumption data
        result = await db.execute(
            select(EnergyConsumption)
            .where(EnergyConsumption.resource_id == resource_id)
            .order_by(EnergyConsumption.timestamp.desc())
            .limit(24)
        )
        consumptions = result.scalars().all()

        if consumptions:
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

        # Generate mock hourly consumption data
        hub_result = await db.execute(select(Hub).where(Hub.id == resource_id))
        hub = hub_result.scalar_one_or_none()

        if not hub:
            return []

        mock_data = []
        now = datetime.now(timezone.utc)
        for i in range(24):
            ts = now - timedelta(hours=23 - i)
            # Simulate day/night variation
            hour = ts.hour
            solar_factor = max(0, (6 - abs(hour - 12)) / 6) if 6 <= hour <= 18 else 0.0
            renewable_pct = min(100, hub.renewable_percentage * (0.5 + solar_factor))
            base_consumption = hub.total_gpus * 0.5  # per hour

            mock_data.append(EnergyConsumptionResponse(
                id=f"mock-{i}",
                resource_id=resource_id,
                resource_type="hub",
                consumption_kwh=round(base_consumption, 3),
                carbon_emissions_kg=round(
                    base_consumption * (1 - renewable_pct / 100) * hub.grid_carbon_intensity / 1000, 3
                ),
                renewable_percentage=round(renewable_pct, 2),
                timestamp=ts,
            ))

        return mock_data
