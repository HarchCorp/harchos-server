"""Carbon intensity ORM models.

Stores real-time and historical carbon intensity data per electricity zone,
enabling the carbon-aware scheduling engine to pick the greenest hub and
the greenest time window for every workload.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class CarbonIntensityRecord(Base):
    """Historical carbon intensity reading for an electricity zone.

    A *zone* maps to a geographical area defined by Electricity Maps
    (e.g. ``"MA"`` for Morocco, ``"FR"`` for France).  Each record
    captures the grid carbon intensity at a specific point in time so
    the scheduler can compute averages, trends, and forecasts.
    """

    __tablename__ = "carbon_intensity_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    zone: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Electricity Maps zone code"
    )
    carbon_intensity_gco2_kwh: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Grid carbon intensity in gCO2/kWh"
    )
    renewable_percentage: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Percentage of renewable energy in the grid mix"
    )
    fossil_percentage: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Percentage of fossil fuel in the grid mix"
    )
    # Breakdown by fuel type (JSON as text — kept simple for SQLite compat)
    fuel_mix_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="JSON breakdown of fuel mix (solar, wind, hydro, nuclear, gas, coal, etc.)"
    )
    source: Mapped[str] = mapped_column(
        String(100), nullable=False, default="electricity_maps",
        comment="Data source: electricity_maps | carbon_intensity_uk | static | manual"
    )
    is_forecast: Mapped[bool] = mapped_column(
        default=False, comment="Whether this is a forecast or actual reading"
    )
    datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="Timestamp of the reading (or forecast target time)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class CarbonOptimizationLog(Base):
    """Log of carbon-aware scheduling decisions.

    Each time the optimizer selects a hub or defers a workload, a record
    is created here so that dashboards can display cumulative carbon
    savings and auditors can verify compliance.
    """

    __tablename__ = "carbon_optimization_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    workload_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    workload_name: Mapped[str] = mapped_column(
        String(255), nullable=False, default=""
    )
    action: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="schedule | defer | migrate | reject"
    )
    selected_hub_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    selected_hub_name: Mapped[str] = mapped_column(
        String(255), nullable=False, default=""
    )
    # Carbon metrics for this decision
    carbon_intensity_at_schedule_gco2_kwh: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    carbon_saved_kg: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Estimated CO2 saved vs. baseline (highest-intensity hub)"
    )
    baseline_carbon_kg: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="What the CO2 would have been on the dirtiest hub"
    )
    actual_carbon_kg: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Actual CO2 for this scheduling decision"
    )
    deferred_hours: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Hours the workload was deferred (0 if scheduled immediately)"
    )
    reason: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="Human-readable explanation of the scheduling decision"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
