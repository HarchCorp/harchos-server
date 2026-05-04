"""Energy ORM models."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

class EnergyReport(Base):
    __tablename__ = "energy_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False, default="hub")

    # Energy metrics
    total_consumption_kwh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    renewable_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    carbon_emissions_kg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pue: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    efficiency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Period
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

class EnergyConsumption(Base):
    __tablename__ = "energy_consumption"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False, default="hub")

    consumption_kwh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    carbon_emissions_kg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    renewable_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
