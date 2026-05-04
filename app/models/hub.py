"""Hub ORM model."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

class Hub(Base):
    __tablename__ = "hubs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="creating")  # creating/ready/updating/scaling/draining/offline/error
    tier: Mapped[str] = mapped_column(String(50), nullable=False, default="standard")  # starter/standard/performance/enterprise

    # Capacity
    total_gpus: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_gpus: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cpu_cores: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_cpu_cores: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_memory_gb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    available_memory_gb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_storage_gb: Mapped[float] = mapped_column(Float, nullable=False, default=50000.0)
    available_storage_gb: Mapped[float] = mapped_column(Float, nullable=False, default=35000.0)

    # Location
    latitude: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    longitude: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    city: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    country: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    # Energy
    renewable_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    grid_carbon_intensity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pue: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    # Sovereignty
    sovereignty_level: Mapped[str] = mapped_column(String(50), nullable=False, default="standard")
    data_residency_policy: Mapped[str] = mapped_column(String(255), nullable=False, default="local_only")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
