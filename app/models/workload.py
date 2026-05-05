"""Workload ORM model with multi-tenancy (user_id) and error tracking."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

class Workload(Base):
    __tablename__ = "workloads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # training/inference/fine_tuning/evaluation/data_pipeline/batch
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")  # pending/scheduled/running/paused/completed/failed/cancelled

    # Multi-tenancy: owner of this workload
    user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)

    # Compute fields
    gpu_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gpu_type: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    cpu_cores: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    memory_gb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    storage_gb: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Relationships
    hub_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("hubs.id"), nullable=True)
    priority: Mapped[str] = mapped_column(String(20), nullable=False, default="normal")  # low/normal/high/critical

    # Sovereignty fields
    sovereignty_level: Mapped[str] = mapped_column(String(50), nullable=False, default="standard")
    data_residency_policy: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    carbon_aware: Mapped[bool] = mapped_column(default=False)
    carbon_intensity_threshold: Mapped[float] = mapped_column(Float, nullable=True)

    # Carbon budget enforcement
    carbon_budget_grams: Mapped[float | None] = mapped_column(Float, nullable=True)
    carbon_actual_grams: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

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
