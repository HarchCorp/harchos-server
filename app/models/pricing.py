"""Pricing and billing ORM models.

Supports multi-currency, multi-tier, multi-region pricing for GPU compute,
CPU cores, memory, and storage.  Billing records track per-workload usage
and cost with status lifecycle (pending → paid / overdue).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Float, Integer, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Pricing(Base):
    """Pricing plan for GPU compute resources.

    Each plan defines per-unit costs for a specific GPU type, region, and
    tier combination.  Currencies supported: USD, MAD, EUR.
    """

    __tablename__ = "pricing"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Human-readable plan name"
    )
    gpu_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="GPU type: H100, A100, L40S, etc."
    )
    price_per_gpu_hour: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Cost per GPU per hour"
    )
    price_per_cpu_core_hour: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Cost per CPU core per hour"
    )
    price_per_gb_storage_month: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Cost per GB of storage per month"
    )
    price_per_gb_memory_hour: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Cost per GB of memory per hour"
    )
    currency: Mapped[str] = mapped_column(
        String(10), nullable=False, default="USD",
        comment="Currency code: USD, MAD, EUR"
    )
    region: Mapped[str] = mapped_column(
        String(255), nullable=False, default="",
        comment="Region this pricing applies to"
    )
    tier: Mapped[str] = mapped_column(
        String(50), nullable=False, default="standard",
        comment="Tier: standard, performance, enterprise"
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean, default=False,
        comment="Whether this is the default plan for its GPU type/region/tier"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class BillingRecord(Base):
    """Billing record for a workload's resource usage.

    Tracks GPU hours, CPU core hours, memory, storage, and total cost
    for a specific billing period.  Status lifecycle:
    ``pending`` → ``paid`` | ``overdue``.
    """

    __tablename__ = "billing_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    workload_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("workloads.id"), nullable=True, index=True
    )
    hub_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("hubs.id"), nullable=True, index=True
    )

    # Usage metrics
    gpu_hours: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Total GPU-hours consumed"
    )
    cpu_core_hours: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Total CPU core-hours consumed"
    )
    memory_gb_hours: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Total memory GB-hours consumed"
    )
    storage_gb_months: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Total storage GB-months consumed"
    )

    # Cost
    total_cost: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Total cost for this billing period"
    )
    currency: Mapped[str] = mapped_column(
        String(10), nullable=False, default="USD",
        comment="Currency code: USD, MAD, EUR"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
        comment="Billing status: pending, paid, overdue"
    )

    # Billing period
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="Start of the billing period"
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        comment="End of the billing period"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
