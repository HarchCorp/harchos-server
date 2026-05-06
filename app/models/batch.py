"""Batch job ORM model — persisted storage for batch inference jobs.

Replaces the in-memory BatchStore with proper database-backed storage,
enabling horizontal scaling and data persistence across restarts.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, Float, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class BatchJob(Base):
    """Batch inference job stored in database.

    Each batch contains a list of inference request items stored as JSON
    in the ``input_data`` column. As items are processed, their status
    and results are updated in-place within the JSON blob, and the
    aggregate counters (completed_items, failed_items) are updated on
    the row itself.
    """
    __tablename__ = "batch_jobs"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True,
        default=lambda: f"batch_{uuid.uuid4().hex[:24]}",
    )
    user_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )

    # Lifecycle: pending / processing / completed / failed / cancelled
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
    )

    # Counters
    total_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # JSON columns — rich structured data
    input_data: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON array of batch request items with their statuses and results",
    )
    results: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="JSON snapshot of results (mirrors input_data after processing)",
    )
    metadata_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Optional user-provided metadata as JSON object",
    )
    aggregate_carbon_footprint: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="JSON object with aggregate carbon footprint data",
    )

    # Flags
    carbon_aware: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Timestamps (stored as datetime objects; API responses use Unix timestamps)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="When the batch expires and can be cleaned up",
    )
