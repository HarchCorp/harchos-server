"""Model ORM model (ML models, not database models)."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, Float, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False, default="pytorch")  # pytorch/tensorflow/jax/onnx/other
    task: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="draft")  # draft/training/ready/deployed/archived

    # Capabilities stored as JSON string
    capabilities_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metrics stored as JSON string
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationship
    hub_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("hubs.id"), nullable=True)

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
