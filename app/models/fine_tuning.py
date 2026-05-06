"""Fine-tuning ORM models — persisted storage for jobs, files, and models.

Replaces the in-memory _InMemoryStore with proper database-backed storage,
enabling horizontal scaling and data persistence across restarts.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Integer, Float, Boolean, Text, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class FineTuningJob(Base):
    """Fine-tuning job stored in database.

    Tracks the full lifecycle of a fine-tuning run: creation, training
    simulation, carbon budget enforcement, and completion/failure.
    Complex structured data (hyperparameters, carbon tracking, metrics)
    is stored as JSON in Text columns.
    """
    __tablename__ = "fine_tuning_jobs"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True,
        default=lambda: f"ft-{uuid.uuid4().hex[:24]}",
    )
    user_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )

    # Lifecycle: pending / running / completed / failed / cancelled
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending",
    )

    # Model & method
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    method: Mapped[str] = mapped_column(String(20), nullable=False, default="lora")

    # File references
    training_file_id: Mapped[str] = mapped_column(String(64), nullable=False)
    validation_file_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Training configuration & state — stored as JSON
    hyperparameters: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON object with hyperparameters",
    )
    carbon_tracking: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON object with carbon tracking data",
    )
    training_metrics: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON object with training metrics",
    )
    cost_estimate: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="JSON object with cost estimate",
    )

    # Result fields
    fine_tuned_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    suffix: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trained_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True, comment="JSON error details")

    # Webhook
    webhook_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    webhook_secret: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # User metadata
    metadata_json: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Arbitrary user-provided metadata as JSON",
    )

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
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )


class FineTuningFile(Base):
    """Uploaded training/validation file metadata.

    Stores the file metadata and raw content (for simulation purposes).
    In production, content would be stored in object storage (S3, GCS)
    and only the path would be persisted here.
    """
    __tablename__ = "fine_tuning_files"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True,
        default=lambda: f"file-{uuid.uuid4().hex[:24]}",
    )
    user_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(String(50), nullable=False, default="fine-tune")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="processed")
    status_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    line_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")

    # Raw file bytes — for training simulation only
    file_content: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True,
        comment="Raw file bytes (simulation only; use object storage in prod)",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class FineTunedModel(Base):
    """A fine-tuned model produced by a completed FineTuningJob.

    These models are available for inference via the chat completions
    endpoint.
    """
    __tablename__ = "fine_tuned_models"

    id: Mapped[str] = mapped_column(
        String(200), primary_key=True,
        comment="Model ID like 'harchos-llama-3.3-70b:my-suffix'",
    )
    user_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True,
    )

    base_model: Mapped[str] = mapped_column(String(100), nullable=False)
    fine_tuning_job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    method: Mapped[str] = mapped_column(String(20), nullable=False)

    # Lifecycle: deploying / ready / failed / archived
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="ready",
    )

    carbon_grams: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    suffix: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
