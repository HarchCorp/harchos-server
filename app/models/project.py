"""Project ORM model — project-scoped resource isolation and API key management.

Projects are the primary unit of isolation in HarchOS. Each project can have
its own API keys with scoped permissions, usage limits, and billing — similar
to Together AI's project-scoped keys but with more granular control.

Every API key can optionally belong to a project, which restricts its access
to only the resources within that project. This enables:
- Team isolation: different teams get different projects
- Budget control: per-project spending limits
- Scoped permissions: keys with only the scopes they need
- Model restrictions: limit which models a key can access
- Region pinning: restrict keys to specific regions for data sovereignty
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship, foreign

from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)

    # Owner — the user who created and manages this project
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # Tier — determines default rate limits and feature access
    # free: limited usage, standard: production workloads, enterprise: unlimited
    tier: Mapped[str] = mapped_column(String(20), nullable=False, default="free")

    # Active flag — deactivated projects block all their API keys
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Per-project usage limits (JSON) — overrides tier defaults
    # Example: {"max_rpm": 120, "max_concurrent_requests": 10, "max_tokens_per_day": 500000}
    usage_limits: Mapped[str | None] = mapped_column(Text, nullable=True)

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

    # Relationship to API keys (for eager loading if needed)
    api_keys: Mapped[list["ApiKey"]] = relationship(  # noqa: F821
        "ApiKey",
        back_populates="project",
        lazy="select",
        primaryjoin="Project.id == foreign(ApiKey.project_id)",
    )
