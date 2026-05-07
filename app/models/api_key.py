"""API Key ORM model with project scoping, tiered access, and fine-grained permissions.

API keys can optionally be scoped to a project, which restricts their access
to only the resources within that project. Project-scoped keys support:
- Scoped permissions (read/write per resource type)
- Model restrictions (limit which models a key can access)
- Region pinning (restrict keys to specific regions for data sovereignty)
- Token budgets (daily token limits)
- Spending limits (monthly USD caps)
"""

import uuid
import hashlib
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Boolean, Float, Integer, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # -----------------------------------------------------------------------
    # Project scoping — nullable means global key (access to all user projects)
    # -----------------------------------------------------------------------
    project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id"), nullable=True, index=True,
    )

    # -----------------------------------------------------------------------
    # Tier — determines default rate limits if not overridden by project
    # -----------------------------------------------------------------------
    tier: Mapped[str] = mapped_column(String(20), nullable=False, default="free")

    # -----------------------------------------------------------------------
    # Fine-grained scopes (JSON array of strings)
    # e.g. ["inference:read", "inference:write", "workloads:read",
    #       "workloads:write", "carbon:read", "models:read", "billing:read"]
    # Empty/null = all scopes (backward compatible)
    # -----------------------------------------------------------------------
    scopes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -----------------------------------------------------------------------
    # Model restrictions (JSON array of model IDs)
    # e.g. ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mixtral-8x7B"]
    # Empty/null = all models allowed
    # -----------------------------------------------------------------------
    allowed_models: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -----------------------------------------------------------------------
    # Region restrictions (JSON array of region identifiers)
    # e.g. ["europe-west", "africa-north"]
    # Empty/null = all regions allowed
    # -----------------------------------------------------------------------
    allowed_regions: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -----------------------------------------------------------------------
    # Token budgets — daily token usage tracking
    # -----------------------------------------------------------------------
    max_tokens_per_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_used_today: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # -----------------------------------------------------------------------
    # Spending limits — monthly USD tracking
    # -----------------------------------------------------------------------
    spending_limit_monthly_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    spent_this_month_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # -----------------------------------------------------------------------
    # Relationships
    # -----------------------------------------------------------------------
    project: Mapped["Project | None"] = relationship(  # noqa: F821
        "Project",
        back_populates="api_keys",
        lazy="select",
    )

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key using SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()
