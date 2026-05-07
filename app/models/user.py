"""User ORM model with RBAC role support."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")  # admin / user / viewer
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == "admin"

    @property
    def is_viewer(self) -> bool:
        """Check if user has viewer-only role."""
        return self.role == "viewer"
