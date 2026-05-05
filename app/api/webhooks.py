"""Webhook management endpoints.

Users can register webhook URLs to receive event notifications
when workloads change status, carbon optimization happens, etc.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.api.deps import require_auth, require_write_access, get_current_user
from app.core.events import EventType, WebhookConfig, WebhookDelivery, event_bus
from app.core.exceptions import not_found, validation_error

router = APIRouter()


# ---------------------------------------------------------------------------
# Database model for webhooks (lightweight, uses SQLAlchemy)
# ---------------------------------------------------------------------------

from app.database import Base
from sqlalchemy import String, DateTime, Boolean, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column


class Webhook(Base):
    """Webhook subscription stored in database."""
    __tablename__ = "webhooks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    events: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array of event types
    secret: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class WebhookCreate(BaseModel):
    """Create a webhook subscription."""
    url: str = Field(..., description="Webhook endpoint URL (must be HTTPS in production)")
    events: list[str] = Field(..., min_length=1, description="List of event types to subscribe to")


class WebhookResponse(BaseModel):
    """Webhook subscription response."""
    id: str
    url: str
    events: list[str]
    secret: str
    is_active: bool
    created_at: datetime


class WebhookTestResponse(BaseModel):
    """Response from webhook test delivery."""
    success: bool
    status_code: int | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    data: WebhookCreate,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_write_access),
    db: AsyncSession = Depends(get_db),
):
    """Register a webhook URL to receive event notifications.

    The webhook will receive POST requests with event data whenever
    one of the subscribed event types occurs. Each request includes
    an HMAC-SHA256 signature in the X-HarchOS-Signature header.
    """
    # Validate event types
    valid_events = {e.value for e in EventType}
    for event in data.events:
        if event not in valid_events:
            raise validation_error("events", f"Unknown event type: {event}. Valid types: {', '.join(sorted(valid_events))}")

    # In production, enforce HTTPS
    if user.is_admin is False and data.url.startswith("http://"):
        from app.config import settings
        if settings.is_production:
            raise validation_error("url", "Webhook URLs must use HTTPS in production")

    secret = uuid.uuid4().hex
    events_json = __import__("json").dumps(data.events)

    webhook = Webhook(
        user_id=api_key.user_id,
        url=data.url,
        events=events_json,
        secret=secret,
        is_active=True,
    )
    db.add(webhook)
    await db.flush()

    return WebhookResponse(
        id=webhook.id,
        url=webhook.url,
        events=data.events,
        secret=secret,
        is_active=webhook.is_active,
        created_at=webhook.created_at,
    )


@router.get("", response_model=list[WebhookResponse])
async def list_webhooks(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List all webhooks for the authenticated user."""
    result = await db.execute(
        select(Webhook).where(Webhook.user_id == api_key.user_id)
    )
    webhooks = result.scalars().all()

    return [
        WebhookResponse(
            id=w.id,
            url=w.url,
            events=__import__("json").loads(w.events),
            secret=w.secret,
            is_active=w.is_active,
            created_at=w.created_at,
        )
        for w in webhooks
    ]


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Delete a webhook subscription."""
    result = await db.execute(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == api_key.user_id)
    )
    webhook = result.scalar_one_or_none()
    if not webhook:
        raise not_found("webhook", webhook_id)

    await db.delete(webhook)
    await db.flush()


@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(
    webhook_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Test a webhook by sending a ping event.

    Sends a test event to the webhook URL to verify connectivity.
    """
    result = await db.execute(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == api_key.user_id)
    )
    webhook = result.scalar_one_or_none()
    if not webhook:
        raise not_found("webhook", webhook_id)

    from app.core.events import Event
    test_event = Event(
        type=EventType.WORKLOAD_CREATED,
        data={"test": True, "message": "Webhook test from HarchOS"},
        user_id=api_key.user_id,
    )

    config = WebhookConfig(
        id=webhook.id,
        user_id=webhook.user_id,
        url=webhook.url,
        events=__import__("json").loads(webhook.events),
        secret=webhook.secret,
    )

    success = await WebhookDelivery.deliver(config, test_event)

    return WebhookTestResponse(
        success=success,
        error=None if success else "Delivery failed after retries",
    )
