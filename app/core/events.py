"""Event and webhook system for HarchOS.

Provides:
- Event bus for internal pub/sub
- Webhook delivery for external notifications
- Event types for workload lifecycle, carbon optimization, and billing

Inspired by Replicate's webhook system and Together AI's event notifications.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

import httpx
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("harchos.events")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """All event types that HarchOS can emit."""
    # Workload lifecycle
    WORKLOAD_CREATED = "workload.created"
    WORKLOAD_SCHEDULED = "workload.scheduled"
    WORKLOAD_RUNNING = "workload.running"
    WORKLOAD_PAUSED = "workload.paused"
    WORKLOAD_COMPLETED = "workload.completed"
    WORKLOAD_FAILED = "workload.failed"
    WORKLOAD_CANCELLED = "workload.cancelled"
    WORKLOAD_DELETED = "workload.deleted"

    # Carbon optimization
    CARBON_OPTIMIZED = "carbon.optimized"
    CARBON_DEFERRED = "carbon.deferred"
    CARBON_BUDGET_EXCEEDED = "carbon.budget_exceeded"
    CARBON_GREEN_WINDOW = "carbon.green_window"

    # Hub events
    HUB_READY = "hub.ready"
    HUB_SCALING = "hub.scaling"
    HUB_OFFLINE = "hub.offline"
    HUB_ERROR = "hub.error"

    # Billing
    BILLING_RECORD_CREATED = "billing.record_created"
    BILLING_PAYMENT_PROCESSED = "billing.payment_processed"

    # Auth
    AUTH_API_KEY_CREATED = "auth.api_key_created"
    AUTH_API_KEY_REVOKED = "auth.api_key_revoked"


class Event(BaseModel):
    """A structured event with metadata."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any]
    user_id: str | None = None
    resource_id: str | None = None
    resource_type: str | None = None


class WebhookConfig(BaseModel):
    """Webhook subscription configuration."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    url: str
    events: list[EventType]
    secret: str = Field(default_factory=lambda: uuid.uuid4().hex)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Event Bus (in-process, async)
# ---------------------------------------------------------------------------

EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Simple in-process async event bus.

    Handlers are registered per event type and called concurrently.
    For production, replace with Redis Pub/Sub or a message queue.
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def on(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def on_any(self, handler: EventHandler) -> None:
        """Register a handler for all event types."""
        self._global_handlers.append(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        import asyncio

        handlers = list(self._handlers.get(event.type, [])) + self._global_handlers
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            try:
                tasks.append(asyncio.create_task(handler(event)))
            except Exception:
                logger.exception("Failed to create task for handler on event %s", event.type)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Event handler error for %s: %s", event.type, result)


# Global event bus instance
event_bus = EventBus()


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------

class WebhookDelivery:
    """Delivers events to external webhook endpoints with retry logic."""

    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 5, 30]  # seconds

    @staticmethod
    def sign_payload(payload: str, secret: str) -> str:
        """Sign a webhook payload with HMAC-SHA256."""
        return hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    async def deliver(
        webhook: WebhookConfig,
        event: Event,
    ) -> bool:
        """Deliver an event to a webhook URL with retry logic.

        Returns True if delivery succeeded, False otherwise.
        """
        payload = event.model_dump(mode="json")
        body = json.dumps(payload, default=str)
        signature = WebhookDelivery.sign_payload(body, webhook.secret)

        headers = {
            "Content-Type": "application/json",
            "X-HarchOS-Event": event.type.value,
            "X-HarchOS-Delivery": event.id,
            "X-HarchOS-Signature": f"sha256={signature}",
            "X-HarchOS-Timestamp": str(int(time.time())),
            "User-Agent": "HarchOS-Webhook/1.0",
        }

        for attempt in range(WebhookDelivery.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        webhook.url,
                        content=body,
                        headers=headers,
                    )
                    if 200 <= resp.status_code < 300:
                        logger.info(
                            "Webhook delivered: %s → %s (status=%d)",
                            event.type.value, webhook.url, resp.status_code,
                        )
                        return True
                    logger.warning(
                        "Webhook delivery failed: %s → %s (status=%d, attempt=%d)",
                        event.type.value, webhook.url, resp.status_code, attempt + 1,
                    )
            except httpx.HTTPError as exc:
                logger.warning(
                    "Webhook delivery error: %s → %s (%s, attempt=%d)",
                    event.type.value, webhook.url, exc, attempt + 1,
                )

            if attempt < WebhookDelivery.MAX_RETRIES - 1:
                import asyncio
                await asyncio.sleep(WebhookDelivery.RETRY_DELAYS[attempt])

        logger.error(
            "Webhook delivery failed after %d attempts: %s → %s",
            WebhookDelivery.MAX_RETRIES, event.type.value, webhook.url,
        )
        return False


# ---------------------------------------------------------------------------
# Helper: emit a workload event
# ---------------------------------------------------------------------------

async def emit_workload_event(
    event_type: EventType,
    workload_id: str,
    workload_name: str,
    user_id: str | None = None,
    hub_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a workload lifecycle event."""
    data = {
        "workload_id": workload_id,
        "workload_name": workload_name,
        "hub_id": hub_id,
    }
    if extra:
        data.update(extra)

    event = Event(
        type=event_type,
        data=data,
        user_id=user_id,
        resource_id=workload_id,
        resource_type="workload",
    )
    await event_bus.emit(event)


async def emit_carbon_event(
    event_type: EventType,
    workload_name: str,
    carbon_saved_kg: float | None = None,
    hub_name: str | None = None,
    user_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a carbon optimization event."""
    data = {
        "workload_name": workload_name,
        "carbon_saved_kg": carbon_saved_kg,
        "hub_name": hub_name,
    }
    if extra:
        data.update(extra)

    event = Event(
        type=event_type,
        data=data,
        user_id=user_id,
        resource_type="carbon_optimization",
    )
    await event_bus.emit(event)
