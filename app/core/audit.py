"""Audit logging for HarchOS.

Every mutation (create, update, delete) on key resources is logged
with the actor, action, resource, and timestamp for compliance and debugging.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("harchos.audit")


class AuditEntry(BaseModel):
    """A single audit log entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: str | None = None
    actor_email: str | None = None
    action: str  # create, update, delete, login, etc.
    resource_type: str  # workload, hub, model, api_key, etc.
    resource_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None


def audit_log(
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    actor_id: str | None = None,
    actor_email: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    """Log an audit entry.

    Currently logs to the Python logger in structured JSON format.
    In production, this should also write to a dedicated audit table
    or external service (e.g., Datadog, Splunk).
    """
    entry = AuditEntry(
        actor_id=actor_id,
        actor_email=actor_email,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
    )
    logger.info(
        "AUDIT %s %s/%s actor=%s details=%s",
        entry.action,
        entry.resource_type,
        entry.resource_id or "*",
        entry.actor_id or "anonymous",
        json.dumps(entry.details, default=str),
    )
