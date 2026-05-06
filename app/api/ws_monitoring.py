"""WebSocket real-time monitoring endpoints for HarchOS.

Provides live-streaming endpoints for platform metrics, workload lifecycle
events, and carbon intensity updates.  All endpoints share a ConnectionManager
that tracks active connections, enforces authentication, supports per-client
subscription filters, and implements heartbeat / ping-pong keep-alive.

WebSocket endpoints
-------------------
- ``/ws/monitoring``  – Platform-wide metrics stream (GPU, hubs, energy) every 5 s
- ``/ws/workloads``   – Real-time workload status updates via the event bus
- ``/ws/carbon``      – Real-time carbon intensity updates every 10 s

Authentication
--------------
Clients authenticate by passing an ``api_key`` query parameter **or** by
sending a JSON ``{"type": "auth", "api_key": "hsk_..."}`` message as their
first message after connecting.  Unauthenticated connections are closed after
a 10-second grace period.

Subscription filtering
----------------------
After authentication, a client may send::

    {"type": "subscribe", "filters": ["hubs", "gpu", "energy"]}

Only data whose *category* matches one of the filters will be forwarded.
An empty or absent ``filters`` list means "receive everything".

Message format
--------------
Every server→client message is a JSON object with a ``type`` field::

    {"type": "metrics", "data": {...}}
    {"type": "workload_event", "data": {...}}
    {"type": "carbon_update", "data": {...}}
    {"type": "heartbeat", "timestamp": "..."}
    {"type": "error", "code": "E0100", "detail": "..."}
    {"type": "connected", "session_id": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.events import EventBus, Event, EventType, event_bus
from app.database import async_session_factory
from app.models.api_key import ApiKey
from app.models.carbon import CarbonIntensityRecord, CarbonOptimizationLog
from app.models.hub import Hub
from app.models.workload import Workload
from app.services.auth_service import AuthService

logger = logging.getLogger("harchos.ws_monitoring")

router = APIRouter()


# ---------------------------------------------------------------------------
# Message type constants
# ---------------------------------------------------------------------------

class WSMessageType(str, Enum):
    """WebSocket message type discriminators."""

    # Client → Server
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    PONG = "pong"
    UNSUBSCRIBE = "unsubscribe"

    # Server → Client
    CONNECTED = "connected"
    METRICS = "metrics"
    WORKLOAD_EVENT = "workload_event"
    CARBON_UPDATE = "carbon_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SUBSCRIPTION_ACK = "subscription_ack"


class MetricCategory(str, Enum):
    """Categories a client can subscribe to for the monitoring stream."""

    HUBS = "hubs"
    GPU = "gpu"
    ENERGY = "energy"
    CARBON = "carbon"
    WORKLOADS = "workloads"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Valid filter sets per endpoint
# ---------------------------------------------------------------------------

MONITORING_FILTERS: set[str] = {
    MetricCategory.HUBS.value,
    MetricCategory.GPU.value,
    MetricCategory.ENERGY.value,
    MetricCategory.CARBON.value,
    MetricCategory.WORKLOADS.value,
    MetricCategory.SYSTEM.value,
}

WORKLOAD_FILTERS: set[str] = {
    "created",
    "scheduled",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
    "deleted",
}

CARBON_FILTERS: set[str] = {
    "intensity",
    "forecast",
    "optimization",
    "green_window",
}


# ---------------------------------------------------------------------------
# Connection representation
# ---------------------------------------------------------------------------

@dataclass
class WSConnection:
    """Tracks state for a single WebSocket connection."""

    websocket: WebSocket
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    api_key_id: str | None = None
    user_id: str | None = None
    authenticated: bool = False
    filters: set[str] = field(default_factory=set)
    last_heartbeat: float = field(default_factory=time.monotonic)
    last_pong: float = field(default_factory=time.monotonic)
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    endpoint: str = ""  # "monitoring" | "workloads" | "carbon"


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages all active WebSocket connections across endpoints.

    Responsibilities:
    - Accept / register / disconnect connections
    - Authenticate connections (query param or first-message)
    - Periodic heartbeat (ping-pong) on a 30-second interval
    - Per-client subscription filtering
    - Broadcast helpers for each data type
    - Graceful cleanup on disconnect or auth failure
    """

    AUTH_GRACE_SECONDS: int = 10
    HEARTBEAT_INTERVAL_SECONDS: int = 30
    HEARTBEAT_TIMEOUT_SECONDS: int = 60  # Close if no pong within 60 s

    def __init__(self) -> None:
        # endpoint → session_id → WSConnection
        self._connections: dict[str, dict[str, WSConnection]] = {
            "monitoring": {},
            "workloads": {},
            "carbon": {},
        }
        self._heartbeat_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._event_bus_handler_registered: bool = False

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start background tasks (heartbeat loop, event-bus listener)."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("WebSocket heartbeat task started (interval=%ds)", self.HEARTBEAT_INTERVAL_SECONDS)

        if not self._event_bus_handler_registered:
            event_bus.on_any(self._on_event_bus_message)
            self._event_bus_handler_registered = True
            logger.info("WebSocket event-bus handler registered")

    async def stop(self) -> None:
        """Cancel background tasks and close all connections."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close every active connection
        for endpoint_conns in self._connections.values():
            for conn in list(endpoint_conns.values()):
                try:
                    await conn.websocket.close(code=1001, reason="Server shutting down")
                except Exception:
                    pass
            endpoint_conns.clear()

        logger.info("ConnectionManager stopped — all connections closed")

    # -- Register / Remove ---------------------------------------------------

    async def register(self, conn: WSConnection) -> None:
        """Add a connection to the manager."""
        async with self._lock:
            self._connections[conn.endpoint][conn.session_id] = conn
        logger.info(
            "WS connected: session=%s endpoint=%s (total=%d)",
            conn.session_id[:8],
            conn.endpoint,
            self.total_active,
        )

    async def remove(self, conn: WSConnection) -> None:
        """Remove a connection and clean up."""
        async with self._lock:
            self._connections[conn.endpoint].pop(conn.session_id, None)
        logger.info(
            "WS disconnected: session=%s endpoint=%s (total=%d)",
            conn.session_id[:8],
            conn.endpoint,
            self.total_active,
        )

    # -- Authentication ------------------------------------------------------

    async def authenticate_connection(
        self,
        conn: WSConnection,
        api_key_raw: str,
    ) -> bool:
        """Validate an API key and mark the connection as authenticated.

        Returns True on success; False on failure.
        """
        async with async_session_factory() as db:
            try:
                api_key_obj = await AuthService.authenticate_api_key(db, api_key_raw)
                if api_key_obj is None:
                    return False
                conn.authenticated = True
                conn.api_key_id = api_key_obj.id
                conn.user_id = api_key_obj.user_id
                return True
            except Exception:
                logger.exception("WS auth error for session=%s", conn.session_id[:8])
                return False

    async def wait_for_auth(self, conn: WSConnection) -> bool:
        """Wait up to AUTH_GRACE_SECONDS for the client to send an auth message.

        Used when no ``api_key`` query param was provided.
        """
        deadline = time.monotonic() + self.AUTH_GRACE_SECONDS
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(
                    conn.websocket.receive_text(),
                    timeout=self.AUTH_GRACE_SECONDS,
                )
            except (asyncio.TimeoutError, WebSocketDisconnect):
                return False

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(conn, "E0200", "Invalid JSON message")
                continue

            if msg.get("type") == WSMessageType.AUTH.value:
                api_key = msg.get("api_key", "")
                if await self.authenticate_connection(conn, api_key):
                    return True
                await self._send_error(conn, "E0101", "Invalid API key")
                return False

            # Non-auth message before authentication
            await self._send_error(conn, "E0100", "Authentication required before other messages")

        return False

    # -- Subscription filtering -----------------------------------------------

    async def handle_subscribe(self, conn: WSConnection, msg: dict[str, Any]) -> None:
        """Process a subscribe message from the client."""
        raw_filters = msg.get("filters", [])
        valid_set = self._valid_filters_for_endpoint(conn.endpoint)
        new_filters: set[str] = set()

        for f in raw_filters:
            if f in valid_set:
                new_filters.add(f)
            else:
                logger.warning(
                    "Invalid filter '%s' for endpoint '%s' (session=%s)",
                    f, conn.endpoint, conn.session_id[:8],
                )

        async with self._lock:
            conn.filters = new_filters

        await self._send_json(conn, {
            "type": WSMessageType.SUBSCRIPTION_ACK.value,
            "filters": sorted(conn.filters),
            "session_id": conn.session_id,
        })

    async def handle_unsubscribe(self, conn: WSConnection, msg: dict[str, Any]) -> None:
        """Process an unsubscribe message — clears all filters (revert to all)."""
        async with self._lock:
            conn.filters.clear()

        await self._send_json(conn, {
            "type": WSMessageType.SUBSCRIPTION_ACK.value,
            "filters": sorted(conn.filters),
            "session_id": conn.session_id,
            "note": "Filters cleared — receiving all data",
        })

    def should_send(self, conn: WSConnection, category: str) -> bool:
        """Check whether a connection should receive data for *category*.

        If the connection has no filters set, everything is forwarded.
        """
        if not conn.filters:
            return True
        return category in conn.filters

    # -- Broadcast helpers ---------------------------------------------------

    async def broadcast_monitoring(self, data: dict[str, Any]) -> None:
        """Send monitoring metrics to all connections on the monitoring endpoint."""
        await self._broadcast_to_endpoint("monitoring", WSMessageType.METRICS.value, data)

    async def broadcast_workload_event(self, data: dict[str, Any], category: str = "running") -> None:
        """Send a workload event to all connections on the workloads endpoint."""
        await self._broadcast_to_endpoint(
            "workloads", WSMessageType.WORKLOAD_EVENT.value, data, category=category,
        )

    async def broadcast_carbon_update(self, data: dict[str, Any], category: str = "intensity") -> None:
        """Send a carbon update to all connections on the carbon endpoint."""
        await self._broadcast_to_endpoint(
            "carbon", WSMessageType.CARBON_UPDATE.value, data, category=category,
        )

    # -- Heartbeat -----------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat pings and close stale connections."""
        while True:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL_SECONDS)
                now = time.monotonic()
                stale_sessions: list[WSConnection] = []

                for endpoint_conns in self._connections.values():
                    for conn in list(endpoint_conns.values()):
                        if not conn.authenticated:
                            continue

                        # Check for stale (no pong received within timeout)
                        if now - conn.last_pong > self.HEARTBEAT_TIMEOUT_SECONDS:
                            stale_sessions.append(conn)
                            continue

                        # Send heartbeat
                        try:
                            await self._send_json(conn, {
                                "type": WSMessageType.HEARTBEAT.value,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            conn.last_heartbeat = now
                        except Exception:
                            stale_sessions.append(conn)

                # Remove stale connections
                for conn in stale_sessions:
                    logger.warning(
                        "Closing stale WS connection: session=%s endpoint=%s",
                        conn.session_id[:8], conn.endpoint,
                    )
                    try:
                        await conn.websocket.close(code=1000, reason="Heartbeat timeout")
                    except Exception:
                        pass
                    await self.remove(conn)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Heartbeat loop error")

    # -- Event bus listener --------------------------------------------------

    async def _on_event_bus_message(self, event: Event) -> None:
        """Forward relevant events from the internal event bus to WS clients."""
        try:
            data = event.model_dump(mode="json")

            if event.type.startswith("workload."):
                status = event.type.value.split(".", 1)[1]  # e.g. "running"
                await self.broadcast_workload_event(data, category=status)

            elif event.type.startswith("carbon."):
                category = "intensity"
                if "green_window" in event.type.value:
                    category = "green_window"
                elif "optimized" in event.type.value or "deferred" in event.type.value:
                    category = "optimization"
                elif "budget" in event.type.value:
                    category = "optimization"
                await self.broadcast_carbon_update(data, category=category)

            elif event.type.startswith("hub."):
                # Hub events go to monitoring endpoint under "hubs" category
                await self.broadcast_monitoring(data)

        except Exception:
            logger.exception("Event-bus → WS forwarding error for event %s", event.type)

    # -- Internal helpers ----------------------------------------------------

    async def _broadcast_to_endpoint(
        self,
        endpoint: str,
        message_type: str,
        data: dict[str, Any],
        category: str | None = None,
    ) -> None:
        """Send a typed message to all connections on an endpoint, respecting filters."""
        disconnected: list[WSConnection] = []

        for conn in list(self._connections.get(endpoint, {}).values()):
            if not conn.authenticated:
                continue
            if category and not self.should_send(conn, category):
                continue

            try:
                await self._send_json(conn, {
                    "type": message_type,
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            except Exception:
                disconnected.append(conn)

        # Clean up disconnected
        for conn in disconnected:
            await self.remove(conn)

    async def _send_json(self, conn: WSConnection, payload: dict[str, Any]) -> None:
        """Send a JSON message to a single connection."""
        await conn.websocket.send_text(json.dumps(payload, default=str))

    async def _send_error(self, conn: WSConnection, code: str, detail: str) -> None:
        """Send an error message to a single connection."""
        await self._send_json(conn, {
            "type": WSMessageType.ERROR.value,
            "code": code,
            "detail": detail,
        })

    def _valid_filters_for_endpoint(self, endpoint: str) -> set[str]:
        """Return the set of valid filter values for a given endpoint."""
        if endpoint == "monitoring":
            return MONITORING_FILTERS
        elif endpoint == "workloads":
            return WORKLOAD_FILTERS
        elif endpoint == "carbon":
            return CARBON_FILTERS
        return set()

    # -- Properties ----------------------------------------------------------

    @property
    def total_active(self) -> int:
        """Total number of active connections across all endpoints."""
        return sum(len(conns) for conns in self._connections.values())

    def connections_for(self, endpoint: str) -> list[WSConnection]:
        """Return a snapshot of connections for a given endpoint."""
        return list(self._connections.get(endpoint, {}).values())


# ---------------------------------------------------------------------------
# Global connection manager instance
# ---------------------------------------------------------------------------

manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

async def collect_platform_metrics() -> dict[str, Any]:
    """Collect current platform metrics from the database.

    Returns a dictionary suitable for inclusion in a ``metrics`` WS message.
    """
    async with async_session_factory() as db:
        try:
            # Hub counts
            hub_count_result = await db.execute(select(func.count(Hub.id)))
            total_hubs = hub_count_result.scalar() or 0

            total_gpus_result = await db.execute(select(func.sum(Hub.total_gpus)))
            total_gpus = total_gpus_result.scalar() or 0

            available_gpus_result = await db.execute(select(func.sum(Hub.available_gpus)))
            available_gpus = available_gpus_result.scalar() or 0

            avg_renewable_result = await db.execute(select(func.avg(Hub.renewable_percentage)))
            avg_renewable = float(avg_renewable_result.scalar() or 0.0)

            avg_carbon_result = await db.execute(select(func.avg(Hub.grid_carbon_intensity)))
            avg_carbon = float(avg_carbon_result.scalar() or 0.0)

            avg_pue_result = await db.execute(select(func.avg(Hub.pue)))
            avg_pue = float(avg_pue_result.scalar() or 1.0)

            # Workload counts
            total_workloads_result = await db.execute(select(func.count(Workload.id)))
            total_workloads = total_workloads_result.scalar() or 0

            active_workloads_result = await db.execute(
                select(func.count(Workload.id)).where(
                    Workload.status.in_(["running", "scheduled"])
                )
            )
            active_workloads = active_workloads_result.scalar() or 0

            # CO2 savings
            co2_saved_result = await db.execute(
                select(func.sum(CarbonOptimizationLog.carbon_saved_kg))
            )
            total_co2_saved = float(co2_saved_result.scalar() or 0.0)

            # GPU utilization
            gpu_utilization = (
                ((total_gpus - available_gpus) / total_gpus * 100)
                if total_gpus > 0
                else 0.0
            )

            # Hub details for per-hub status
            hub_rows = await db.execute(
                select(Hub.id, Hub.name, Hub.status, Hub.region, Hub.total_gpus,
                       Hub.available_gpus, Hub.renewable_percentage,
                       Hub.grid_carbon_intensity)
                .order_by(Hub.name)
            )
            hubs_detail = []
            for row in hub_rows.all():
                hubs_detail.append({
                    "id": row[0],
                    "name": row[1],
                    "status": row[2],
                    "region": row[3],
                    "total_gpus": row[4],
                    "available_gpus": row[5],
                    "renewable_percentage": round(row[6], 2),
                    "grid_carbon_intensity": round(row[7], 2),
                })

            return {
                "hubs": {
                    "total": total_hubs,
                    "detail": hubs_detail,
                },
                "gpu": {
                    "total": total_gpus,
                    "available": available_gpus,
                    "in_use": total_gpus - available_gpus,
                    "utilization_percent": round(gpu_utilization, 2),
                },
                "workloads": {
                    "total": total_workloads,
                    "active": active_workloads,
                },
                "energy": {
                    "avg_renewable_percentage": round(avg_renewable, 2),
                    "avg_pue": round(avg_pue, 4),
                    "estimated_energy_kwh": round(
                        (total_gpus - available_gpus) * 0.3 * 24, 2
                    ),
                },
                "carbon": {
                    "avg_intensity_gco2_kwh": round(avg_carbon, 2),
                    "total_co2_saved_kg": round(total_co2_saved, 4),
                    "green_threshold_gco2_kwh": settings.carbon_green_threshold_gco2_kwh,
                },
                "system": {
                    "app_version": settings.app_version,
                    "environment": settings.environment,
                    "ws_connections": manager.total_active,
                },
            }
        except Exception:
            logger.exception("Error collecting platform metrics")
            return {
                "error": "Failed to collect metrics",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


async def collect_carbon_data() -> dict[str, Any]:
    """Collect current carbon intensity data from the database.

    Returns a dictionary suitable for inclusion in a ``carbon_update`` WS message.
    """
    async with async_session_factory() as db:
        try:
            # Latest carbon intensity records per zone
            latest_records = await db.execute(
                select(CarbonIntensityRecord)
                .order_by(CarbonIntensityRecord.datetime.desc())
                .limit(20)
            )
            intensity_data = []
            for record in latest_records.scalars().all():
                intensity_data.append({
                    "zone": record.zone,
                    "carbon_intensity_gco2_kwh": record.carbon_intensity_gco2_kwh,
                    "renewable_percentage": record.renewable_percentage,
                    "fossil_percentage": record.fossil_percentage,
                    "source": record.source,
                    "is_forecast": record.is_forecast,
                    "datetime": record.datetime.isoformat() if record.datetime else None,
                })

            # Recent optimization logs
            recent_opts = await db.execute(
                select(CarbonOptimizationLog)
                .order_by(CarbonOptimizationLog.created_at.desc())
                .limit(10)
            )
            optimization_data = []
            for opt in recent_opts.scalars().all():
                optimization_data.append({
                    "workload_id": opt.workload_id,
                    "workload_name": opt.workload_name,
                    "action": opt.action,
                    "selected_hub_name": opt.selected_hub_name,
                    "carbon_saved_kg": round(opt.carbon_saved_kg, 4),
                    "carbon_intensity_at_schedule_gco2_kwh": round(
                        opt.carbon_intensity_at_schedule_gco2_kwh, 2
                    ),
                    "reason": opt.reason,
                    "created_at": opt.created_at.isoformat() if opt.created_at else None,
                })

            # Aggregate stats
            avg_intensity_result = await db.execute(
                select(func.avg(Hub.grid_carbon_intensity))
            )
            avg_intensity = float(avg_intensity_result.scalar() or 0.0)

            total_saved_result = await db.execute(
                select(func.sum(CarbonOptimizationLog.carbon_saved_kg))
            )
            total_saved = float(total_saved_result.scalar() or 0.0)

            # Check for green windows (zones below threshold)
            green_zones = [
                r for r in intensity_data
                if not r.get("is_forecast", False)
                and r.get("carbon_intensity_gco2_kwh", float("inf"))
                <= settings.carbon_green_threshold_gco2_kwh
            ]

            return {
                "intensity": {
                    "zones": intensity_data,
                    "avg_gco2_kwh": round(avg_intensity, 2),
                },
                "forecast": [
                    r for r in intensity_data if r.get("is_forecast", False)
                ],
                "optimization": {
                    "recent": optimization_data,
                    "total_co2_saved_kg": round(total_saved, 4),
                },
                "green_window": {
                    "zones_below_threshold": len(green_zones),
                    "threshold_gco2_kwh": settings.carbon_green_threshold_gco2_kwh,
                    "green_zone_names": [r["zone"] for r in green_zones],
                },
            }
        except Exception:
            logger.exception("Error collecting carbon data")
            return {
                "error": "Failed to collect carbon data",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


async def collect_workload_snapshot() -> dict[str, Any]:
    """Collect a current snapshot of workload statuses.

    Returns a dictionary with workload counts by status plus recent events.
    """
    async with async_session_factory() as db:
        try:
            # Count by status
            status_counts_result = await db.execute(
                select(Workload.status, func.count(Workload.id))
                .group_by(Workload.status)
            )
            status_counts = {row[0]: row[1] for row in status_counts_result.all()}

            # Recent workloads
            recent_result = await db.execute(
                select(
                    Workload.id, Workload.name, Workload.type, Workload.status,
                    Workload.priority, Workload.hub_id, Workload.gpu_count,
                    Workload.carbon_aware, Workload.started_at, Workload.updated_at,
                )
                .order_by(Workload.updated_at.desc())
                .limit(25)
            )
            recent_workloads = []
            for row in recent_result.all():
                recent_workloads.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "status": row[3],
                    "priority": row[4],
                    "hub_id": row[5],
                    "gpu_count": row[6],
                    "carbon_aware": row[7],
                    "started_at": row[8].isoformat() if row[8] else None,
                    "updated_at": row[9].isoformat() if row[9] else None,
                })

            return {
                "status_counts": status_counts,
                "total": sum(status_counts.values()),
                "recent": recent_workloads,
            }
        except Exception:
            logger.exception("Error collecting workload snapshot")
            return {
                "error": "Failed to collect workload data",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# ---------------------------------------------------------------------------
# Periodic broadcast tasks
# ---------------------------------------------------------------------------

async def monitoring_broadcast_loop() -> None:
    """Broadcast platform metrics every 5 seconds to monitoring connections."""
    while True:
        try:
            await asyncio.sleep(5)
            conns = manager.connections_for("monitoring")
            if not conns:
                continue

            data = await collect_platform_metrics()
            await manager.broadcast_monitoring(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Monitoring broadcast loop error")
            await asyncio.sleep(5)


async def carbon_broadcast_loop() -> None:
    """Broadcast carbon data every 10 seconds to carbon connections."""
    while True:
        try:
            await asyncio.sleep(10)
            conns = manager.connections_for("carbon")
            if not conns:
                continue

            data = await collect_carbon_data()
            await manager.broadcast_carbon_update(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Carbon broadcast loop error")
            await asyncio.sleep(10)


# Track background tasks so we can cancel on shutdown
_background_tasks: list[asyncio.Task] = []


async def start_background_tasks() -> None:
    """Start the periodic broadcast and heartbeat tasks."""
    await manager.start()

    task_monitoring = asyncio.create_task(monitoring_broadcast_loop())
    task_carbon = asyncio.create_task(carbon_broadcast_loop())

    _background_tasks.extend([task_monitoring, task_carbon])
    logger.info("WS monitoring background tasks started (monitoring=5s, carbon=10s)")


async def stop_background_tasks() -> None:
    """Cancel all background tasks and stop the connection manager."""
    for task in _background_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _background_tasks.clear()

    await manager.stop()
    logger.info("WS monitoring background tasks stopped")


# ---------------------------------------------------------------------------
# Client message handler
# ---------------------------------------------------------------------------

async def handle_client_messages(conn: WSConnection) -> None:
    """Main loop that reads messages from a single WebSocket client.

    Handles: auth (already done), subscribe, unsubscribe, pong.
    Any other message type is ignored with a warning.
    """
    while True:
        try:
            raw = await conn.websocket.receive_text()
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.warning(
                "WS receive error: session=%s endpoint=%s",
                conn.session_id[:8], conn.endpoint,
            )
            raise

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await manager._send_error(conn, "E0200", "Invalid JSON message")
            continue

        msg_type = msg.get("type", "")

        if msg_type == WSMessageType.PONG.value:
            conn.last_pong = time.monotonic()

        elif msg_type == WSMessageType.SUBSCRIBE.value:
            await manager.handle_subscribe(conn, msg)

        elif msg_type == WSMessageType.UNSUBSCRIBE.value:
            await manager.handle_unsubscribe(conn, msg)

        elif msg_type == WSMessageType.AUTH.value:
            # Re-authentication mid-session
            api_key = msg.get("api_key", "")
            if await manager.authenticate_connection(conn, api_key):
                await manager._send_json(conn, {
                    "type": WSMessageType.CONNECTED.value,
                    "session_id": conn.session_id,
                    "authenticated": True,
                })
            else:
                await manager._send_error(conn, "E0101", "Invalid API key")

        else:
            logger.debug(
                "Unhandled WS message type '%s' from session=%s",
                msg_type, conn.session_id[:8],
            )


# ---------------------------------------------------------------------------
# WebSocket endpoint: /ws/monitoring
# ---------------------------------------------------------------------------

@router.websocket("/monitoring")
async def ws_monitoring(
    websocket: WebSocket,
    api_key: str | None = Query(default=None, description="API key for authentication"),
) -> None:
    """Real-time platform metrics stream.

    Broadcasts GPU utilization, hub status, energy, and carbon metrics
    every 5 seconds.  Clients can subscribe to specific metric categories
    via ``{"type": "subscribe", "filters": ["hubs", "gpu"]}``.

    Valid filters: hubs, gpu, energy, carbon, workloads, system
    """
    conn = WSConnection(websocket=websocket, endpoint="monitoring")

    await websocket.accept()
    await manager.register(conn)

    try:
        # --- Authentication ---
        if api_key:
            auth_ok = await manager.authenticate_connection(conn, api_key)
        else:
            auth_ok = await manager.wait_for_auth(conn)

        if not auth_ok:
            await manager._send_error(conn, "E0100", "Authentication failed or timed out")
            await websocket.close(code=4001, reason="Authentication required")
            return

        # Send connected confirmation
        await manager._send_json(conn, {
            "type": WSMessageType.CONNECTED.value,
            "session_id": conn.session_id,
            "authenticated": True,
            "endpoint": "monitoring",
            "interval_seconds": 5,
            "available_filters": sorted(MONITORING_FILTERS),
        })

        # Send an immediate snapshot so the client doesn't wait 5 seconds
        initial_data = await collect_platform_metrics()
        await manager._send_json(conn, {
            "type": WSMessageType.METRICS.value,
            "data": initial_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # --- Main message loop ---
        await handle_client_messages(conn)

    except WebSocketDisconnect:
        logger.debug("WS monitoring client disconnected: session=%s", conn.session_id[:8])
    except Exception:
        logger.exception(
            "Unexpected error in WS /monitoring: session=%s", conn.session_id[:8]
        )
    finally:
        await manager.remove(conn)


# ---------------------------------------------------------------------------
# WebSocket endpoint: /ws/workloads
# ---------------------------------------------------------------------------

@router.websocket("/workloads")
async def ws_workloads(
    websocket: WebSocket,
    api_key: str | None = Query(default=None, description="API key for authentication"),
) -> None:
    """Real-time workload status updates.

    Forwards workload lifecycle events from the internal event bus (created,
    scheduled, running, paused, completed, failed, cancelled, deleted).
    Clients can filter by status via
    ``{"type": "subscribe", "filters": ["running", "failed"]}``.

    On connection, an initial snapshot of current workload statuses is sent.
    """
    conn = WSConnection(websocket=websocket, endpoint="workloads")

    await websocket.accept()
    await manager.register(conn)

    try:
        # --- Authentication ---
        if api_key:
            auth_ok = await manager.authenticate_connection(conn, api_key)
        else:
            auth_ok = await manager.wait_for_auth(conn)

        if not auth_ok:
            await manager._send_error(conn, "E0100", "Authentication failed or timed out")
            await websocket.close(code=4001, reason="Authentication required")
            return

        # Send connected confirmation
        await manager._send_json(conn, {
            "type": WSMessageType.CONNECTED.value,
            "session_id": conn.session_id,
            "authenticated": True,
            "endpoint": "workloads",
            "available_filters": sorted(WORKLOAD_FILTERS),
        })

        # Send an initial workload snapshot
        snapshot = await collect_workload_snapshot()
        await manager._send_json(conn, {
            "type": WSMessageType.WORKLOAD_EVENT.value,
            "data": snapshot,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot": True,
        })

        # --- Main message loop ---
        await handle_client_messages(conn)

    except WebSocketDisconnect:
        logger.debug("WS workloads client disconnected: session=%s", conn.session_id[:8])
    except Exception:
        logger.exception(
            "Unexpected error in WS /workloads: session=%s", conn.session_id[:8]
        )
    finally:
        await manager.remove(conn)


# ---------------------------------------------------------------------------
# WebSocket endpoint: /ws/carbon
# ---------------------------------------------------------------------------

@router.websocket("/carbon")
async def ws_carbon(
    websocket: WebSocket,
    api_key: str | None = Query(default=None, description="API key for authentication"),
) -> None:
    """Real-time carbon intensity updates.

    Broadcasts carbon intensity readings, forecasts, optimization logs, and
    green-window alerts every 10 seconds.  Clients can filter by category
    via ``{"type": "subscribe", "filters": ["intensity", "green_window"]}``.

    Valid filters: intensity, forecast, optimization, green_window
    """
    conn = WSConnection(websocket=websocket, endpoint="carbon")

    await websocket.accept()
    await manager.register(conn)

    try:
        # --- Authentication ---
        if api_key:
            auth_ok = await manager.authenticate_connection(conn, api_key)
        else:
            auth_ok = await manager.wait_for_auth(conn)

        if not auth_ok:
            await manager._send_error(conn, "E0100", "Authentication failed or timed out")
            await websocket.close(code=4001, reason="Authentication required")
            return

        # Send connected confirmation
        await manager._send_json(conn, {
            "type": WSMessageType.CONNECTED.value,
            "session_id": conn.session_id,
            "authenticated": True,
            "endpoint": "carbon",
            "interval_seconds": 10,
            "available_filters": sorted(CARBON_FILTERS),
        })

        # Send an immediate carbon snapshot
        initial_data = await collect_carbon_data()
        await manager._send_json(conn, {
            "type": WSMessageType.CARBON_UPDATE.value,
            "data": initial_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # --- Main message loop ---
        await handle_client_messages(conn)

    except WebSocketDisconnect:
        logger.debug("WS carbon client disconnected: session=%s", conn.session_id[:8])
    except Exception:
        logger.exception(
            "Unexpected error in WS /carbon: session=%s", conn.session_id[:8]
        )
    finally:
        await manager.remove(conn)
