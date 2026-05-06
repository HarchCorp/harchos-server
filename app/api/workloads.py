"""Workloads endpoints with multi-tenancy, RBAC, and event emission."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.models.workload import Workload
from app.schemas.workload import WorkloadCreate, WorkloadUpdate, WorkloadResponse
from app.schemas.common import PaginatedResponse
from app.services.workload_service import WorkloadService
from app.api.deps import require_auth, require_write_access, get_current_user
from app.config import settings
from app.core.exceptions import not_found, validation_error, invalid_enum_value
from app.core.enums import WorkloadType, WorkloadStatus, WorkloadPriority
from app.core.events import emit_workload_event, EventType
from app.core.audit import audit_log

router = APIRouter()

# Valid enum values for quick validation
VALID_WORKLOAD_TYPES = [e.value for e in WorkloadType]
VALID_WORKLOAD_STATUSES = [e.value for e in WorkloadStatus]
VALID_PRIORITIES = [e.value for e in WorkloadPriority]


@router.get("", response_model=PaginatedResponse[WorkloadResponse])
async def list_workloads(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size, description="Items per page"),
    status: str | None = Query(None, description="Filter by status"),
    type: str | None = Query(None, alias="type", description="Filter by workload type"),
    hub_id: str | None = Query(None, description="Filter by hub ID"),
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List workloads with pagination.

    - Regular users see only their own workloads
    - Admins see all workloads
    """
    # Validate filter values
    if status and status not in VALID_WORKLOAD_STATUSES:
        raise invalid_enum_value("status", status, VALID_WORKLOAD_STATUSES)
    if type and type not in VALID_WORKLOAD_TYPES:
        raise invalid_enum_value("type", type, VALID_WORKLOAD_TYPES)

    # Non-admin users can only see their own workloads
    user_id_filter = None if user.is_admin else api_key.user_id

    return await WorkloadService.list_workloads(
        db, page=page, per_page=per_page, status=status,
        workload_type=type, hub_id=hub_id, user_id=user_id_filter,
    )


@router.get("/active", response_model=PaginatedResponse[WorkloadResponse])
async def list_active_workloads(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size, description="Items per page"),
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List currently active workloads (running and scheduled).

    Returns only workloads that are actively consuming GPU resources,
    sorted by creation time (most recent first).
    """
    user_id_filter = None if user.is_admin else api_key.user_id
    return await WorkloadService.list_workloads(
        db, page=page, per_page=per_page,
        status="running,scheduled",
        user_id=user_id_filter,
    )


@router.post("", response_model=WorkloadResponse, status_code=status.HTTP_201_CREATED)
async def create_workload(
    data: WorkloadCreate,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_write_access),
    db: AsyncSession = Depends(get_db),
):
    """Create a new workload.

    Requires write access (user or admin role). The workload is
    automatically associated with the authenticated user.
    """
    # Validate enum fields
    if data.type not in VALID_WORKLOAD_TYPES:
        raise invalid_enum_value("type", data.type, VALID_WORKLOAD_TYPES)
    if data.priority not in VALID_PRIORITIES:
        raise invalid_enum_value("priority", data.priority, VALID_PRIORITIES)

    result = await WorkloadService.create_workload(
        db, data, user_id=api_key.user_id,
    )

    # Emit event
    await emit_workload_event(
        EventType.WORKLOAD_CREATED,
        workload_id=result.metadata.id,
        workload_name=data.name,
        user_id=api_key.user_id,
        hub_id=data.hub_id,
    )

    # Audit log
    audit_log(
        action="create",
        resource_type="workload",
        resource_id=result.metadata.id,
        actor_id=api_key.user_id,
        details={"name": data.name, "type": data.type, "gpu_count": data.compute.gpu_count},
    )

    return result


@router.get("/stats")
async def workload_stats(
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get workload statistics for the authenticated user.

    - Regular users see only their own stats
    - Admins see platform-wide stats

    Performance: Uses a single GROUP BY query instead of 4 separate queries.
    """
    user_id_filter = None if user.is_admin else api_key.user_id

    # Single optimized GROUP BY query (was 4 separate queries)
    status_counts_query = (
        select(Workload.status, func.count(Workload.id).label("count"))
        .group_by(Workload.status)
    )
    if user_id_filter:
        status_counts_query = status_counts_query.where(Workload.user_id == user_id_filter)

    result = await db.execute(status_counts_query)
    rows = result.all()

    # Build counts from GROUP BY result
    counts = {"total": 0, "active": 0, "completed": 0, "failed": 0}
    for row in rows:
        counts["total"] += row.count
        if row.status in ("running", "scheduled"):
            counts["active"] += row.count
        elif row.status == "completed":
            counts["completed"] = row.count
        elif row.status == "failed":
            counts["failed"] = row.count

    return counts


@router.get("/{workload_id}", response_model=WorkloadResponse)
async def get_workload(
    workload_id: str,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a workload by ID.

    Regular users can only access their own workloads.
    """
    result = await WorkloadService.get_workload(db, workload_id)
    if not result:
        raise not_found("workload", workload_id)

    # Authorization check: non-admin users can only see their own workloads
    # Single DB query for user_id check (not fetching full Workload object)
    if not user.is_admin:
        wl = await db.execute(select(Workload.user_id).where(Workload.id == workload_id))
        wl_user_id = wl.scalar_one_or_none()
        if wl_user_id and wl_user_id != api_key.user_id:
            raise not_found("workload", workload_id)  # Return 404, not 403, to avoid info leak

    return result


@router.patch("/{workload_id}", response_model=WorkloadResponse)
async def update_workload(
    workload_id: str,
    data: WorkloadUpdate,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_write_access),
    db: AsyncSession = Depends(get_db),
):
    """Update a workload.

    Regular users can only update their own workloads.
    """
    # Validate status if provided
    if data.status and data.status not in VALID_WORKLOAD_STATUSES:
        raise invalid_enum_value("status", data.status, VALID_WORKLOAD_STATUSES)

    # Authorization check: non-admin users can only update their own workloads
    # Only select user_id, not the full Workload object
    if not user.is_admin:
        wl = await db.execute(select(Workload.user_id).where(Workload.id == workload_id))
        wl_user_id = wl.scalar_one_or_none()
        if wl_user_id is None:
            raise not_found("workload", workload_id)
        if wl_user_id != api_key.user_id:
            raise not_found("workload", workload_id)

    result = await WorkloadService.update_workload(db, workload_id, data)
    if not result:
        raise not_found("workload", workload_id)

    # Emit event based on status change
    if data.status:
        event_map = {
            "running": EventType.WORKLOAD_RUNNING,
            "completed": EventType.WORKLOAD_COMPLETED,
            "failed": EventType.WORKLOAD_FAILED,
            "cancelled": EventType.WORKLOAD_CANCELLED,
            "scheduled": EventType.WORKLOAD_SCHEDULED,
            "paused": EventType.WORKLOAD_PAUSED,
        }
        event_type = event_map.get(data.status)
        if event_type:
            await emit_workload_event(
                event_type, workload_id=workload_id,
                workload_name=result.spec.name,
                user_id=api_key.user_id,
            )

    # Audit log
    audit_log(
        action="update",
        resource_type="workload",
        resource_id=workload_id,
        actor_id=api_key.user_id,
        details=data.model_dump(exclude_unset=True),
    )

    return result


@router.delete("/{workload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workload(
    workload_id: str,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_write_access),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workload.

    Regular users can only delete their own workloads.
    """
    # Authorization check: non-admin users can only delete their own workloads
    # Only select user_id, not the full Workload object
    if not user.is_admin:
        wl = await db.execute(select(Workload.user_id).where(Workload.id == workload_id))
        wl_user_id = wl.scalar_one_or_none()
        if wl_user_id is None:
            raise not_found("workload", workload_id)
        if wl_user_id != api_key.user_id:
            raise not_found("workload", workload_id)

    deleted = await WorkloadService.delete_workload(db, workload_id)
    if not deleted:
        raise not_found("workload", workload_id)

    # Emit event
    await emit_workload_event(
        EventType.WORKLOAD_DELETED, workload_id=workload_id,
        workload_name="", user_id=api_key.user_id,
    )

    # Audit log
    audit_log(
        action="delete",
        resource_type="workload",
        resource_id=workload_id,
        actor_id=api_key.user_id,
    )
