"""Workloads endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.workload import Workload
from app.schemas.workload import WorkloadCreate, WorkloadUpdate, WorkloadResponse
from app.schemas.common import PaginatedResponse
from app.services.workload_service import WorkloadService
from app.api.deps import require_auth
from app.config import settings

router = APIRouter()


@router.get("", response_model=PaginatedResponse[WorkloadResponse])
async def list_workloads(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size, description="Items per page"),
    status: str | None = Query(None, description="Filter by status"),
    type: str | None = Query(None, alias="type", description="Filter by workload type"),
    hub_id: str | None = Query(None, description="Filter by hub ID"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List workloads with pagination."""
    return await WorkloadService.list_workloads(
        db, page=page, per_page=per_page, status=status, workload_type=type, hub_id=hub_id
    )


@router.get("/active", response_model=PaginatedResponse[WorkloadResponse])
async def list_active_workloads(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size, description="Items per page"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List currently active workloads (running or scheduled).

    Returns only workloads that are actively consuming GPU resources,
    sorted by creation time (most recent first).
    """
    return await WorkloadService.list_workloads(
        db, page=page, per_page=per_page, status="running",
    )


@router.post("", response_model=WorkloadResponse, status_code=status.HTTP_201_CREATED)
async def create_workload(
    data: WorkloadCreate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a new workload."""
    return await WorkloadService.create_workload(db, data)


@router.get("/stats")
async def workload_stats(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get workload statistics for the authenticated user."""
    # Total workloads
    total_result = await db.execute(select(func.count(Workload.id)))
    total = total_result.scalar() or 0

    # Active workloads
    active_result = await db.execute(
        select(func.count(Workload.id)).where(
            Workload.status.in_(["running", "scheduled"])
        )
    )
    active = active_result.scalar() or 0

    # Completed workloads
    completed_result = await db.execute(
        select(func.count(Workload.id)).where(Workload.status == "completed")
    )
    completed = completed_result.scalar() or 0

    # Failed workloads
    failed_result = await db.execute(
        select(func.count(Workload.id)).where(Workload.status == "failed")
    )
    failed = failed_result.scalar() or 0

    return {
        "total": total,
        "active": active,
        "completed": completed,
        "failed": failed,
    }


@router.get("/{workload_id}", response_model=WorkloadResponse)
async def get_workload(
    workload_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get a workload by ID."""
    result = await WorkloadService.get_workload(db, workload_id)
    if not result:
        raise HTTPException(status_code=404, detail="Workload not found")
    return result


@router.patch("/{workload_id}", response_model=WorkloadResponse)
async def update_workload(
    workload_id: str,
    data: WorkloadUpdate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update a workload."""
    result = await WorkloadService.update_workload(db, workload_id, data)
    if not result:
        raise HTTPException(status_code=404, detail="Workload not found")
    return result


@router.delete("/{workload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workload(
    workload_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workload."""
    deleted = await WorkloadService.delete_workload(db, workload_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Workload not found")
