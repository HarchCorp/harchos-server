"""Hubs endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.schemas.hub import HubCreate, HubUpdate, HubResponse, HubCapacity
from app.schemas.common import PaginatedResponse
from app.services.hub_service import HubService
from app.api.deps import require_auth, get_current_api_key
from app.config import settings

router = APIRouter()


@router.get("", response_model=PaginatedResponse[HubResponse])
async def list_hubs(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size, description="Items per page"),
    status: str | None = Query(None, description="Filter by status"),
    tier: str | None = Query(None, description="Filter by tier"),
    region: str | None = Query(None, description="Filter by region"),
    api_key: ApiKey | None = Depends(get_current_api_key),
    db: AsyncSession = Depends(get_db),
):
    """List hubs with pagination. Public endpoint (auth optional)."""
    return await HubService.list_hubs(
        db, page=page, per_page=per_page, status=status, tier=tier, region=region
    )


@router.post("", response_model=HubResponse, status_code=status.HTTP_201_CREATED)
async def create_hub(
    data: HubCreate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a new hub."""
    return await HubService.create_hub(db, data)


@router.get("/{hub_id}", response_model=HubResponse)
async def get_hub(
    hub_id: str,
    api_key: ApiKey | None = Depends(get_current_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Get a hub by ID. Public endpoint."""
    result = await HubService.get_hub(db, hub_id)
    if not result:
        raise HTTPException(status_code=404, detail="Hub not found")
    return result


@router.patch("/{hub_id}", response_model=HubResponse)
async def update_hub(
    hub_id: str,
    data: HubUpdate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update a hub."""
    result = await HubService.update_hub(db, hub_id, data)
    if not result:
        raise HTTPException(status_code=404, detail="Hub not found")
    return result


@router.delete("/{hub_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_hub(
    hub_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Delete a hub."""
    deleted = await HubService.delete_hub(db, hub_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Hub not found")


@router.get("/{hub_id}/capacity", response_model=HubCapacity)
async def get_hub_capacity(
    hub_id: str,
    api_key: ApiKey | None = Depends(get_current_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Get capacity info for a specific hub."""
    result = await HubService.get_hub_capacity(db, hub_id)
    if not result:
        raise HTTPException(status_code=404, detail="Hub not found")
    return result
