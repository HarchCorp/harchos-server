"""Models endpoints (ML models) with admin-only mutation operations.

Security: Creating, updating, and deleting models requires admin role.
Listing models requires authentication. Model catalog is managed by admins.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.model import ModelCreate, ModelUpdate, ModelResponse
from app.schemas.common import PaginatedResponse
from app.services.model_service import ModelService
from app.api.deps import require_auth, require_admin, get_current_user
from app.config import settings

router = APIRouter()


@router.get("", response_model=PaginatedResponse[ModelResponse])
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size, description="Items per page"),
    framework: str | None = Query(None, description="Filter by framework"),
    status: str | None = Query(None, description="Filter by status"),
    hub_id: str | None = Query(None, description="Filter by hub ID"),
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List models with pagination. Requires authentication."""
    return await ModelService.list_models(
        db, page=page, per_page=per_page, framework=framework, status=status, hub_id=hub_id
    )


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(
    data: ModelCreate,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_admin),  # Admin-only: model catalog management
    db: AsyncSession = Depends(get_db),
):
    """Create a new model. Requires admin role.

    Model catalog management is an infrastructure-level operation.
    Only administrators can add, update, or remove models from the catalog.
    """
    return await ModelService.create_model(db, data)


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get a model by ID."""
    result = await ModelService.get_model(db, model_id)
    if not result:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.patch("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: str,
    data: ModelUpdate,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_admin),  # Admin-only: model catalog management
    db: AsyncSession = Depends(get_db),
):
    """Update a model. Requires admin role."""
    result = await ModelService.update_model(db, model_id, data)
    if not result:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    api_key: ApiKey = Depends(require_auth),
    user: User = Depends(require_admin),  # Admin-only: model catalog management
    db: AsyncSession = Depends(get_db),
):
    """Delete a model. Requires admin role."""
    deleted = await ModelService.delete_model(db, model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model not found")
