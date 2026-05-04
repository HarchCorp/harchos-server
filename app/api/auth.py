"""Auth endpoints – API key management, token exchange, user info."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.schemas.auth import ApiKeyCreate, ApiKeyCreateResponse, TokenResponse, UserInfo
from app.services.auth_service import AuthService
from app.api.deps import require_auth

router = APIRouter()

@router.post("/api-keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    data: ApiKeyCreate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key for the authenticated user."""
    return await AuthService.create_api_key(db, user_id=api_key.user_id, name=data.name)

@router.post("/token", response_model=TokenResponse)
async def exchange_api_key_for_token(
    api_key: ApiKey = Depends(require_auth),
):
    """Exchange a valid API key for a JWT token."""
    return AuthService.create_jwt_token(
        api_key_id=api_key.id,
        user_id=api_key.user_id,
    )

@router.get("/me", response_model=UserInfo)
async def get_current_user(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get current user info."""
    user_info = await AuthService.get_user_info(db, api_key.user_id)
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")
    return user_info
