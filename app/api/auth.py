"""Auth endpoints – API key management, token exchange, user info, registration."""

import hashlib
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.auth import ApiKeyCreate, ApiKeyCreateResponse, TokenResponse, UserInfo
from app.services.auth_service import AuthService
from app.api.deps import require_auth, get_current_api_key
from app.config import settings

router = APIRouter()


# ---------------------------------------------------------------------------
# Additional schemas for login/register
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Login request body — exchange credentials for JWT token."""
    email: EmailStr = Field(..., description="User email")
    api_key: str = Field(..., description="API key (hsk_...)")


class RegisterRequest(BaseModel):
    """Register a new user and get an API key."""
    email: EmailStr = Field(..., description="User email")
    name: str = Field(..., min_length=1, max_length=255, description="Full name")


class RegisterResponse(BaseModel):
    """Response after successful registration."""
    user: UserInfo
    api_key: ApiKeyCreateResponse
    token: TokenResponse


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------

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
    """Exchange a valid API key for a JWT token.

    Send your API key via:
    - Authorization: Bearer hsk_xxxxx
    - X-API-Key: hsk_xxxxx
    """
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


# ---------------------------------------------------------------------------
# New endpoints: login and register
# ---------------------------------------------------------------------------

@router.post("/login", response_model=TokenResponse)
async def login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """Login with email + API key, get a JWT token.

    This is a convenience endpoint that combines user lookup and
    token exchange in a single call.
    """
    # Validate the API key
    api_key_obj = await AuthService.authenticate_api_key(db, data.api_key)
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Verify the user exists and email matches
    user_result = await db.execute(select(User).where(User.id == api_key_obj.user_id))
    user = user_result.scalar_one_or_none()
    if not user or user.email.lower() != data.email.lower():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email does not match API key owner",
        )

    # Create JWT token
    return AuthService.create_jwt_token(
        api_key_id=api_key_obj.id,
        user_id=api_key_obj.user_id,
    )


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    data: RegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user and receive an API key + JWT token.

    **Only available in development mode.** In production, user
    registration should be handled through an admin panel or
    invitation system to prevent abuse.
    """
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration is disabled in production. Contact admin@harchos.ai for access.",
        )

    # Check if email already exists
    existing = await db.execute(select(User).where(User.email == data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists",
        )

    # Create user
    user = User(
        email=data.email,
        name=data.name,
        is_active=True,
    )
    db.add(user)
    await db.flush()

    # Create default API key
    api_key_response = await AuthService.create_api_key(
        db, user_id=user.id, name="Default API Key"
    )

    # Create JWT token
    # We need the ApiKey object for token creation, fetch it
    api_key_obj_result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_response.id))
    api_key_obj = api_key_obj_result.scalar_one()

    token_response = AuthService.create_jwt_token(
        api_key_id=api_key_obj.id,
        user_id=user.id,
    )

    return RegisterResponse(
        user=UserInfo(
            id=user.id,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            created_at=user.created_at,
        ),
        api_key=api_key_response,
        token=token_response,
    )
