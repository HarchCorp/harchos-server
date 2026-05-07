"""Auth endpoints – API key management, token exchange, user info, registration, key revocation."""

import hashlib
import hmac
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.auth import ApiKeyCreate, ApiKeyCreateResponse, TokenResponse, UserInfo
from app.services.auth_service import AuthService
from app.api.deps import require_auth, get_current_user, require_admin
from app.config import settings
from app.core.exceptions import (
    HarchOSError,
    already_exists,
    not_found,
    invalid_api_key,
)
from app.core.enums import UserRole
from app.core.audit import audit_log
from app.core.events import event_bus, EventType, emit_workload_event

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
    role: str = Field("user", description="User role: admin, user, or viewer")


class RegisterResponse(BaseModel):
    """Response after successful registration."""
    user: UserInfo
    api_key: ApiKeyCreateResponse
    token: TokenResponse


class AdminBootstrapRequest(BaseModel):
    """Request body for admin bootstrap — one-time setup."""
    email: EmailStr = Field(..., description="Admin email address")
    name: str = Field(..., min_length=1, max_length=255, description="Admin full name")
    bootstrap_token: str = Field(..., description="Bootstrap token from HARCHOS_ADMIN_BOOTSTRAP_TOKEN env var")


class ApiKeyRevokeResponse(BaseModel):
    """Response after revoking an API key."""
    id: str
    name: str
    revoked: bool = True


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
    result = await AuthService.create_api_key(
        db, user_id=api_key.user_id, name=data.name, tier=api_key.tier,
    )

    audit_log(
        action="create",
        resource_type="api_key",
        resource_id=result.id,
        actor_id=api_key.user_id,
        details={"name": data.name},
    )

    return result


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
async def get_current_user_info(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get current user info."""
    user_info = await AuthService.get_user_info(db, api_key.user_id)
    if not user_info:
        raise not_found("user", api_key.user_id)
    return user_info


# ---------------------------------------------------------------------------
# API key revocation
# ---------------------------------------------------------------------------

@router.delete("/api-keys/{key_id}", response_model=ApiKeyRevokeResponse)
async def revoke_api_key(
    key_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Revoke (deactivate) an API key by its ID.

    Users can only revoke their own API keys. Admins can revoke any key.
    """
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    target_key = result.scalar_one_or_none()

    if not target_key:
        raise not_found("api_key", key_id)

    # Authorization: only own keys or admin
    if target_key.user_id != api_key.user_id:
        user_result = await db.execute(select(User).where(User.id == api_key.user_id))
        user = user_result.scalar_one_or_none()
        if not user or not user.is_admin:
            raise not_found("api_key", key_id)  # 404, not 403

    target_key.is_active = False
    await db.flush()

    audit_log(
        action="revoke",
        resource_type="api_key",
        resource_id=key_id,
        actor_id=api_key.user_id,
    )

    return ApiKeyRevokeResponse(
        id=target_key.id,
        name=target_key.name,
        revoked=True,
    )


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
        raise invalid_api_key()

    # Verify the user exists and email matches
    user_result = await db.execute(select(User).where(User.id == api_key_obj.user_id))
    user = user_result.scalar_one_or_none()
    if not user or user.email.lower() != data.email.lower():
        raise HarchOSError("E0103", detail="Email does not match API key owner")

    if not user.is_active:
        raise HarchOSError("E0101", detail="Account is deactivated")

    audit_log(
        action="login",
        resource_type="user",
        resource_id=user.id,
        actor_id=user.id,
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
        raise HarchOSError("E0105", detail="Public registration is disabled in production. Contact admin@harchos.ai for access.")

    # Validate role — ALWAYS reject admin role, even in dev
    valid_roles = [r.value for r in UserRole]
    if data.role not in valid_roles:
        raise HarchOSError("E0201", detail=f"Invalid role '{data.role}'. Allowed: {', '.join(valid_roles)}", meta={"allowed": valid_roles})

    # SECURITY: Never allow admin role via public registration, even in dev
    if data.role == "admin":
        raise HarchOSError(
            "E0201",
            detail="Admin role cannot be assigned via registration. Use the admin panel or CLI.",
            meta={"requested_role": "admin", "allowed_via_register": ["user", "viewer"]},
        )

    # Check if email already exists (case-insensitive to prevent
    # Test@Example.COM / test@example.com duplicates)
    existing = await db.execute(
        select(User).where(func.lower(User.email) == data.email.lower())
    )
    if existing.scalar_one_or_none():
        raise already_exists("user", "email")

    # Create user with role — normalize email to lowercase for storage
    user = User(
        email=data.email.lower(),
        name=data.name,
        is_active=True,
        role=data.role,
    )
    db.add(user)
    await db.flush()

    # Create default API key with standard tier for registered users
    api_key_response = await AuthService.create_api_key(
        db, user_id=user.id, name="Default API Key", tier="standard",
    )

    # Create JWT token
    api_key_obj_result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_response.id))
    api_key_obj = api_key_obj_result.scalar_one()

    token_response = AuthService.create_jwt_token(
        api_key_id=api_key_obj.id,
        user_id=user.id,
    )

    audit_log(
        action="register",
        resource_type="user",
        resource_id=user.id,
        details={"email": data.email, "role": data.role},
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


@router.post("/bootstrap", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def admin_bootstrap(
    data: AdminBootstrapRequest,
    db: AsyncSession = Depends(get_db),
):
    """Bootstrap the first admin user for production setup.

    This endpoint ONLY works when:
    1. No admin user exists in the database yet
    2. The HARCHOS_ADMIN_BOOTSTRAP_TOKEN env var is set and matches the request token

    This provides a secure way to create the initial admin in production
    without leaving registration open. After the first admin is created,
    this endpoint returns 403 permanently.

    To use: Set HARCHOS_ADMIN_BOOTSTRAP_TOKEN in your environment, then:
    POST /v1/auth/bootstrap
    {
        "email": "admin@yourcompany.com",
        "name": "Your Name",
        "bootstrap_token": "your-secret-token"
    }
    """
    # Verify bootstrap token
    bootstrap_token = getattr(settings, "admin_bootstrap_token", "")
    if not bootstrap_token:
        raise HarchOSError(
            "E0105",
            detail="Admin bootstrap is not configured. Set HARCHOS_ADMIN_BOOTSTRAP_TOKEN to enable one-time admin setup.",
        )

    if not hmac.compare_digest(data.bootstrap_token, bootstrap_token):
        raise HarchOSError("E0101", detail="Invalid bootstrap token")

    # Check if any admin already exists
    admin_result = await db.execute(
        select(User).where(User.role == "admin")
    )
    existing_admin = admin_result.scalar_one_or_none()
    if existing_admin:
        raise HarchOSError(
            "E0105",
            detail="Admin user already exists. Bootstrap is a one-time operation. Use the admin panel to manage users.",
        )

    # Check if email already exists (case-insensitive)
    existing = await db.execute(
        select(User).where(func.lower(User.email) == data.email.lower())
    )
    if existing.scalar_one_or_none():
        raise already_exists("user", "email")

    # Create admin user — normalize email to lowercase for storage
    user = User(
        email=data.email.lower(),
        name=data.name,
        is_active=True,
        role="admin",
    )
    db.add(user)
    await db.flush()

    # Create admin API key with enterprise tier
    api_key_response = await AuthService.create_api_key(
        db, user_id=user.id, name="Admin API Key", tier="enterprise",
        scopes=["inference:write", "workloads:write", "carbon:read", "admin:full"],
    )

    # Create JWT token
    api_key_obj_result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_response.id))
    api_key_obj = api_key_obj_result.scalar_one()

    token_response = AuthService.create_jwt_token(
        api_key_id=api_key_obj.id,
        user_id=user.id,
    )

    audit_log(
        action="admin_bootstrap",
        resource_type="user",
        resource_id=user.id,
        details={"email": data.email, "role": "admin"},
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
