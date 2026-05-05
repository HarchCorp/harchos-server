"""Auth dependency for API key / JWT token validation with RBAC support.

Provides:
- get_current_api_key: Optional auth (returns ApiKey | None)
- require_auth: Mandatory auth (returns ApiKey)
- require_admin: Requires admin role
- require_write_access: Requires user or admin role (viewers are read-only)
- get_current_user: Returns the User object for the authenticated session
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.services.auth_service import AuthService
from app.config import settings
from app.core.exceptions import (
    auth_required,
    invalid_api_key,
    invalid_token,
    insufficient_permissions,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_api_key(
    api_key: str | None = Depends(api_key_header),
    bearer_creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> ApiKey | None:
    """
    Dependency that extracts and validates an API key or JWT token.
    Returns the ApiKey object if authenticated, None if no auth provided.
    """
    key_to_check = None

    # 1. Check X-API-Key header
    if api_key and api_key.startswith(settings.api_key_prefix):
        key_to_check = api_key

    # 2. Check Authorization: Bearer hsk_... (API key) or hst_... (JWT)
    elif bearer_creds:
        token = bearer_creds.credentials
        if token.startswith(settings.api_key_prefix):
            key_to_check = token
        elif token.startswith(settings.token_prefix):
            # JWT token validation
            payload = AuthService.verify_jwt_token(token)
            if payload:
                api_key_id = payload.get("api_key_id")
                result = await db.execute(
                    select(ApiKey).where(ApiKey.id == api_key_id, ApiKey.is_active.is_(True))
                )
                return result.scalar_one_or_none()
            raise invalid_token()

    if key_to_check:
        api_key_obj = await AuthService.authenticate_api_key(db, key_to_check)
        if api_key_obj:
            return api_key_obj
        raise invalid_api_key()

    # No auth provided – return None (some endpoints may be public)
    return None


async def require_auth(
    api_key: ApiKey | None = Depends(get_current_api_key),
) -> ApiKey:
    """Dependency that requires authentication (raises 401 if not authenticated)."""
    if api_key is None:
        raise auth_required()
    return api_key


async def get_current_user(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency that returns the authenticated User object."""
    result = await db.execute(select(User).where(User.id == api_key.user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise auth_required()
    if not user.is_active:
        raise invalid_api_key()
    return user


async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Dependency that requires admin role."""
    if not user.is_admin:
        raise insufficient_permissions(required_role="admin")
    return user


async def require_write_access(
    user: User = Depends(get_current_user),
) -> User:
    """Dependency that requires write access (user or admin role).

    Viewers can only read, not create/update/delete.
    """
    if user.is_viewer:
        raise insufficient_permissions(required_role="user")
    return user
