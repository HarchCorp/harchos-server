"""Auth dependency for API key / JWT token validation."""

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.services.auth_service import AuthService
from app.config import settings


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
                from sqlalchemy import select
                result = await db.execute(
                    select(ApiKey).where(ApiKey.id == api_key_id, ApiKey.is_active == True)
                )
                return result.scalar_one_or_none()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

    if key_to_check:
        api_key_obj = await AuthService.authenticate_api_key(db, key_to_check)
        if api_key_obj:
            return api_key_obj
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # No auth provided – return None (some endpoints may be public)
    return None


async def require_auth(
    api_key: ApiKey | None = Depends(get_current_api_key),
) -> ApiKey:
    """Dependency that requires authentication (raises 401 if not authenticated)."""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide X-API-Key header or Bearer token.",
        )
    return api_key
