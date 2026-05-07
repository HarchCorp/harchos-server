"""Auth dependency for API key / JWT token validation with RBAC and project-scoped support.

Provides:
- get_current_api_key: Optional auth (returns ApiKey | None)
- require_auth: Mandatory auth (returns ApiKey)
- require_admin: Requires admin role
- require_write_access: Requires user or admin role (viewers are read-only)
- get_current_user: Returns the User object for the authenticated session
- get_current_project: Returns the Project if the API key is project-scoped
- require_project_access: Requires access to a specific project
- require_scope: Factory that creates a dependency checking for a specific scope
"""

import json
from typing import Callable

from fastapi import Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.project import Project
from app.models.user import User
from app.services.auth_service import AuthService
from app.config import settings
from app.core.exceptions import (
    auth_required,
    invalid_api_key,
    invalid_token,
    insufficient_permissions,
    project_not_found,
    project_access_denied,
    project_inactive,
    insufficient_scope,
    token_budget_exceeded,
    spending_limit_exceeded,
    model_not_allowed,
    region_not_allowed,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def _parse_json_field(value: str | None, default=None):
    """Safely parse a JSON text field from the database."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


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


# ---------------------------------------------------------------------------
# Project-scoped auth dependencies
# ---------------------------------------------------------------------------

async def get_current_project(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
) -> Project | None:
    """Dependency that returns the Project if the API key is project-scoped.

    Returns None for global (non-project-scoped) keys.
    Raises if the key's project is inactive or not found.
    """
    if api_key.project_id is None:
        return None

    result = await db.execute(select(Project).where(Project.id == api_key.project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise project_not_found(api_key.project_id)

    if not project.is_active:
        raise project_inactive(api_key.project_id)

    return project


async def require_project_access(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
    project_id: str = "",
) -> Project:
    """Dependency that requires access to a specific project.

    The authenticated user must be the owner of the project.
    For project-scoped API keys, the key's project_id must match.

    Note: When using this dependency, pass project_id explicitly or
    ensure it's available as a path parameter in the route.
    In most cases, you should use _get_user_project() directly
    in the endpoint function instead.
    """
    if not project_id:
        raise project_not_found("")

    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise project_not_found(project_id)

    # User must own the project
    if project.user_id != api_key.user_id:
        raise project_access_denied(project_id)

    # If the key is project-scoped, it can only access its own project
    if api_key.project_id is not None and api_key.project_id != project_id:
        raise project_access_denied(project_id)

    if not project.is_active:
        raise project_inactive(project_id)

    return project


def require_scope(required_scope: str) -> Callable:
    """Factory that creates a dependency checking if the API key has a specific scope.

    Usage:
        @router.get("/inference", dependencies=[Depends(require_scope("inference:read"))])

    If the key has no scopes set (null/empty), it has full access (backward compatible).
    If the key has scopes set, the required scope must be in the list.

    The scope format is 'resource:action' (e.g. 'inference:read', 'workloads:write').
    A 'resource:write' scope implies 'resource:read' access.
    """
    async def _check_scope(
        api_key: ApiKey = Depends(require_auth),
    ) -> ApiKey:
        scopes_raw = _parse_json_field(api_key.scopes)

        # No scopes set = full access (backward compatible)
        if scopes_raw is None or len(scopes_raw) == 0:
            return api_key

        # Check if the required scope is present
        if required_scope in scopes_raw:
            return api_key

        # Check for write implies read: if we need "resource:read" and have "resource:write"
        resource, action = required_scope.split(":")
        if action == "read" and f"{resource}:write" in scopes_raw:
            return api_key

        raise insufficient_scope(required_scope)

    return _check_scope


async def check_model_access(
    api_key: ApiKey,
    model_id: str,
) -> None:
    """Check if an API key is allowed to access a specific model.

    Raises model_not_allowed if the model is not in the allowed list.
    """
    allowed_models = _parse_json_field(api_key.allowed_models)
    if allowed_models is None or len(allowed_models) == 0:
        return  # No restrictions

    if model_id not in allowed_models:
        raise model_not_allowed(model_id)


async def check_region_access(
    api_key: ApiKey,
    region: str,
) -> None:
    """Check if an API key is allowed to access a specific region.

    Raises region_not_allowed if the region is not in the allowed list.
    """
    allowed_regions = _parse_json_field(api_key.allowed_regions)
    if allowed_regions is None or len(allowed_regions) == 0:
        return  # No restrictions

    if region not in allowed_regions:
        raise region_not_allowed(region)


async def check_token_budget(
    api_key: ApiKey,
    tokens_to_use: int = 0,
) -> None:
    """Check if an API key has remaining token budget for today.

    Raises token_budget_exceeded if the daily limit is reached.
    """
    if api_key.max_tokens_per_day is None:
        return  # Unlimited

    if api_key.tokens_used_today + tokens_to_use > api_key.max_tokens_per_day:
        raise token_budget_exceeded(api_key.tokens_used_today, api_key.max_tokens_per_day)


async def check_spending_limit(
    api_key: ApiKey,
    additional_cost_usd: float = 0.0,
) -> None:
    """Check if an API key has remaining spending limit for this month.

    Raises spending_limit_exceeded if the monthly limit is reached.
    """
    if api_key.spending_limit_monthly_usd is None:
        return  # Unlimited

    if api_key.spent_this_month_usd + additional_cost_usd > api_key.spending_limit_monthly_usd:
        raise spending_limit_exceeded(api_key.spent_this_month_usd, api_key.spending_limit_monthly_usd)
