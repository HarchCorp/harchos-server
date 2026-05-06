"""Project endpoints — project-scoped API key management, a major security feature.

Projects provide resource isolation, budget control, and fine-grained
permissions for API keys. This is similar to Together AI's project-scoped
keys but with more granular control including:
- Scoped permissions (read/write per resource type)
- Model restrictions (limit which models a key can access)
- Region pinning (restrict keys to specific regions for data sovereignty)
- Token budgets (daily token limits)
- Spending limits (monthly USD caps)
"""

import json

from fastapi import APIRouter, Depends, status
from sqlalchemy import select, func, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey
from app.models.project import Project
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectApiKeyCreate,
    ProjectApiKeyResponse,
    ProjectApiKeyCreateResponse,
    ProjectUsageResponse,
)
from app.api.deps import require_auth, get_current_user
from app.services.auth_service import AuthService
from app.core.exceptions import (
    project_not_found,
    project_access_denied,
    project_inactive,
    invalid_enum_value,
)
from app.core.enums import ProjectTier, ApiKeyTier
from app.core.audit import audit_log

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_field(value: str | None, default=None):
    """Safely parse a JSON text field from the database."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _dump_json_field(value) -> str | None:
    """Serialize a value to JSON string for database storage."""
    if value is None:
        return None
    return json.dumps(value)


async def _get_user_project(
    project_id: str,
    user_id: str,
    db: AsyncSession,
) -> Project:
    """Get a project and verify the user owns it. Raises on not found / access denied."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise project_not_found(project_id)

    if project.user_id != user_id:
        raise project_access_denied(project_id)

    return project


def _project_to_response(project: Project) -> ProjectResponse:
    """Convert a Project ORM object to a response schema."""
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        user_id=project.user_id,
        tier=project.tier,
        is_active=project.is_active,
        usage_limits=_parse_json_field(project.usage_limits),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


def _api_key_to_project_response(api_key: ApiKey) -> ProjectApiKeyResponse:
    """Convert an ApiKey ORM object to a project-scoped response schema."""
    return ProjectApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        project_id=api_key.project_id,
        tier=api_key.tier,
        scopes=_parse_json_field(api_key.scopes),
        allowed_models=_parse_json_field(api_key.allowed_models),
        allowed_regions=_parse_json_field(api_key.allowed_regions),
        max_tokens_per_day=api_key.max_tokens_per_day,
        tokens_used_today=api_key.tokens_used_today,
        spending_limit_monthly_usd=api_key.spending_limit_monthly_usd,
        spent_this_month_usd=api_key.spent_this_month_usd,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
    )


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create a new project.

    Projects are the primary unit of isolation. Each project can have its own
    API keys with scoped permissions, usage limits, and billing.
    """
    # Validate tier
    valid_tiers = [t.value for t in ProjectTier]
    if data.tier not in valid_tiers:
        raise invalid_enum_value("tier", data.tier, valid_tiers)

    project = Project(
        name=data.name,
        description=data.description,
        user_id=api_key.user_id,
        tier=data.tier,
        is_active=True,
        usage_limits=_dump_json_field(data.usage_limits),
    )
    db.add(project)
    await db.flush()

    audit_log(
        action="create",
        resource_type="project",
        resource_id=project.id,
        actor_id=api_key.user_id,
        details={"name": data.name, "tier": data.tier},
    )

    return _project_to_response(project)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List all projects owned by the authenticated user."""
    result = await db.execute(
        select(Project)
        .where(Project.user_id == api_key.user_id)
        .order_by(Project.created_at.desc())
    )
    projects = result.scalars().all()
    return [_project_to_response(p) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific project."""
    project = await _get_user_project(project_id, api_key.user_id, db)
    return _project_to_response(project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    data: ProjectUpdate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update a project's settings.

    Only the project owner can update the project. Deactivating a project
    will block all API keys scoped to that project.
    """
    project = await _get_user_project(project_id, api_key.user_id, db)

    # Apply updates
    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.tier is not None:
        valid_tiers = [t.value for t in ProjectTier]
        if data.tier not in valid_tiers:
            raise invalid_enum_value("tier", data.tier, valid_tiers)
        project.tier = data.tier
    if data.is_active is not None:
        project.is_active = data.is_active
    if data.usage_limits is not None:
        project.usage_limits = _dump_json_field(data.usage_limits)

    await db.flush()

    audit_log(
        action="update",
        resource_type="project",
        resource_id=project.id,
        actor_id=api_key.user_id,
        details=data.model_dump(exclude_none=True),
    )

    return _project_to_response(project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Delete a project and deactivate all its API keys.

    **Warning**: This is a hard delete. All API keys scoped to this project
    will be deactivated. Consider deactivating the project instead.
    """
    project = await _get_user_project(project_id, api_key.user_id, db)

    # Deactivate all API keys belonging to this project
    keys_result = await db.execute(
        select(ApiKey).where(ApiKey.project_id == project_id)
    )
    for key in keys_result.scalars().all():
        key.is_active = False

    # Delete the project
    await db.delete(project)
    await db.flush()

    audit_log(
        action="delete",
        resource_type="project",
        resource_id=project_id,
        actor_id=api_key.user_id,
        details={"name": project.name},
    )


# ---------------------------------------------------------------------------
# Project-scoped API keys
# ---------------------------------------------------------------------------

@router.get("/{project_id}/api-keys", response_model=list[ProjectApiKeyResponse])
async def list_project_api_keys(
    project_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys scoped to a specific project."""
    project = await _get_user_project(project_id, api_key.user_id, db)

    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.project_id == project_id)
        .order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [_api_key_to_project_response(k) for k in keys]


@router.post("/{project_id}/api-keys", response_model=ProjectApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_project_api_key(
    project_id: str,
    data: ProjectApiKeyCreate,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Create an API key scoped to a specific project.

    The key will only have access to the resources permitted by its scopes,
    allowed models, allowed regions, and budget limits.
    """
    project = await _get_user_project(project_id, api_key.user_id, db)

    if not project.is_active:
        raise project_inactive(project_id)

    # Validate tier
    valid_tiers = [t.value for t in ApiKeyTier]
    if data.tier not in valid_tiers:
        raise invalid_enum_value("tier", data.tier, valid_tiers)

    # Generate the key
    raw_key = AuthService.generate_api_key()
    key_hash = AuthService.hash_key(raw_key)
    key_prefix = raw_key[:8]

    new_key = ApiKey(
        user_id=api_key.user_id,
        name=data.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        is_active=True,
        project_id=project_id,
        tier=data.tier,
        scopes=_dump_json_field(data.scopes),
        allowed_models=_dump_json_field(data.allowed_models),
        allowed_regions=_dump_json_field(data.allowed_regions),
        max_tokens_per_day=data.max_tokens_per_day,
        tokens_used_today=0,
        spending_limit_monthly_usd=data.spending_limit_monthly_usd,
        spent_this_month_usd=0.0,
        expires_at=data.expires_at,
    )
    db.add(new_key)
    await db.flush()

    audit_log(
        action="create",
        resource_type="api_key",
        resource_id=new_key.id,
        actor_id=api_key.user_id,
        details={
            "name": data.name,
            "project_id": project_id,
            "tier": data.tier,
            "scopes": data.scopes,
        },
    )

    return ProjectApiKeyCreateResponse(
        id=new_key.id,
        name=data.name,
        key=raw_key,
        key_prefix=key_prefix,
        is_active=True,
        project_id=project_id,
        tier=data.tier,
        scopes=data.scopes,
        allowed_models=data.allowed_models,
        allowed_regions=data.allowed_regions,
        max_tokens_per_day=data.max_tokens_per_day,
        spending_limit_monthly_usd=data.spending_limit_monthly_usd,
        created_at=new_key.created_at,
    )


# ---------------------------------------------------------------------------
# Project usage stats
# ---------------------------------------------------------------------------

@router.get("/{project_id}/usage", response_model=ProjectUsageResponse)
async def get_project_usage(
    project_id: str,
    api_key: ApiKey = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics for a project.

    Returns aggregated token usage and spending across all API keys
    in the project.
    """
    project = await _get_user_project(project_id, api_key.user_id, db)

    # Aggregate stats from all API keys in the project
    result = await db.execute(
        select(
            func.count(ApiKey.id).label("total_keys"),
            func.sum(func.cast(ApiKey.is_active, Integer)).label("active_keys"),  # type: ignore[arg-type]
            func.coalesce(func.sum(ApiKey.tokens_used_today), 0).label("total_tokens_today"),
            func.coalesce(func.sum(ApiKey.spent_this_month_usd), 0.0).label("total_spent_month"),
        ).where(ApiKey.project_id == project_id)
    )
    row = result.one()

    return ProjectUsageResponse(
        project_id=project.id,
        project_name=project.name,
        tier=project.tier,
        total_api_keys=row.total_keys or 0,
        active_api_keys=row.active_keys or 0,
        total_tokens_used_today=row.total_tokens_today or 0,
        total_spent_this_month_usd=round(row.total_spent_month or 0.0, 4),
        spending_limit_monthly_usd=None,  # Per-key, not per-project in aggregation
        usage_limits=_parse_json_field(project.usage_limits),
    )
