"""Project Pydantic schemas — request/response models for project-scoped API key management.

These schemas handle validation for all project-related operations including
project CRUD, project-scoped API key creation, and usage stats.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.validators import validate_name, validate_string_field


# ---------------------------------------------------------------------------
# Project schemas
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    """Schema for creating a new project."""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: str | None = Field(None, max_length=2000, description="Project description")
    tier: str = Field("free", description="Project tier: free, standard, enterprise")
    usage_limits: dict[str, Any] | None = Field(
        None,
        description="Per-project usage limits (e.g. {'max_rpm': 120, 'max_tokens_per_day': 500000})",
    )

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        return validate_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_string_field(v, field_name="description", max_len=2000)

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("free", "standard", "enterprise"):
            raise ValueError("Tier must be one of: free, standard, enterprise")
        return v


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: str | None = Field(None, min_length=1, max_length=255, description="Project name")
    description: str | None = Field(None, max_length=2000, description="Project description")
    tier: str | None = Field(None, description="Project tier: free, standard, enterprise")
    is_active: bool | None = Field(None, description="Whether the project is active")
    usage_limits: dict[str, Any] | None = Field(None, description="Per-project usage limits")

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_string_field(v, field_name="description", max_len=2000)

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip().lower()
        if v not in ("free", "standard", "enterprise"):
            raise ValueError("Tier must be one of: free, standard, enterprise")
        return v


class ProjectResponse(BaseModel):
    """Project response schema."""
    id: str
    name: str
    description: str | None
    user_id: str
    tier: str
    is_active: bool
    usage_limits: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = {'from_attributes': True}


# ---------------------------------------------------------------------------
# Project-scoped API key schemas
# ---------------------------------------------------------------------------

class ProjectApiKeyCreate(BaseModel):
    """Schema for creating an API key scoped to a project."""
    name: str = Field(..., min_length=1, max_length=64, description="Human-readable API key name")
    tier: str = Field("free", description="API key tier: free, standard, enterprise")
    scopes: list[str] | None = Field(
        None,
        description="Permission scopes (e.g. ['inference:read', 'workloads:write']). None = all scopes.",
    )
    allowed_models: list[str] | None = Field(
        None,
        description="Allowed model IDs. None = all models allowed.",
    )
    allowed_regions: list[str] | None = Field(
        None,
        description="Allowed regions. None = all regions allowed.",
    )
    max_tokens_per_day: int | None = Field(
        None,
        ge=0,
        description="Daily token budget. None = unlimited.",
    )
    spending_limit_monthly_usd: float | None = Field(
        None,
        ge=0.0,
        description="Monthly spending cap in USD. None = unlimited.",
    )
    expires_at: datetime | None = Field(
        None,
        description="Optional expiration time for the API key.",
    )

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("API key name must not be empty")
        if len(v) > 64:
            raise ValueError("API key name must be at most 64 characters")
        return v

    @field_validator("tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("free", "standard", "enterprise"):
            raise ValueError("Tier must be one of: free, standard, enterprise")
        return v

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        valid_prefixes = {"inference", "workloads", "carbon", "models", "billing", "energy", "hubs", "regions", "monitoring", "batch", "embeddings", "fine_tuning"}
        for scope in v:
            parts = scope.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid scope format '{scope}'. Expected 'resource:action' (e.g. 'inference:read')")
            resource, action = parts
            if resource not in valid_prefixes:
                raise ValueError(f"Unknown scope resource '{resource}'. Valid: {', '.join(sorted(valid_prefixes))}")
            if action not in ("read", "write"):
                raise ValueError(f"Invalid scope action '{action}'. Must be 'read' or 'write'")
        return v


class ProjectApiKeyResponse(BaseModel):
    """Response for a project-scoped API key (never includes full key after creation)."""
    id: str
    name: str
    key_prefix: str
    is_active: bool
    project_id: str | None
    tier: str
    scopes: list[str] | None
    allowed_models: list[str] | None
    allowed_regions: list[str] | None
    max_tokens_per_day: int | None
    tokens_used_today: int
    spending_limit_monthly_usd: float | None
    spent_this_month_usd: float
    created_at: datetime
    expires_at: datetime | None = None


class ProjectApiKeyCreateResponse(BaseModel):
    """Response when creating a project-scoped API key (includes full key once)."""
    id: str
    name: str
    key: str  # Only shown on creation
    key_prefix: str
    is_active: bool
    project_id: str | None
    tier: str
    scopes: list[str] | None
    allowed_models: list[str] | None
    allowed_regions: list[str] | None
    max_tokens_per_day: int | None
    spending_limit_monthly_usd: float | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Project usage stats
# ---------------------------------------------------------------------------

class ProjectUsageResponse(BaseModel):
    """Usage statistics for a project."""
    project_id: str
    project_name: str
    tier: str
    total_api_keys: int
    active_api_keys: int
    total_tokens_used_today: int
    total_spent_this_month_usd: float
    spending_limit_monthly_usd: float | None
    usage_limits: dict[str, Any] | None
