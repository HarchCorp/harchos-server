"""Auth Pydantic schemas with comprehensive input validation.

Every field is validated to prevent injection attacks, enforce
business rules, and provide clear error messages.
"""

from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from app.schemas.validators import validate_api_key_name, validate_email_field


class ApiKeyCreate(BaseModel):
    """Schema for creating an API key."""
    name: str = Field(..., min_length=1, max_length=64, description="Human-readable API key name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_api_key_name(v)


class ApiKeyResponse(BaseModel):
    """API key response (never return the full key, only prefix)."""
    id: str
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    expires_at: datetime | None = None

    model_config = {'from_attributes': True}


class ApiKeyCreateResponse(BaseModel):
    """Response when creating an API key (includes full key once)."""
    id: str
    name: str
    key: str  # Only shown on creation
    key_prefix: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: str = Field(..., description="User email address")
    api_key: str = Field(..., min_length=20, description="API key (hsk_...)")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_email_field(v)

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("hsk_"):
            raise ValueError("API key must start with 'hsk_'")
        if len(v) < 20:
            raise ValueError("API key appears to be too short")
        return v


class RegisterRequest(BaseModel):
    """Schema for user registration (dev mode only)."""
    email: str = Field(..., description="User email address")
    name: str = Field(..., min_length=1, max_length=128, description="Full name")
    role: str = Field("user", description="User role: admin, user, viewer")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_email_field(v)

    @field_validator("name")
    @classmethod
    def validate_name_field(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name must not be empty")
        if len(v) > 128:
            raise ValueError("Name must be at most 128 characters")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("admin", "user", "viewer"):
            raise ValueError("Role must be one of: admin, user, viewer")
        return v


class UserInfo(BaseModel):
    """Current user info response."""
    id: str
    email: str
    name: str
    is_active: bool
    role: str = "user"
    created_at: datetime
