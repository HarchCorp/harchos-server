"""Auth Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel

class ApiKeyCreate(BaseModel):
    """Schema for creating an API key."""
    name: str

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

class UserInfo(BaseModel):
    """Current user info response."""
    id: str
    email: str
    name: str
    is_active: bool
    role: str = "user"
    created_at: datetime
