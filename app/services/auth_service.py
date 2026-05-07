"""Authentication service – API key validation, JWT token generation, project-scoped keys."""

import hashlib
import secrets
from datetime import datetime, timezone, timedelta

import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.auth import ApiKeyCreateResponse, TokenResponse, UserInfo


class AuthService:
    """Handles API key and JWT token operations."""

    @staticmethod
    def generate_api_key() -> str:
        """Generate a new API key with the hsk_ prefix."""
        raw = secrets.token_urlsafe(32)
        return f"{settings.api_key_prefix}{raw}"

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def validate_key_format(key: str) -> bool:
        """Validate that an API key has the correct hsk_ prefix and min length."""
        return key.startswith(settings.api_key_prefix) and len(key) >= 20

    @staticmethod
    async def authenticate_api_key(db: AsyncSession, key: str) -> ApiKey | None:
        """Validate an API key and return the ApiKey object if valid.

        Also checks that the project (if any) is active and not deleted.
        """
        if not AuthService.validate_key_format(key):
            return None
        key_hash = AuthService.hash_key(key)
        result = await db.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create_api_key(
        db: AsyncSession,
        user_id: str,
        name: str,
        *,
        project_id: str | None = None,
        tier: str = "standard",  # Default to 'standard' — 'free' tier is too restrictive for real use
        scopes: list[str] | None = None,
        allowed_models: list[str] | None = None,
        allowed_regions: list[str] | None = None,
        max_tokens_per_day: int | None = None,
        spending_limit_monthly_usd: float | None = None,
        expires_at: datetime | None = None,
    ) -> ApiKeyCreateResponse:
        """Create a new API key for a user, optionally scoped to a project.

        When project_id is provided, the key is scoped to that project and
        inherits the project's restrictions. The scopes, allowed_models,
        allowed_regions, and budget fields provide fine-grained control.
        """
        import json

        raw_key = AuthService.generate_api_key()
        key_hash = AuthService.hash_key(raw_key)
        key_prefix = raw_key[:8]  # hsk_ + 4 chars

        api_key = ApiKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            is_active=True,
            project_id=project_id,
            tier=tier,
            scopes=json.dumps(scopes) if scopes else None,
            allowed_models=json.dumps(allowed_models) if allowed_models else None,
            allowed_regions=json.dumps(allowed_regions) if allowed_regions else None,
            max_tokens_per_day=max_tokens_per_day,
            tokens_used_today=0,
            spending_limit_monthly_usd=spending_limit_monthly_usd,
            spent_this_month_usd=0.0,
            expires_at=expires_at,
        )
        db.add(api_key)
        await db.flush()

        return ApiKeyCreateResponse(
            id=api_key.id,
            name=name,
            key=raw_key,
            key_prefix=key_prefix,
            is_active=True,
            created_at=api_key.created_at,
        )

    @staticmethod
    def create_jwt_token(api_key_id: str, user_id: str) -> TokenResponse:
        """Create a JWT token for an authenticated user."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=settings.access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "api_key_id": api_key_id,
            "iat": now,
            "exp": expires,
        }
        token = jwt.encode(payload, settings.get_effective_secret_key(), algorithm="HS256")
        # Add token prefix
        token = f"{settings.token_prefix}{token}"

        return TokenResponse(
            access_token=token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60,
        )

    @staticmethod
    def verify_jwt_token(token: str) -> dict | None:
        """Verify and decode a JWT token. Returns payload or None."""
        if token.startswith(settings.token_prefix):
            token = token[len(settings.token_prefix):]
        try:
            payload = jwt.decode(token, settings.get_effective_secret_key(), algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            return None

    @staticmethod
    async def get_user_info(db: AsyncSession, user_id: str) -> UserInfo | None:
        """Get user info by ID."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return None
        return UserInfo(
            id=user.id,
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            created_at=user.created_at,
        )
