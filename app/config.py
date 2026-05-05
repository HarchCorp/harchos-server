"""Application configuration via pydantic-settings."""

import secrets
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings use the HARCHOS_ prefix to avoid collision with system
    environment variables like DATABASE_URL.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="HARCHOS_",  # Avoid collision with system DATABASE_URL
    )

    # Application
    app_name: str = "HarchOS Server"
    app_version: str = "0.3.0"
    debug: bool = False  # SAFE DEFAULT: debug off in production
    environment: str = "production"  # dev / staging / production
    log_level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR / CRITICAL

    # Database — PostgreSQL for production, SQLite for local dev
    # Supabase connection string format:
    #   postgresql+asyncpg://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres
    database_url: str = "sqlite+aiosqlite:///./harchos.db"
    db_pool_size: int = 20  # Connection pool size (PostgreSQL only)
    db_max_overflow: int = 10  # Extra connections beyond pool_size
    db_pool_recycle: int = 3600  # Recycle connections after 1h

    # Auth
    # IMPORTANT: In production, ALWAYS set HARCHOS_SECRET_KEY to a random 64+ char string
    # Generate with: python -c "import secrets; print(secrets.token_urlsafe(64))"
    secret_key: str = ""  # Empty = will fail to start in production (safe default)
    api_key_prefix: str = "hsk_"
    token_prefix: str = "hst_"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # CORS — restrictive by default
    cors_origins: list[str] = []  # Empty = no CORS in production by default
    # For dev: HARCHOS_CORS_ORIGINS='["http://localhost:3000","http://localhost:8000"]'

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100

    # Default test API key (hashed on seed) — ONLY for dev/testing
    default_api_key: str = "hsk_test_development_key_12345"

    # Carbon-aware scheduling
    # Get your free trial key at: https://app.electricitymaps.com/auth/sign-up
    # Academic access: free with .edu email
    electricity_maps_api_key: str = ""  # Required for real carbon data
    carbon_intensity_uk_api_key: str = ""  # Optional, UK Carbon Intensity API
    carbon_green_threshold_gco2_kwh: float = 200.0  # Below this = "green"
    carbon_cache_ttl_minutes: int = 30  # How long to cache API results
    carbon_static_fallback: bool = True  # Use static data when APIs unavailable

    # Rate limiting
    rate_limit_requests_per_minute: int = 60

    # Redis caching (Upstash) — optional but recommended
    # Get free at: https://console.upstash.com
    # REST API format (preferred for serverless/Railway)
    upstash_redis_url: str = ""  # e.g. https://us1-xxx.upstash.io
    upstash_redis_token: str = ""  # e.g. 2553feg6a2d9842h2a0gcdb5f8efe9934
    # Traditional Redis URL (alternative, not for Upstash)
    redis_url: str = ""  # e.g. redis://localhost:6379/0

    # Security headers
    enable_security_headers: bool = True

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "dev"

    def get_effective_secret_key(self) -> str:
        """Get the secret key, generating a dev key if in development mode."""
        if self.secret_key:
            return self.secret_key
        if self.is_development:
            # Auto-generate for dev only — will change on restart
            return "harchos-dev-secret-key-change-in-production"
        # Production without a secret key = fatal error
        raise ValueError(
            "HARCHOS_SECRET_KEY must be set in production! "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )


settings = Settings()
