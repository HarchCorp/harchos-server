"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="HARCHOS_",  # Avoid collision with system DATABASE_URL
    )

    # Application
    app_name: str = "HarchOS Server"
    app_version: str = "0.2.0"
    debug: bool = True
    environment: str = "dev"  # dev / staging / production
    log_level: str = "INFO"  # DEBUG / INFO / WARNING / ERROR / CRITICAL

    # Database
    database_url: str = "sqlite+aiosqlite:///./harchos.db"
    # For production PostgreSQL:
    # database_url: str = "postgresql+asyncpg://harchos:harchos@localhost:5432/harchos"

    # Auth
    secret_key: str = "harchos-dev-secret-key-change-in-production"
    api_key_prefix: str = "hsk_"
    token_prefix: str = "hst_"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # CORS
    cors_origins: list[str] = ["*"]

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100

    # Default test API key (hashed on seed)
    default_api_key: str = "hsk_test_development_key_12345"

    # Carbon-aware scheduling
    electricity_maps_api_key: str = ""  # Free at https://api.electricitymap.org/
    carbon_intensity_uk_api_key: str = ""  # Optional, UK Carbon Intensity API
    carbon_green_threshold_gco2_kwh: float = 200.0  # Below this = "green"
    carbon_cache_ttl_minutes: int = 30  # How long to cache API results
    carbon_static_fallback: bool = True  # Use static data when APIs unavailable

    # Rate limiting
    rate_limit_requests_per_minute: int = 60

    # Redis (optional, for caching and rate limiting)
    redis_url: str = ""  # e.g. redis://localhost:6379/0

settings = Settings()
