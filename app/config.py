"""Application configuration via pydantic-settings."""

from typing import Optional
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
    app_version: str = "0.1.0"
    debug: bool = True

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


settings = Settings()
