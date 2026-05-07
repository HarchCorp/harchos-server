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
    app_version: str = "0.7.0"
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
    api_key_prefix: str = "hsk_"  # 4+ char prefix to prevent enumeration
    token_prefix: str = "hst_"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # Trusted proxy IPs — X-Forwarded-For is only accepted from these IPs
    # Set HARCHOS_TRUSTED_PROXIES='["10.0.0.1","10.0.0.2"]' for your load balancer
    trusted_proxies: list[str] = []  # Empty = never trust X-Forwarded-For

    # CORS — restrictive by default
    cors_origins: list[str] = []  # Empty = no CORS in production by default
    # For dev: HARCHOS_CORS_ORIGINS='["http://localhost:3000","http://localhost:8000"]'

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100

    # Default test API key (hashed on seed) — ONLY for dev/testing
    # Set HARCHOS_DEFAULT_API_KEY in Railway/env to use a fixed test key
    # If empty, a random key is generated at seed time (logged once)
    default_api_key: str = ""

    # Admin bootstrap — one-time admin creation in production
    # Set this to a strong random token, then use POST /v1/auth/bootstrap
    admin_bootstrap_token: str = ""  # Empty = bootstrap disabled

    # Carbon-aware scheduling
    # Get your free trial key at: https://app.electricitymaps.com/auth/sign-up
    # Academic access: free with .edu email
    electricity_maps_api_key: str = ""  # Required for real carbon data
    carbon_intensity_uk_api_key: str = ""  # Optional, UK Carbon Intensity API
    carbon_green_threshold_gco2_kwh: float = 200.0  # Below this = "green"
    carbon_cache_ttl_minutes: int = 30  # How long to cache API results
    carbon_static_fallback: bool = True  # Use static data when APIs unavailable

    # Rate limiting
    rate_limit_requests_per_minute: int = 60  # Default, overridden by tiered limits
    rate_limit_free_rpm: int = 30
    rate_limit_standard_rpm: int = 120
    rate_limit_enterprise_rpm: int = 600
    rate_limit_inference_free_rpm: int = 10
    rate_limit_inference_standard_rpm: int = 60
    rate_limit_inference_enterprise_rpm: int = 300
    rate_limit_batch_free_rpm: int = 5
    rate_limit_batch_standard_rpm: int = 20
    rate_limit_batch_enterprise_rpm: int = 100

    # Redis caching (Upstash) — optional but recommended
    # Get free at: https://console.upstash.com
    # REST API format (preferred for serverless/Railway)
    upstash_redis_url: str = ""  # e.g. https://us1-xxx.upstash.io
    upstash_redis_token: str = ""  # e.g. 2553feg6a2d9842h2a0gcdb5f8efe9934
    # Traditional Redis URL (alternative, not for Upstash)
    redis_url: str = ""  # e.g. redis://localhost:6379/0

    # Security headers
    enable_security_headers: bool = True

    # Inference backend — proxy to real LLM backends
    # Supported: vLLM, Together AI, Ollama, any OpenAI-compatible API
    inference_backend_url: str = ""  # e.g. http://vllm:8000/v1 or https://api.together.xyz/v1
    inference_backend_api_key: str = ""  # API key for the backend
    inference_backend_timeout_seconds: int = 30

    # Webhook configuration
    webhook_max_retries: int = 3
    webhook_timeout_seconds: int = 10

    # Performance
    max_request_size_bytes: int = 10 * 1024 * 1024  # 10MB
    enable_response_compression: bool = True
    enable_response_caching: bool = True

    # Fine-tuning
    fine_tuning_max_training_hours: int = 72  # Max fine-tuning job duration
    fine_tuning_max_file_size_bytes: int = 500 * 1024 * 1024  # 500MB

    # Batch inference
    batch_max_items: int = 100  # Max items per batch
    batch_concurrency: int = 10  # Max concurrent items processed
    batch_retention_days: int = 7  # How long to keep batch results

    # WebSocket
    ws_heartbeat_interval_seconds: int = 30
    ws_max_connections: int = 1000
    ws_monitoring_interval_seconds: int = 5
    ws_carbon_interval_seconds: int = 10

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "dev"

    def get_effective_secret_key(self) -> str:
        """Get the secret key, generating a secure dev key if in development mode."""
        if self.secret_key:
            return self.secret_key
        if self.is_development:
            # Auto-generate a secure random key for dev — changes on each restart
            # This is intentional: dev sessions don't survive restarts
            if not hasattr(self, '_dev_secret_key'):
                self._dev_secret_key = secrets.token_urlsafe(64)
            return self._dev_secret_key
        # Production without a secret key = fatal error
        raise ValueError(
            "HARCHOS_SECRET_KEY must be set in production! "
            "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )


settings = Settings()
