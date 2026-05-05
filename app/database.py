"""SQLAlchemy async engine and session setup.

Supports both SQLite (local dev) and PostgreSQL (production/Supabase).
When HARCHOS_DATABASE_URL starts with "postgresql", connection pooling is
enabled with configurable pool size and recycling.
"""

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger("harchos.database")

# Determine if we're using PostgreSQL
_is_postgres = settings.database_url.startswith("postgresql")

# Engine kwargs differ between SQLite and PostgreSQL
_engine_kwargs: dict = {
    "echo": settings.debug,
    "future": True,
}

if _is_postgres:
    # PostgreSQL with connection pooling (Supabase / any managed PG)
    _engine_kwargs.update({
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_recycle": settings.db_pool_recycle,
        "pool_pre_ping": True,  # Verify connections before use
        "connect_args": {
            "sslmode": "require",  # Required for Supabase
            "statement_timeout": "30000",  # 30s query timeout
        },
    })
    logger.info(
        "Database: PostgreSQL (pool_size=%d, max_overflow=%d)",
        settings.db_pool_size, settings.db_max_overflow,
    )
else:
    # SQLite for local development
    _engine_kwargs.update({
        "connect_args": {"check_same_thread": False},
    })
    logger.info("Database: SQLite (local development mode)")

engine = create_async_engine(settings.database_url, **_engine_kwargs)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """Dependency that yields an async database session.

    Automatically commits on success and rolls back on exception.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables.

    For SQLite (dev): uses create_all directly.
    For PostgreSQL (prod): should use Alembic migrations, but this
    provides a safety net for first deployment.
    """
    async with engine.begin() as conn:
        if _is_postgres:
            # In production, Alembic should handle migrations.
            # create_all is safe for first deploy (creates tables if not exist).
            logger.info("Running create_all on PostgreSQL (use Alembic for migrations)")
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the database engine (call on app shutdown)."""
    await engine.dispose()
    logger.info("Database engine disposed")
