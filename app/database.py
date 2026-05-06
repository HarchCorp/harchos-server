"""SQLAlchemy async engine and session setup.

Supports both SQLite (local dev) and PostgreSQL (production/Supabase).
When HARCHOS_DATABASE_URL starts with "postgresql", connection pooling is
enabled with configurable pool size and recycling.

Performance improvements:
- No auto-commit on GET requests (avoids unnecessary commits)
- Pool pre-ping for connection health
- Proper pool recycling for long-running connections
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
    })
    # asyncpg uses ssl in connection string, not connect_args
    # Add ?ssl=require to connection string if not already present
    db_url = settings.database_url
    if "ssl=" not in db_url and "sslmode=" not in db_url:
        separator = "&" if "?" in db_url else "?"
        settings.database_url = f"{db_url}{separator}ssl=require"
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

    Automatically commits on success (for mutations) and rolls back on exception.
    Only commits if the session has pending changes (avoids unnecessary commits
    on GET requests that don't modify data).
    """
    async with async_session_factory() as session:
        try:
            yield session
            # Only commit if there are pending changes
            # This avoids unnecessary commits on read-only GET requests
            if session.in_transaction() and session.is_active:
                # Check if any changes were made
                if session.dirty or session.new or session.deleted:
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
