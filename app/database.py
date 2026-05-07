"""SQLAlchemy async engine and session setup.

Supports both SQLite (local dev / Railway) and PostgreSQL (production/Supabase).
When HARCHOS_DATABASE_URL starts with "postgresql", connection pooling is
enabled with configurable pool size and recycling.

Performance improvements:
- Pool pre-ping for connection health
- Proper pool recycling for long-running connections
- SQLite: WAL mode for concurrent reads, busy_timeout to avoid lock failures,
  strict pool limits to prevent connection exhaustion
- Read-only sessions skip commit to reduce SQLite write-lock contention
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
    # SQLite — properly configured for async use with aiosqlite.
    #
    # CRITICAL: Unlike sync SQLite (which auto-uses StaticPool), async SQLite
    # defaults to AsyncAdaptedQueuePool with pool_size=5 + max_overflow=10,
    # allowing up to 15 concurrent connections. Each aiosqlite connection runs
    # in its own thread, creating lock contention since SQLite only supports
    # one writer at a time.
    #
    # HOWEVER: In-memory SQLite ("sqlite+aiosqlite://") forces StaticPool
    # which does NOT accept pool_size / max_overflow. Detect this case.
    #
    # Fixes:
    # - pool_size=5, max_overflow=0: Cap at 5 connections (no unbounded overflow)
    # - busy_timeout=30: Wait up to 30s for locks instead of failing immediately
    # - pool_pre_ping: Detect stale connections before use
    # - WAL mode enabled in init_db() for concurrent reads during writes
    _is_in_memory = "://" in settings.database_url and settings.database_url.split("://", 1)[1].rstrip("/") == ""

    _engine_kwargs.update({
        "connect_args": {
            "check_same_thread": False,
            "timeout": 30,  # SQLite busy_timeout (seconds) — wait for locks
        },
        "pool_pre_ping": True,  # Verify connections before use
    })

    if _is_in_memory:
        # In-memory SQLite auto-selects StaticPool; pool_size/max_overflow
        # are invalid for StaticPool and will raise TypeError if passed.
        logger.info("Database: SQLite in-memory (StaticPool, busy_timeout=30s)")
    else:
        _engine_kwargs.update({
            "pool_size": 5,
            "max_overflow": 0,  # Strict limit — no overflow connections
        })
        logger.info("Database: SQLite (pool_size=5, max_overflow=0, busy_timeout=30s, WAL mode)")

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

    Commits on success if the session has pending changes (mutations).
    Read-only sessions (no dirty/new/deleted objects) skip the commit to
    avoid unnecessary write-lock acquisition — this is critical for SQLite
    where even a no-op COMMIT briefly contends for the database write lock.
    Rolls back on exception.
    """
    async with async_session_factory() as session:
        try:
            yield session
            # Only commit if there are actual pending changes.
            # Read-only requests (GET) skip the commit entirely, avoiding
            # SQLite write-lock contention. PostgreSQL handles no-op commits
            # efficiently too, so this is safe for both backends.
            if session.is_active and (session.dirty or session.new or session.deleted):
                await session.commit()
        except Exception:
            if session.is_active:
                await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables.

    For SQLite (dev/Railway): enables WAL mode for concurrent reads during
    writes, sets busy_timeout to wait for locks, and uses synchronous=NORMAL
    for better write performance (still safe with WAL).
    For PostgreSQL (prod): should use Alembic migrations, but this
    provides a safety net for first deployment.
    """
    async with engine.begin() as conn:
        if not _is_postgres:
            from sqlalchemy import text
            # Enable WAL mode: allows concurrent readers while one writer
            # is active. Without WAL, SQLite uses DELETE journal mode which
            # blocks ALL access (reads included) during writes.
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            # Wait up to 30 seconds for locks instead of failing immediately.
            # This is the primary fix for "database is locked" errors under
            # concurrent access with aiosqlite.
            await conn.execute(text("PRAGMA busy_timeout=30000"))
            # NORMAL is safe with WAL and much faster than FULL.
            # FULL is only needed for NFS filesystems (not the case on Railway).
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            logger.info("SQLite pragmas applied: WAL, busy_timeout=30s, synchronous=NORMAL")

        if _is_postgres:
            # In production, Alembic should handle migrations.
            # create_all is safe for first deploy (creates tables if not exist).
            logger.info("Running create_all on PostgreSQL (use Alembic for migrations)")

        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Dispose of the database engine (call on app shutdown)."""
    await engine.dispose()
    logger.info("Database engine disposed")
