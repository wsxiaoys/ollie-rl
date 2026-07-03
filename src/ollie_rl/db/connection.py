import asyncio
import logging
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import BaseModel

logger = logging.getLogger(__name__)

# Default to a file-backed SQLite database for local development.
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///data/db.sqlite"

# Directory that holds the Alembic environment (env.py + versions/). Resolved
# relative to this file so it works regardless of the process's cwd (e.g. the
# Docker image, where WORKDIR is /app).
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_engine = None
_sessionmaker = None


def resolve_database_url() -> str:
    """Return the configured database URL.

    Reads ``DATABASE_URL`` from the environment, falling back to
    :data:`DEFAULT_DATABASE_URL`. Kept as a standalone function so the Alembic
    environment resolves the URL exactly the same way the application does.
    """
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def is_in_memory(url: str) -> bool:
    """Whether ``url`` points at an ephemeral in-memory SQLite database."""
    return ":memory:" in url


def get_engine():
    """Create and return an async SQLAlchemy engine singleton."""
    global _engine
    if _engine is None:
        url = resolve_database_url()
        if is_in_memory(url):
            logger.warning(
                "SQLite in-memory backend is being used. "
                "This should only be used for local development/testing and does not persist data across restarts."
            )
        _engine = create_async_engine(url, echo=False)
    return _engine


def get_sessionmaker():
    """Create and return an async session maker singleton."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _sessionmaker


def _build_alembic_config():
    """Build an Alembic ``Config`` pointed at the packaged migration env.

    The config is constructed programmatically (rather than loading
    ``alembic.ini``) so it does not depend on the process's working directory.
    ``env.py`` resolves ``sqlalchemy.url`` from the environment itself.
    """
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    return cfg


def _run_migrations_sync() -> None:
    """Upgrade the database to the latest revision (blocking).

    Alembic's async ``env.py`` drives its own event loop via ``asyncio.run``,
    so this must run in a thread without an active loop (see :func:`init_db`).
    """
    from alembic import command

    command.upgrade(_build_alembic_config(), "head")


async def init_db() -> None:
    """Bring the database schema up to date.

    - In-memory SQLite (tests / throwaway dev): create tables directly from the
      model metadata, since each connection is a fresh empty database and
      Alembic has nothing to persist against.
    - Any persistent backend (file SQLite, Postgres, ...): apply Alembic
      migrations up to ``head`` so schema changes are versioned and applied
      deterministically across restarts and upgrades.
    """
    url = resolve_database_url()
    if is_in_memory(url):
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(BaseModel.metadata.create_all)
        return

    logger.info("Applying database migrations (alembic upgrade head)...")
    # Alembic's env.py starts its own event loop; run it off the running loop.
    await asyncio.to_thread(_run_migrations_sync)


async def shutdown_db() -> None:
    """Close the database engine connections."""
    global _engine, _sessionmaker
    if _engine:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
