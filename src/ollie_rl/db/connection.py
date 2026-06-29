import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from .models import BaseModel

logger = logging.getLogger(__name__)

# Default to in-memory SQLite for local development
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///data/db.sqlite"

_engine = None
_sessionmaker = None


def get_engine():
    """Create and return an async SQLAlchemy engine singleton."""
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
        if url == DEFAULT_DATABASE_URL:
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


async def init_db() -> None:
    """Create database tables if they do not exist."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.create_all)


async def shutdown_db() -> None:
    """
    Close the database engine connections.
    """
    global _engine, _sessionmaker
    if _engine:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
