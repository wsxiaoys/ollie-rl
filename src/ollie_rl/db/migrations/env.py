"""Alembic migration environment for ollie-rl.

This module wires Alembic up to the application's async SQLAlchemy setup:

- The database URL is resolved from ``DATABASE_URL`` (falling back to the same
  default the app uses) rather than ``alembic.ini`` so migrations always target
  the same database as the running service.
- ``target_metadata`` points at ``BaseModel.metadata`` so ``--autogenerate``
  can diff the models against the live schema.
- Migrations run through an async engine, mirroring the application runtime.
- ``render_as_batch`` is enabled for SQLite so ``ALTER TABLE`` operations
  (which SQLite only partially supports) are emulated via table rebuilds.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from ollie_rl.db.connection import resolve_database_url
from ollie_rl.db.models import BaseModel

# Alembic Config object, providing access to values within alembic.ini.
config = context.config

# Resolve the database URL the same way the application does. Escape any '%'
# so ConfigParser interpolation (used by set_main_option) does not choke.
_db_url = resolve_database_url()
config.set_main_option("sqlalchemy.url", _db_url.replace("%", "%%"))

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata used as the autogenerate target.
target_metadata = BaseModel.metadata


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Batch mode is required for SQLite ALTER TABLE support; harmless
        # elsewhere but only enabled for SQLite to keep other backends' DDL
        # native.
        render_as_batch=_is_sqlite(_db_url),
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DBAPI connection)."""
    context.configure(
        url=_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_is_sqlite(_db_url),
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _db_url.replace("%", "%%")

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
