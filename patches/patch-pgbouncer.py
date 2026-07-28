#!/usr/bin/env python3
"""Point the ollie server at pgbouncer (transaction pooling) instead of PG direct.

Companion to the pgbouncer container (127.0.0.1:6432 -> PG 127.0.0.1:5432,
pool_mode=transaction). This decouples the server's app-side concurrency from
Postgres' own max_connections=500 cap, which was the hard ceiling that killed
the driver at --concurrency 1024 (QueuePool exhaustion at 470 conns all
contending for <=500 real PG backends).

Two edits:
  1. start-ollie.sh: DATABASE_URL host port 5432 -> 6432 (pgbouncer), and append
     `?prepared_statement_cache_size=0`. With the cache disabled asyncpg uses
     UNNAMED prepared statements (prepared+executed within one transaction),
     which are safe across pgbouncer transaction-pool boundaries -- avoiding the
     `prepared statement "__asyncpg_..." does not exist` error that named/cached
     statements hit under transaction pooling. The param flows through
     resolve_database_url() to BOTH the app engine and the Alembic migration
     engine (both call it), so startup migrations also go through pgbouncer
     safely. (SQLAlchemy asyncpg dialect parses this as a DBAPI arg -- see
     docs.sqlalchemy.org PostgreSQL/asyncpg "Prepared Statement Cache".)
  2. connection.py: raise the SQLAlchemy pool 350/120 -> 600/300. These are now
     client connections TO pgbouncer (cap max_client_conn=2000), NOT to PG, so
     the old "stay under PG 500" constraint no longer applies. pgbouncer maps
     only ~200 of them (default_pool_size) onto real PG backends at a time.

Idempotent. Run on VM: python3 ~/patch-pgbouncer.py [--revert]
"""

import pathlib
import sys

START = pathlib.Path.home() / "start-ollie.sh"
CONN = pathlib.Path.home() / "ollie-rl/src/ollie_rl/db/connection.py"

START_ANCHOR = (
    'export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/prod"\n'
)
START_NEW = 'export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:6432/prod?prepared_statement_cache_size=0"  # PGBOUNCER\n'

# Anchor on the SCALE lines left by patch-server-scale.py (confirmed present).
CONN_ANCHOR = """                pool_size=350,  # SCALE: steady conns (under PG max_connections=500)
                max_overflow=120,  # SCALE: burst headroom -> 470 max
"""
CONN_NEW = """                pool_size=600,  # PGBOUNCER: client conns to pgbouncer (cap 2000), not PG
                max_overflow=300,  # PGBOUNCER: burst headroom -> 900 max client conns
"""


def apply(path, anchor, new, revert):
    src = path.read_text()
    if revert:
        if new in src:
            path.write_text(src.replace(new, anchor))
            return "REVERTED"
        return "nothing to revert"
    if new in src:
        return "already patched"
    if anchor not in src:
        return "ANCHOR NOT FOUND"
    path.write_text(src.replace(anchor, new, 1))
    return "PATCHED"


revert = "--revert" in sys.argv
print("start-ollie.sh:", apply(START, START_ANCHOR, START_NEW, revert))
print("connection.py: ", apply(CONN, CONN_ANCHOR, CONN_NEW, revert))
