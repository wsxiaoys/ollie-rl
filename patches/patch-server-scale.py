#!/usr/bin/env python3
"""Scale the ollie server for high concurrency (target driver --concurrency 1024):
  1. gemini_msrl/client.py: give the server->MSRL httpx.AsyncClient explicit
     high connection Limits (default is 100 max conns -> bottleneck).
  2. db/connection.py: raise the Postgres pool (was 20+20=40) to 350+120=470,
     staying under the DB's max_connections=500.
Idempotent. Run on VM: python3 ~/patch-server-scale.py [--revert]
"""

import sys, pathlib

CLIENT = pathlib.Path.home() / "ollie-rl/src/gemini_msrl/client.py"
CONN = pathlib.Path.home() / "ollie-rl/src/ollie_rl/db/connection.py"

CLIENT_ANCHOR = "            self._client = httpx.AsyncClient(headers=self.headers)\n"
CLIENT_NEW = (
    "            self._client = httpx.AsyncClient(\n"
    "                headers=self.headers,\n"
    "                limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512),\n"
    "                timeout=httpx.Timeout(600.0, connect=30.0, pool=60.0),  # SCALE\n"
    "            )\n"
)

CONN_ANCHOR = """                pool_size=20,  # steady connections held open
                max_overflow=20,  # burst headroom -> 40 max
"""
CONN_NEW = """                pool_size=350,  # SCALE: steady conns (under PG max_connections=500)
                max_overflow=120,  # SCALE: burst headroom -> 470 max
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
print("client.py:", apply(CLIENT, CLIENT_ANCHOR, CLIENT_NEW, revert))
print("connection.py:", apply(CONN, CONN_ANCHOR, CONN_NEW, revert))
