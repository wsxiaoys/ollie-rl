#!/bin/bash
set -euo pipefail
ulimit -n 65536
export PATH="$HOME/.local/bin:$PATH"
cd ~/ollie-rl
set -a; source .env; set +a
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:6432/prod?prepared_statement_cache_size=0"  # PGBOUNCER
export OLLIE_MAX_OUTPUT_TOKENS=8192   # operating point (was hardcoded in gemini_msrl.py)
export OLLIE_THINKING_LEVEL=MINIMAL   # operating point; set ""/none/off to disable
exec .venv/bin/python -m uvicorn ollie_rl.server.app:app --host 0.0.0.0 --port 8000
