#!/usr/bin/env python3
"""B1: stop the dispenser from scanning the ENTIRE runs table on every dispense.

Root cause of the gen-4 wedge (2026-07-17): `_load_pool_and_runs` did
`SELECT * FROM runs WHERE tuner_id=?` -- loading ALL runs (40,423 rows, 38,311
of them dead unrewarded-past-lease) into Python on every `POST /runs`. A single
isolated dispense took >90s. The two pure consumers of that list --
`scheduler_scores`/`pick_datum` and `pick_eval_datum` -- BOTH count a run only
when it is rewarded (`reward IS NOT NULL`) or pending (`expires_at > now`);
their docstrings state "Expired-unrewarded attempts don't count". So loading
only those rows is behaviour-preserving and drops exactly the dead weight.

Filter: `reward IS NOT NULL OR expires_at > now`.
  - keeps ALL rewarded runs (trained or not -- scheduler_scores' fresh-tier
    tie-break reads trained_count on rewarded rows);
  - keeps ALL pending (lease-live) runs;
  - drops only unrewarded past-lease runs (the `lost`/`expired` dead rows),
    which both schedulers already ignore.

Idempotent. Run on VM: python3 ~/patch-dispense-scan.py [--revert]
"""

import pathlib
import sys

BASE = pathlib.Path.home() / "ollie-rl/src/ollie_rl/service/tuner/base.py"

# Add or_ to the existing sqlalchemy import so the WHERE clause can OR the two
# keep-conditions. (func, select already imported on this line.)
IMPORT_ANCHOR = "from sqlalchemy import func, select\n"
IMPORT_NEW = "from sqlalchemy import func, or_, select\n"

# Replace the unfiltered load with the filtered one. utcnow is already imported
# in this module (used by _expired_datums); reference it directly.
LOAD_ANCHOR = """        runs_result = await session.execute(
            select(RunModel).where(RunModel.tuner_id == tuner_id)
        )
        runs = list(runs_result.scalars().all())
        return datum_pool, runs
"""
LOAD_NEW = """        # SCALE (B1): load only runs the pure schedulers actually count --
        # rewarded (reward IS NOT NULL) or pending (lease not expired). Both
        # `scheduler_scores`/`pick_datum` and `pick_eval_datum` ignore
        # unrewarded past-lease runs ("Expired-unrewarded attempts don't
        # count"), so excluding them here is behaviour-preserving and keeps the
        # dispense hot path O(live runs) instead of O(all runs ever) -- the
        # 40k-row full-table scan that wedged dispense at >90s.
        runs_result = await session.execute(
            select(RunModel).where(
                RunModel.tuner_id == tuner_id,
                or_(
                    RunModel.reward.is_not(None),
                    RunModel.expires_at > utcnow(),
                ),
            )
        )
        runs = list(runs_result.scalars().all())
        return datum_pool, runs
"""

# utcnow import guard: base.py imports RUN_EXPIRE_GENERATION_BUDGET_MS from
# constants and uses utcnow() in _expired_datums, but confirm utcnow is imported
# at module scope; if not, add it. (Checked at apply time below.)
UTCNOW_IMPORT_ANCHOR = "from ollie_rl.db.types import utcnow\n"


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


def ensure_utcnow_import(path, revert):
    """utcnow must be importable at module scope for the WHERE clause.

    base.py has no module-scope utcnow (its _expired_datums takes `now` as a
    param), so add the import after the known-present RunModel import line.
    Idempotent: skips if the exact import line is already present.
    """
    src = path.read_text()
    rm_anchor = "from ollie_rl.db.models import RunModel\n"
    if revert:
        if UTCNOW_IMPORT_ANCHOR in src:
            path.write_text(src.replace(rm_anchor + UTCNOW_IMPORT_ANCHOR, rm_anchor, 1))
            return "REVERTED utcnow import"
        return "nothing to revert"
    if UTCNOW_IMPORT_ANCHOR in src:
        return "already imported"
    if rm_anchor in src:
        path.write_text(src.replace(rm_anchor, rm_anchor + UTCNOW_IMPORT_ANCHOR, 1))
        return "ADDED utcnow import"
    return "COULD NOT ADD utcnow import (anchor missing)"


revert = "--revert" in sys.argv
print("utcnow import:", ensure_utcnow_import(BASE, revert))
print("or_ import:   ", apply(BASE, IMPORT_ANCHOR, IMPORT_NEW, revert))
print("load filter:  ", apply(BASE, LOAD_ANCHOR, LOAD_NEW, revert))
