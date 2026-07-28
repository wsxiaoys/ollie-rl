#!/usr/bin/env python3
"""Lever A (driver side): call POST /runs/{run_id}/release on the abandon paths
so a dropped run frees its lease immediately instead of squatting 30 min.

Pairs with patch-release-endpoint.py (server). Adds a `release_run` helper
(fire-and-forget: retries transient transport errors like the other control
-plane calls, swallows 404/any status so releasing never kills the worker) and
calls it at the three worker() abandon sites:
  1. dispensed an unknown datum_id (no local task_data)
  2. rollout crashed (exception)
  3. NaN/None reward (timeout / sampling failure, no policy signal)

Idempotent. Run on VM: python3 ~/patch-driver-release.py [--revert]
"""

import pathlib
import sys

DRIVER = (
    pathlib.Path.home()
    / "vmsrl-cs/internal_src/example_recipe/codesearch/codesearch_ollie_train.py"
)

# --- 1. helper: inserted right after submit_reward ------------------------
HELPER_ANCHOR = """    resp.raise_for_status()
    return True


# --------------------------------------------------------------------------
# The rollout: one CodesearchAgent rollout == one ollie-rl Run
# --------------------------------------------------------------------------
"""
HELPER_NEW = '''    resp.raise_for_status()
    return True


async def release_run(
    client: httpx.AsyncClient, tuner_id: str, run_id: str
) -> None:
    """Release an abandoned run so its lease frees immediately (fire-and-forget).

    Called on the worker abandon paths (crash / NaN / unknown datum) instead of
    silently letting the 30-min lease expire: the server sets ``expires_at=now``
    so the datum is re-dispensable in seconds and stops counting as
    ``in_flight``. Best-effort -- transient transport errors are retried like
    the other control-plane calls, and ANY resulting status (incl. 404 for an
    already-gone run) is swallowed so releasing can never crash the worker.
    """
    try:
        resp = await _request_with_retry(
            f"release (run {run_id})",
            lambda: client.post(f"/tuners/{tuner_id}/runs/{run_id}/release"),
        )
        if resp.status_code >= 400:
            logger.debug(
                "run %s release returned %d (non-fatal): %s",
                run_id,
                resp.status_code,
                resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001 - releasing is best-effort
        logger.debug("run %s release failed (non-fatal): %s", run_id, exc)


# --------------------------------------------------------------------------
# The rollout: one CodesearchAgent rollout == one ollie-rl Run
# --------------------------------------------------------------------------
'''

# --- 2. call site: unknown datum_id ---------------------------------------
UNKNOWN_ANCHOR = """        if task_data is None:
            logger.warning(
                "run %04d dispensed unknown datum_id=%s; skipping", run, datum_id
            )
            continue
"""
UNKNOWN_NEW = """        if task_data is None:
            logger.warning(
                "run %04d dispensed unknown datum_id=%s; releasing", run, datum_id
            )
            await release_run(client, tuner_id, run_id)
            continue
"""

# --- 3. call site: rollout crashed ----------------------------------------
CRASH_ANCHOR = """            logger.warning(
                "run %04d rollout crashed (datum=%s); skipping reward: %s",
                run,
                datum_id,
                exc,
            )
            continue
"""
CRASH_NEW = """            logger.warning(
                "run %04d rollout crashed (datum=%s); releasing run: %s",
                run,
                datum_id,
                exc,
            )
            await release_run(client, tuner_id, run_id)
            continue
"""

# --- 4. call site: NaN/None reward ----------------------------------------
NAN_ANCHOR = """            logger.info(
                "run %04d datum=%s no policy signal (NaN/None); skipping reward",
                run,
                datum_id,
            )
            continue
"""
NAN_NEW = """            logger.info(
                "run %04d datum=%s no policy signal (NaN/None); releasing run",
                run,
                datum_id,
            )
            await release_run(client, tuner_id, run_id)
            continue
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
print("helper:        ", apply(DRIVER, HELPER_ANCHOR, HELPER_NEW, revert))
print("unknown-datum: ", apply(DRIVER, UNKNOWN_ANCHOR, UNKNOWN_NEW, revert))
print("rollout-crash: ", apply(DRIVER, CRASH_ANCHOR, CRASH_NEW, revert))
print("nan-reward:    ", apply(DRIVER, NAN_ANCHOR, NAN_NEW, revert))
