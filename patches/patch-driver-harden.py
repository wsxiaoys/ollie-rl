#!/usr/bin/env python3
"""Harden the codesearch-ollie driver so a control-plane 5xx can't kill it.

Root cause of the --concurrency 1024 rc=1 crash (2026-07-16): under load the
server returned a dispense 500; `dispense_run`/`submit_reward` call
`resp.raise_for_status()`, but `_request_with_retry` only catches
Timeout/Transport errors -- so the HTTPStatusError escaped the worker, unwound
`asyncio.gather` (which had no return_exceptions), and killed the whole driver.

Two defense-in-depth layers:
  1. `_request_with_retry`: treat server-side 5xx like a transient transport
     error -- back off (jittered, capped) and retry -- so dispense/reward blips
     under load recover instead of raising. (204/4xx still returned to caller,
     preserving the existing 204-barrier and 409-finalized handling.)
  2. `asyncio.gather(..., return_exceptions=True)`: a backstop so any single
     worker's unhandled exception ends only that worker, not the pool; the
     collected exceptions are logged.

Idempotent. Run on VM: python3 ~/patch-driver-harden.py [--revert]
"""

import pathlib
import sys

DRIVER = (
    pathlib.Path.home()
    / "vmsrl-cs/internal_src/example_recipe/codesearch/codesearch_ollie_train.py"
)

# --- Layer 1: retry-on-5xx in _request_with_retry ------------------------
RETRY_ANCHOR = """    attempt = 0
    while True:
        try:
            return await send()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            ceiling = min(
                CONTROL_PLANE_BACKOFF_CAP_SEC,
                CONTROL_PLANE_BACKOFF_BASE_SEC * (2**attempt),
            )
            delay = random.uniform(0, ceiling)  # full jitter
            logger.warning(
                "%s: transient transport error (%s); retrying in %.1fs",
                describe,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
"""
RETRY_NEW = """    attempt = 0
    while True:
        try:
            resp = await send()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            reason = type(exc).__name__
        else:
            # HARDEN: a server-side 5xx (pool exhaustion, brief stall, transient
            # dispense/reward failure under load) is retryable -- fall through to
            # backoff instead of returning it, so the caller's raise_for_status()
            # never fires on a transient 5xx and kills the driver. Non-5xx
            # (2xx/204/4xx) go straight back to the caller unchanged.
            if resp.status_code < 500:
                return resp
            reason = f"HTTP {resp.status_code}"
        ceiling = min(
            CONTROL_PLANE_BACKOFF_CAP_SEC,
            CONTROL_PLANE_BACKOFF_BASE_SEC * (2**attempt),
        )
        delay = random.uniform(0, ceiling)  # full jitter
        logger.warning(
            "%s: transient failure (%s); retrying in %.1fs",
            describe,
            reason,
            delay,
        )
        await asyncio.sleep(delay)
        attempt += 1
"""

# --- Layer 2: return_exceptions backstop on the worker gather -------------
GATHER_ANCHOR = """                for i in range(max(1, args.concurrency))
            )
        )

    return 0
"""
GATHER_NEW = """                for i in range(max(1, args.concurrency))
            ),
            return_exceptions=True,  # HARDEN: one worker's crash must not kill the pool
        )
        for _r in results:
            if isinstance(_r, BaseException):
                logger.error("worker ended with unhandled exception: %r", _r)

    return 0
"""

# The gather result must be captured for the logging loop above.
ASSIGN_ANCHOR = "        await asyncio.gather(\n"
ASSIGN_NEW = "        results = await asyncio.gather(\n"


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
# Order matters: on apply do retry+gather first, then the assign rename (whose
# anchor "await asyncio.gather(" still exists until renamed). On revert, undo the
# rename first so the gather block anchor matches again.
if revert:
    print("assign:", apply(DRIVER, ASSIGN_ANCHOR, ASSIGN_NEW, revert))
    print("gather:", apply(DRIVER, GATHER_ANCHOR, GATHER_NEW, revert))
    print("retry: ", apply(DRIVER, RETRY_ANCHOR, RETRY_NEW, revert))
else:
    print("retry: ", apply(DRIVER, RETRY_ANCHOR, RETRY_NEW, revert))
    print("gather:", apply(DRIVER, GATHER_ANCHOR, GATHER_NEW, revert))
    print("assign:", apply(DRIVER, ASSIGN_ANCHOR, ASSIGN_NEW, revert))
