#!/usr/bin/env python3
"""Lever A (server side): add an explicit 'release run' endpoint so an abandoned
run frees its lease IMMEDIATELY instead of squatting for the full 30-min
RUN_LEASE_SECONDS.

Today when a rollout can't produce a real reward (executor infra failure, NaN
timeout, unknown datum) the driver just drops it: the run sits with reward=NULL
and a 30-min lease, counting as `in_flight` (blocking group formation) and
lingering as a dead row. This adds:

  * service method `TunerService.release_run(tuner_id, run_id)` -- sets
    `expires_at = now` on an unrewarded run so it is instantly past-lease:
    excluded from `in_flight`, immediately re-dispensable, classified `lost`
    (no training signal injected). Idempotent + tolerant:
      - run not found                     -> RunNotFoundError (404)
      - run already rewarded              -> no-op success (nothing to release)
      - run already expired               -> no-op success
      - else                              -> set expires_at=now
  * route `POST /tuners/{tuner_id}/runs/{run_id}/release` -> 200 on success,
    404 when the run doesn't exist. Never 5xx on the normal paths so the driver
    can fire-and-forget it.

Edits: dispensing.py (service method) + app.py (route). connection to DB reuses
the mixin's async_session, mirroring update_reward.

Idempotent. Run on VM: python3 ~/patch-release-endpoint.py [--revert]
"""

import pathlib
import sys

DISP = pathlib.Path.home() / "ollie-rl/src/ollie_rl/service/tuner/dispensing.py"
APP = pathlib.Path.home() / "ollie-rl/src/ollie_rl/server/app.py"

# --- 1. service method: appended inside DispenseMixin -----------------------
# Anchor on the end of dispense_run's return (the first method in the mixin).
# We insert the new method right after the `dispense_run` return block, before
# `_maybe_dispense_eval`.
DISP_ANCHOR = """    async def _maybe_dispense_eval(
        self,
        tuner_id: str,
        session,"""
DISP_NEW = '''    async def release_run(self, tuner_id: str, run_id: str) -> str:
        """Release an unrewarded run early by expiring its lease NOW.

        Lever A: when the driver abandons a run (infra failure / NaN timeout /
        unknown datum) it calls this instead of silently dropping it, so the
        lease frees in seconds rather than squatting for RUN_LEASE_SECONDS.
        Setting ``expires_at = now`` makes the run immediately past-lease:
        excluded from ``in_flight``, re-dispensable, and classified ``lost``
        (carries no reward, so no training signal is injected). Idempotent:
        a missing run raises RunNotFoundError; an already-rewarded or
        already-expired run is a no-op success. Returns a short status string.
        """
        from ollie_rl.db.models import RunModel
        from ollie_rl.db.types import utcnow
        from ollie_rl.service.tuner.errors import RunNotFoundError
        from sqlalchemy import select

        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(RunModel).where(
                        RunModel.id == run_id,
                        RunModel.tuner_id == tuner_id,
                    )
                )
                record = result.scalar_one_or_none()
                if record is None:
                    raise RunNotFoundError(
                        f"Run '{run_id}' not found under tuner '{tuner_id}'"
                    )
                now = utcnow()
                if record.reward is not None:
                    return "already_rewarded"
                if record.expires_at <= now:
                    return "already_expired"
                record.expires_at = now
                record.updated_at = now
        return "released"

    async def _maybe_dispense_eval(
        self,
        tuner_id: str,
        session,'''

# --- 2. app route: inserted right after the put_reward endpoint ------------
APP_ANCHOR = """        return PutRewardResponse(run_id=run_id, reward=request.reward)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RunExpiredError, RewardAlreadySetError, EmptyRunError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to record reward for run '{run_id}'")
        raise HTTPException(status_code=500, detail=str(e))
"""
APP_NEW = '''        return PutRewardResponse(run_id=run_id, reward=request.reward)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RunExpiredError, RewardAlreadySetError, EmptyRunError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to record reward for run '{run_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tuners/{tuner_id}/runs/{run_id}/release")
async def release_run(tuner_id: str, run_id: str) -> dict:
    """Release an unrewarded run early (expire its lease now).

    The driver calls this when a rollout can't yield a real reward so the
    datum's lease frees immediately instead of squatting for the full lease
    window. Idempotent: already-rewarded / already-expired runs return a no-op
    success; a missing run is 404. Deliberately never 5xx on normal paths so
    the driver can fire-and-forget.
    """
    from ollie_rl.service.tuner import RunNotFoundError

    try:
        status = await services.tuner.release_run(tuner_id=tuner_id, run_id=run_id)
        return {"run_id": run_id, "status": status}
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to release run '{run_id}'")
        raise HTTPException(status_code=500, detail=str(e))
'''


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
print("dispensing.py service method:", apply(DISP, DISP_ANCHOR, DISP_NEW, revert))
print("app.py route:               ", apply(APP, APP_ANCHOR, APP_NEW, revert))
