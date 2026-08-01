"""Run dispensing: pure datum selection plus database-backed leasing.

The scheduler helpers are plain functions of their arguments so selection and
progress previews share one testable implementation. Eval remains the
highest-priority tier; otherwise every training datum is eligible for the
most-full-first scheduler.
"""

from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple

from ollie_rl.cookbook import Recipe
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.constants import RUN_LEASE_SECONDS
from ollie_rl.service.tuner.types import SchedulerScores
from ollie_rl.types import DispenseRun, TunerStatus


def scheduler_scores(
    datum_pool: List[str],
    runs: List[RunModel],
) -> SchedulerScores:
    """Scheduler-view consumable tallies per datum (no staleness filter).

    Shared by ``pick_datum`` and the progress builder's ``next_pick`` labeling
    so the two never drift. Rewarded and live pending runs both count toward a
    group; trained, rejected, and expired runs do not.
    """
    now = utcnow()
    score = {d: 0 for d in datum_pool}
    trained = {d: 0 for d in datum_pool}
    for r in runs:
        if r.datum_id not in score:
            continue
        if r.trained_count > 0:
            # Track prior training exposure for the fresh-tier tie-break.
            trained[r.datum_id] += r.trained_count
            continue
        if r.rejected_count > 0:
            continue
        has_reward = r.reward is not None
        is_pending = r.reward is None and r.expires_at > now
        if has_reward or is_pending:
            score[r.datum_id] += 1
    return SchedulerScores(score=score, trained=trained)


def pick_tier(
    datum: str, score: Dict[str, int], recipe: Recipe
) -> Tuple[Literal["incomplete", "fresh", "saturated", "none"], str]:
    """Label the scheduler tier (+ human reason) for a candidate datum.

    Mirrors the tiers in ``pick_datum.priority`` so a dispense preview can
    explain *why* a datum would be chosen next.
    """
    count = score.get(datum, 0)
    group_size = recipe.group_size
    if 0 < count < group_size:
        return "incomplete", f"closest-to-complete group ({count}/{group_size})"
    if count == 0:
        return "fresh", "starting a new group from a fresh (least-trained) datum"
    if recipe.max_off_policy_generation > 0:
        return (
            "saturated",
            f"group full ({count}/{group_size}); dispensing off-policy surplus",
        )
    return "none", "all groups saturated; on-policy surplus would be requeued"


def pick_datum(
    datum_pool: List[str],
    runs: List[RunModel],
    recipe: Recipe,
) -> Optional[str]:
    """Pick the next datum to dispense a run for.

    Pure scheduling helper (no service/DB state) so it can be reasoned about
    and unit-tested in isolation.

    Uses a greedy "most-full-first" strategy via tiered priority. Only runs
    that are still *consumable* by a future train step are counted, i.e.
    not yet trained (``trained_count <= 0``), not requeued
    (``rejected_count <= 0``), and either rewarded or pending (not expired).
    This mirrors ``TunerService._collect_consumable_batch`` so a datum whose
    group was already trained resets to "fresh" for the next generation.

    Tiers, highest priority first:

    1. Started groups (0 < count < group_size), closest to complete first.
    2. Fresh datums (count == 0), least prior training exposure first.
    3. Saturated datums (count >= group_size), only when off-policy surplus is
       enabled, least saturated first.
    """
    if not datum_pool:
        return None

    group_size = recipe.group_size
    allow_surplus = recipe.max_off_policy_generation > 0
    scores = scheduler_scores(datum_pool, runs)

    def priority(datum: str) -> Tuple[int, int]:
        count = scores.score[datum]
        if 0 < count < group_size:
            return (2, count)
        if count == 0:
            return (1, -scores.trained[datum])
        if allow_surplus:
            return (0, -count)
        return (-1, 0)

    best = max(datum_pool, key=priority)
    if priority(best)[0] < 0:
        return None
    return best


def pick_eval_datum(
    eval_pool: List[str],
    runs: List[RunModel],
    checkpoint_id: str,
    group_size: int,
) -> Optional[str]:
    """First eval datum with fewer than ``group_size`` live attempts against
    ``checkpoint_id``.

    An attempt "counts" for a datum/checkpoint when it is a ``RunModel`` whose
    ``checkpoint_id`` equals the target and is either rewarded or still pending
    (reward ``None``, lease not expired). Expired-unrewarded attempts don't
    count, so a dropped eval rollout is re-dispensed. Among under-filled datums
    the least-covered wins (spread coverage). Returns ``None`` when every eval
    datum already has ``group_size`` live attempts for this checkpoint, when the
    pool is empty, or when ``group_size <= 0``.

    Pure helper (no service/DB state), mirroring :func:`pick_datum`.
    """
    if not eval_pool or group_size <= 0:
        return None

    now = utcnow()
    covered = {d: 0 for d in eval_pool}
    for r in runs:
        if r.checkpoint_id != checkpoint_id:
            continue
        if r.datum_id not in covered:
            continue
        has_reward = r.reward is not None
        is_pending = r.reward is None and r.expires_at > now
        if has_reward or is_pending:
            covered[r.datum_id] += 1

    # Least-covered under-filled datum wins; None when all are full.
    best = min(eval_pool, key=lambda d: covered[d])
    if covered[best] >= group_size:
        return None
    return best


class DispenseMixin(TunerServiceBase):
    """Serialized read-pick-insert dispensing of runs for a tuner."""

    async def dispense_run(self, tuner_id: str) -> Optional[DispenseRun]:
        """Dispense one run assignment for a tuner."""
        # Do not lease work to an external sandbox until the backend can serve
        # it. A pending/cancelled tuner follows the existing no-work path.
        trainer = await self._get_trainer(tuner_id)
        if await trainer.get_status() is not TunerStatus.IN_PROGRESS:
            return None

        recipe = await self._recipe_for(tuner_id)

        # Serialize the complete read-pick-insert sequence so concurrent local
        # dispensers cannot make decisions from the same scheduler snapshot.
        async with self._dispense_lock:
            async with self.async_session() as session:
                async with session.begin():
                    datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)
                    datum_id: Optional[str] = None
                    checkpoint_id: Optional[str] = None

                    # Eval remains the highest-priority tier.
                    if recipe.eval_group_size > 0:
                        latest = await self._latest_checkpoint(tuner_id, session)
                        if latest is not None:
                            eval_pool = await self._load_datums(
                                tuner_id, session, kind="eval"
                            )
                            datum_id = pick_eval_datum(
                                eval_pool,
                                runs,
                                latest.id,
                                recipe.eval_group_size,
                            )
                            if datum_id is not None:
                                checkpoint_id = latest.id

                    if datum_id is None:
                        datum_id = pick_datum(datum_pool, runs, recipe)

                    if datum_id is None:
                        return None

                    run_record = RunModel(
                        tuner_id=tuner_id,
                        datum_id=datum_id,
                        checkpoint_id=checkpoint_id,
                        reward=None,
                        trained_count=0,
                        rejected_count=0,
                        expires_at=utcnow() + timedelta(seconds=RUN_LEASE_SECONDS),
                    )
                    session.add(run_record)
                    await session.flush()

        return DispenseRun(
            run_id=run_record.id,
            datum_id=run_record.datum_id,
            expires_at=run_record.expires_at,
        )
