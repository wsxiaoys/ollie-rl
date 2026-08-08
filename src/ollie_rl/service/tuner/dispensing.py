"""Run dispensing: pure datum selection plus database-backed leasing.

The selection helpers are plain functions of already-computed per-datum
tallies, so selection and progress previews share one testable implementation
and neither has to load the run table. Eval remains the highest-priority tier;
training datums are prioritized by prior training exposure and then by their
persisted corpus order.
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


def outstanding_run_budget(recipe: Recipe) -> int:
    """How many runs may be outstanding across the whole training pool.

    The current generation plus ``max_off_policy_generation`` older ones may
    each contribute one batch, so anything beyond that horizon is work that
    would be rejected as stale before it could be trained on.
    """
    return (
        (recipe.max_off_policy_generation + 1)
        * recipe.group_size
        * recipe.num_groups_per_batch
    )


def pick_tier(
    datum: str,
    score: Dict[str, int],
    recipe: Recipe,
) -> Tuple[Literal["incomplete", "fresh", "saturated", "budget", "none"], str]:
    """Label the scheduler tier (+ human reason) for a candidate datum.

    Mirrors ``pick_datum`` so a dispense preview can explain *why* a datum
    would be chosen next -- including the tuner-wide horizon budget, which
    outranks every per-datum tier. Leaving that out made the preview attribute
    a budget stall to saturation, which is a different problem with a different
    fix (one clears when the trainer catches up, the other needs more datums).
    """
    outstanding = sum(score.values())
    budget = outstanding_run_budget(recipe)
    if outstanding >= budget:
        return (
            "budget",
            f"outstanding runs fill the policy-valid horizon "
            f"({outstanding}/{budget}); waiting for a train step",
        )
    count = score.get(datum, 0)
    group_size = recipe.group_size
    if 0 < count < group_size:
        reason = f"continuing the next group in training order ({count}/{group_size})"
        return "incomplete", reason
    if count == 0:
        return "fresh", "starting the next group in training order"
    if recipe.max_off_policy_generation > 0:
        return (
            "saturated",
            f"group full ({count}/{group_size}); dispensing off-policy surplus",
        )
    return "none", "all groups saturated; on-policy surplus would be requeued"


def pick_datum(
    datum_pool: List[str],
    scores: SchedulerScores,
    recipe: Recipe,
) -> Optional[str]:
    """Pick the next datum to dispense a run for.

    Pure scheduling helper (no service/DB state) so it can be reasoned about
    and unit-tested in isolation. ``scores`` comes from
    ``TunerServiceBase._scheduler_scores``, which counts only runs still
    *consumable* by a future train step -- not yet trained
    (``trained_count <= 0``), not requeued (``rejected_count <= 0``), and
    either rewarded or pending (not expired). That mirrors
    ``TunerService._collect_consumable_batch`` so a datum whose group was
    already trained resets to "fresh" for the next generation.

    Unsaturated datums are selected in training order: least prior training
    exposure first, then their order in ``datum_pool`` (which is loaded from
    the persisted corpus position). A group's current fill level does not
    affect that ordering.

    Saturated datums are excluded when strictly on-policy. When off-policy
    surplus is enabled and every datum is saturated, the least-saturated datum
    wins, with corpus order breaking ties. Dispensing stops once outstanding
    rewarded/untrained runs plus active leases fill the policy-valid horizon:
    ``(max_off_policy_generation + 1) * group_size * num_groups_per_batch``.
    """
    if not datum_pool:
        return None

    group_size = recipe.group_size

    # Bound rollout production by the complete policy-valid training horizon.
    # The current generation plus `max_off_policy_generation` older generations
    # may each contribute one batch. Counting both rewarded/untrained runs and
    # active leases prevents workers from reserving unbounded work that would
    # later be rejected as stale. Expired, rejected, trained, and eval runs are
    # already excluded from `scores`.
    if sum(scores.score.values()) >= outstanding_run_budget(recipe):
        return None

    unsaturated = [d for d in datum_pool if scores.score[d] < group_size]
    if unsaturated:
        return min(unsaturated, key=lambda d: scores.trained[d])

    if recipe.max_off_policy_generation <= 0:
        return None
    return min(datum_pool, key=lambda d: scores.score[d])


def pick_eval_datum(
    eval_pool: List[str],
    covered: Dict[str, int],
    group_size: int,
) -> Optional[str]:
    """First eval datum with fewer than ``group_size`` live attempts.

    ``covered`` comes from ``TunerServiceBase._eval_coverage``, which counts a
    datum's attempts against one checkpoint when they are rewarded or still
    pending (reward ``None``, lease not expired). Expired-unrewarded attempts
    don't count, so a dropped eval rollout is re-dispensed. Among under-filled
    datums the least-covered wins (spread coverage). Returns ``None`` when
    every eval datum already has ``group_size`` live attempts for this
    checkpoint, when the pool is empty, or when ``group_size <= 0``.

    Pure helper (no service/DB state), mirroring :func:`pick_datum`.
    """
    if not eval_pool or group_size <= 0:
        return None

    # Least-covered under-filled datum wins; None when all are full.
    best = min(eval_pool, key=lambda d: covered.get(d, 0))
    if covered.get(best, 0) >= group_size:
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
        # Everything inside this lock is on the critical path of every worker,
        # so it reads only per-datum aggregates -- never the run rows
        # themselves (see `_scheduler_scores`).
        async with self._dispense_lock:
            async with self.async_session() as session:
                async with session.begin():
                    datum_pool = await self._load_datums(
                        tuner_id, session, kind="train"
                    )
                    datum_id: Optional[str] = None
                    checkpoint_id: Optional[str] = None

                    # Eval remains the highest-priority tier.
                    if recipe.eval_group_size > 0:
                        latest = await self._latest_checkpoint(tuner_id, session)
                        if latest is not None:
                            eval_pool = await self._load_datums(
                                tuner_id, session, kind="eval"
                            )
                            covered = await self._eval_coverage(
                                tuner_id, eval_pool, latest.id, session
                            )
                            datum_id = pick_eval_datum(
                                eval_pool,
                                covered,
                                recipe.eval_group_size,
                            )
                            if datum_id is not None:
                                checkpoint_id = latest.id

                    if datum_id is None:
                        scores = await self._scheduler_scores(
                            tuner_id, datum_pool, session
                        )
                        datum_id = pick_datum(datum_pool, scores, recipe)

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
