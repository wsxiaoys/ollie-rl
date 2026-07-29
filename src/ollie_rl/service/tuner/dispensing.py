"""Run dispensing: datum selection plus opt-in quarantine filtering.

This module holds both the *pure decision logic* behind ``POST
/tuners/{id}/runs`` and the service mixin that wires it to the DB.

The pure scheduler helpers (``scheduler_scores``, ``pick_tier``,
``pick_datum``, ``pick_eval_datum``) are plain functions of their arguments --
given the datum pool, the current runs, and the recipe they decide *which*
datum to dispense next and *why*. They carry no service/DB state so they can be
read and unit-tested in isolation. The complementary *which datums to skip*
question -- the quarantine predicate (``terminal_stats`` /
``quarantined_datums``) -- lives in the leaf module
:mod:`ollie_rl.service.tuner.quarantine`, and is wired to the DB by
:meth:`TunerServiceBase._quarantined_datums`.

:class:`DispenseMixin` performs the DB reads (loading the pool and the runs)
and then feeds them to those helpers, applying the centralized quarantine
filter before the train pick. Eval picks are never quarantine-filtered:
held-out datums are scored every checkpoint regardless of signal health.
"""

from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple

from sqlalchemy import select

from ollie_rl.cookbook import Recipe
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.constants import RUN_LEASE_SECONDS
from ollie_rl.service.tuner.errors import RunNotFoundError
from ollie_rl.service.tuner.types import SchedulerScores
from ollie_rl.types import DispenseRun, TunerStatus


def scheduler_scores(
    datum_pool: List[str],
    runs: List[RunModel],
) -> SchedulerScores:
    """Scheduler-view consumable tallies per datum (no staleness filter).

    Returns a :class:`SchedulerScores` (``score`` / ``trained`` / ``rewarded``
    maps; see its docstring for the semantics of each).

    Shared by ``pick_datum`` (dispense decision) and the progress builder
    (``next_pick`` labeling) so the two never drift.

    Note: ``rewarded`` here intentionally counts *every* run with a reward set,
    including content-filtered (malformed) runs -- unlike the quarantine
    denominator in :func:`terminal_stats`, which excludes them. The two measure
    different things: this is batch/group accounting, and a malformed run is a
    first-class GRPO sample (it carries the ``content_filter_penalty`` reward
    and trains with a negative advantage; see ``_collect_consumable_batch``), so
    it must count toward ``group_size``. Excluding it here would leave any group
    with a malformed run permanently short of ``group_size``, over-dispensing to
    replace a sample that is actually trainable. Quarantine, by contrast, asks
    whether the datum still yields a useful signal *distribution*, where a
    malformed outcome is noise -- hence the deliberate mismatch.
    """
    now = utcnow()
    score = {d: 0 for d in datum_pool}
    trained = {d: 0 for d in datum_pool}
    rewarded = {d: 0 for d in datum_pool}
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
        if has_reward:
            rewarded[r.datum_id] += 1
    return SchedulerScores(score=score, trained=trained, rewarded=rewarded)


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

    1. Started groups (0 < count < group_size) are filled via the two-phase
       probe gate below, so the closest-to-complete (already-probed) group
       finishes ASAP -- getting complete groups ready for training soonest.
    2. Fresh datums (count == 0) come next, so we start new distinct groups
       before over-producing existing ones. Among fresh datums the
       least-trained one wins, so never-trained datums are sampled before
       re-sampling datums that already contributed a trained group (better
       dataset coverage).
    3. Saturated datums (count >= group_size) come last, and only when
       off-policy samples are allowed (``max_off_policy_generation > 0``):
       the surplus runs can be consumed by a later train step within the
       off-policy window. Among saturated datums we prefer the
       least-saturated to spread the surplus. When off-policy is disabled
       the surplus would just be requeued, so saturated datums are excluded
       and ``None`` is returned if nothing else is available.

    **Two-phase probe gate** (always active). The quarantine filters decide a
    datum is problematic after just ``recipe.quarantine_min_samples`` rewarded
    attempts, so filling a whole group before that verdict wastes rollouts on
    datums that will be quarantined anyway. A started group is therefore
    dispensed in two phases:

    * **Probe** -- dispense at most ``probe = recipe.quarantine_min_samples``
      runs, then *hold*: while the group has reached ``probe`` dispensed runs
      but fewer than ``probe`` rewards have returned, the datum is not dispensed
      again (the scheduler moves on to fresh datums / other probes). ``None`` is
      returned only if every datum is held or saturated.
    * **Fill** -- once ``probe`` rewards are back (and the datum was not
      quarantined out of the pool upstream), the rest of the group is dispensed,
      taking top priority so ready-to-train groups finish fast.
    """
    if not datum_pool:
        return None

    group_size = recipe.group_size
    allow_surplus = recipe.max_off_policy_generation > 0
    # `quarantine_min_samples` rewarded attempts is enough to run the quarantine
    # check (matches the dispenser's `min_samples`), so the probe phase caps
    # dispensing at this many runs and holds for their rewards.
    probe = recipe.quarantine_min_samples

    scores = scheduler_scores(datum_pool, runs)

    def priority(datum: str) -> Tuple[int, int]:
        count = scores.score[datum]
        if count >= group_size:
            # Saturated: the group is already complete, so any further runs are
            # surplus. Only dispatchable as off-policy samples for a later train
            # step; spread across the least-saturated.
            if allow_surplus:
                return (0, -count)
            # Strictly on-policy: surplus would be requeued, so don't dispatch.
            return (-2, 0)
        if count == 0:
            # Fresh: start new distinct groups before over-producing, and
            # prefer the least-trained datum so never-trained ones go first.
            return (1, -scores.trained[datum])
        # Two-phase probe gate.
        if scores.rewarded[datum] >= probe:
            # Probe cleared quarantine (problematic datums are already filtered
            # out of the pool upstream): fill the rest of the group, ahead of
            # starting fresh ones.
            return (3, count)
        if count < probe:
            # Still probing: keep dispensing up to `probe` runs.
            return (2, count)
        # Probe capacity reached but not enough rewards are back yet: hold so we
        # don't over-commit a datum that may still be quarantined.
        return (-1, 0)

    best = max(datum_pool, key=priority)
    if priority(best)[0] < 0:
        # Nothing dispensable: all datums are saturated (on-policy) or awaiting
        # probe results.
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
        """Dispense one run while preserving the original service contract."""
        runs = await self.dispense_runs(tuner_id, batch_size=1)
        return runs[0] if runs else None

    async def release_run(
        self, tuner_id: str, run_id: str
    ) -> Literal["released", "already_rewarded", "already_expired"]:
        """Expire an unrewarded run's lease immediately.

        Rewarded and already-expired runs are successful no-ops so callers can
        safely fire-and-forget retries. Only an unknown run is an error.
        """
        async with self.async_session() as session:
            async with session.begin():
                record = await session.scalar(
                    select(RunModel).where(
                        RunModel.id == run_id,
                        RunModel.tuner_id == tuner_id,
                    )
                )
                if record is None:
                    raise RunNotFoundError(
                        f"Run '{run_id}' not found under tuner '{tuner_id}'"
                    )

                if record.reward is not None:
                    return "already_rewarded"

                now = utcnow()
                if record.expires_at <= now:
                    return "already_expired"

                record.expires_at = now
                record.updated_at = now

        return "released"

    async def dispense_runs(self, tuner_id: str, batch_size: int) -> List[DispenseRun]:
        """Dispense up to ``batch_size`` runs from one scheduler snapshot.

        Quarantine is computed once because the newly leased runs have neither
        rewards nor completion finish reasons. Each provisional run is appended
        to the in-memory snapshot before the next pick so eval coverage,
        two-phase probe gating, and group capacity match repeated scalar calls.
        The batch is inserted atomically in the same transaction as its reads.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        # Do not lease work to an external sandbox until the backend can serve
        # it. A pending/cancelled tuner follows the existing no-work path.
        trainer = await self._get_trainer(tuner_id)
        if await trainer.get_status() is not TunerStatus.IN_PROGRESS:
            return []

        recipe = await self._recipe_for(tuner_id)
        run_records: List[RunModel] = []

        # Serialize the complete read-pick-insert sequence. The transaction and
        # lock cover the full batch so no local dispenser can observe a partial
        # scheduler decision.
        async with self._dispense_lock:
            async with self.async_session() as session:
                async with session.begin():
                    datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)

                    latest = None
                    eval_pool: List[str] = []
                    if recipe.eval_group_size > 0:
                        latest = await self._latest_checkpoint(tuner_id, session)
                        if latest is not None:
                            eval_pool = await self._load_datums(
                                tuner_id, session, kind="eval"
                            )

                    train_pool: Optional[List[str]] = None
                    for _ in range(batch_size):
                        datum_id: Optional[str] = None
                        checkpoint_id: Optional[str] = None

                        # Eval remains the highest-priority tier on every pick.
                        if latest is not None:
                            datum_id = pick_eval_datum(
                                eval_pool,
                                runs,
                                latest.id,
                                recipe.eval_group_size,
                            )
                            if datum_id is not None:
                                checkpoint_id = latest.id

                        if datum_id is None:
                            # Delay the all-history quarantine queries until the
                            # batch actually needs its first training lease.
                            if train_pool is None:
                                excluded = await self._quarantined_datums(
                                    tuner_id, session, datum_pool, recipe
                                )
                                train_pool = [
                                    datum
                                    for datum in datum_pool
                                    if datum not in excluded
                                ]
                            datum_id = pick_datum(train_pool, runs, recipe)

                        if datum_id is None:
                            break

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
                        run_records.append(run_record)
                        runs.append(run_record)

                    # Materialize generated run IDs before constructing DTOs.
                    await session.flush()

        return [
            DispenseRun(
                run_id=run.id,
                datum_id=run.datum_id,
                expires_at=run.expires_at,
            )
            for run in run_records
        ]
