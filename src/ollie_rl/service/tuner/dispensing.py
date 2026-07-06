"""Run dispensing: datum selection plus opt-in quarantine filtering."""

from datetime import timedelta
from typing import Dict, Optional

from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.constants import RUN_LEASE_SECONDS
from ollie_rl.service.tuner.dispense import pick_datum, quarantined_datums
from ollie_rl.types import DispenseRun


class DispenseMixin(TunerServiceBase):
    """Serialized read-pick-insert dispensing of runs for a tuner."""

    async def dispense_run(
        self,
        tuner_id: str,
        *,
        max_length_ratio: Optional[float] = None,
        max_succeed_ratio: Optional[float] = None,
    ) -> Optional[DispenseRun]:
        """
        Dispense a run for a tuner.

        When ``max_length_ratio`` is provided, datums that repeatedly produce
        length-limited completions are quarantined and excluded from the
        candidate pool. The rate is measured over *all* of the datum's rewarded
        attempts (no recency window) and a datum is skipped once it has
        accumulated at least ``0.5 * recipe.group_size`` rewarded attempts (half
        a group's worth) with a length rate ``>= max_length_ratio``. Length runs
        are rewarded runs with at least one completion whose finish reason is
        ``"length"`` and that received ``recipe.length_penalty``.
        When ``max_length_ratio`` is ``None`` this feature is disabled.

        When ``max_succeed_ratio`` is provided, datums that are solved too
        reliably are quarantined too: a datum is skipped once it has at least
        ``0.5 * recipe.group_size`` rewarded attempts and a success ratio (runs
        with reward ``== 1.0`` over rewarded attempts) that is
        ``> max_succeed_ratio``. Such datums are considered too easy and no
        longer produce a useful learning signal (see ``quarantined_datums``).
        When ``max_succeed_ratio`` is ``None`` this feature is disabled. Both
        filters may be combined; a datum caught by either is excluded.
        """
        # Ensure trainer is initialized.
        _trainer = await self._get_trainer(tuner_id)
        recipe = await self._recipe_for(tuner_id)

        # Serialize the read-pick-insert sequence: the scheduler decision in
        # `pick_datum` depends on the current set of in-flight runs, so two
        # concurrent dispenses that both read the pre-insert snapshot would
        # otherwise pick the same datum and over-dispense it past
        # `group_size` (the source of the inflated in_flight counts).
        async with self._dispense_lock:
            async with self.async_session() as session:
                datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)

                quarantine_enabled = (
                    max_length_ratio is not None or max_succeed_ratio is not None
                )
                if quarantine_enabled and datum_pool:
                    # Both filters share the rewarded denominator: a rewarded
                    # run has no in-flight row (deleted on success). Each
                    # `RewardedRun` carries its datum_id + reward, so the success
                    # numerator (reward == 1.0) is derived from this same map --
                    # no separate query. Counted over the datum's entire history
                    # (no recency window), so the pure helper just tallies.
                    rewarded_by_run = await self._rewarded_datums(tuner_id, session)
                    # The length numerator is only needed for the length filter.
                    # It is intersected with `rewarded_by_run` in the pure helper
                    # so length rate is length-limited rewarded attempts divided
                    # by all rewarded attempts.
                    length_datum_by_run: Dict[str, str] = {}
                    if max_length_ratio is not None:
                        length_datum_by_run = await self._length_datums(
                            tuner_id, session
                        )
                    excluded = quarantined_datums(
                        datum_pool,
                        rewarded_by_run,
                        length_datum_by_run,
                        min_samples=0.5 * recipe.group_size,
                        max_length_ratio=max_length_ratio,
                        max_succeed_ratio=max_succeed_ratio,
                    )
                    if excluded:
                        datum_pool = [d for d in datum_pool if d not in excluded]

            datum_id = pick_datum(datum_pool, runs, recipe)
            if datum_id is None:
                return None

            run_record = RunModel(
                tuner_id=tuner_id,
                datum_id=datum_id,
                reward=None,
                trained_count=0,
                expires_at=utcnow() + timedelta(seconds=RUN_LEASE_SECONDS),
            )
            async with self.async_session() as session:
                async with session.begin():
                    session.add(run_record)

        return DispenseRun(
            run_id=run_record.id,
            datum_id=run_record.datum_id,
            expires_at=run_record.expires_at,
        )
