"""Tests for the dispense scheduler.

The per-datum tallies live in SQL (``TunerServiceBase._scheduler_scores`` /
``_eval_coverage``) because dispensing recomputes them on every request and
must not pay for loading the run table. These tests therefore go through a
real (in-memory) database rather than hand-building tallies, so the counting
rules themselves stay covered: which runs are consumable, which are excluded,
and how prior training exposure orders the pool.
"""

import unittest
from datetime import timedelta
from typing import List, Optional

from sqlalchemy import select

from ollie_rl.cookbook import Recipe
from ollie_rl.db.connection import get_sessionmaker, init_db, shutdown_db
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner import TunerService
from ollie_rl.service.tuner.dispensing import (
    pick_datum,
    pick_eval_datum,
    pick_tier,
)

_TUNER = "tuner_test"


def _pick_run(
    datum_id: str,
    *,
    reward: Optional[float] = None,
    trained_count: int = 0,
    rejected_count: int = 0,
    expires_in: float = 3600.0,
    checkpoint_id: Optional[str] = None,
) -> RunModel:
    return RunModel(
        tuner_id=_TUNER,
        datum_id=datum_id,
        reward=reward,
        trained_count=trained_count,
        rejected_count=rejected_count,
        checkpoint_id=checkpoint_id,
        expires_at=utcnow() + timedelta(seconds=expires_in),
    )


class SchedulerTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        self.service = TunerService()

    async def asyncTearDown(self):
        await shutdown_db()

    async def _add(self, runs: List[RunModel]) -> None:
        async with get_sessionmaker()() as session:
            async with session.begin():
                for run in runs:
                    session.add(run)

    async def _scores(self, datum_pool: List[str]):
        async with get_sessionmaker()() as session:
            return await self.service._scheduler_scores(_TUNER, datum_pool, session)

    async def _coverage(self, eval_pool: List[str], checkpoint_id: str):
        async with get_sessionmaker()() as session:
            return await self.service._eval_coverage(
                _TUNER, eval_pool, checkpoint_id, session
            )

    async def _pick(self, datum_pool: List[str], recipe: Recipe) -> Optional[str]:
        return pick_datum(datum_pool, await self._scores(datum_pool), recipe)

    async def _mark_one_trained(self) -> None:
        async with get_sessionmaker()() as session:
            async with session.begin():
                run = (
                    await session.execute(
                        select(RunModel).where(RunModel.tuner_id == _TUNER).limit(1)
                    )
                ).scalar_one()
                run.trained_count = 1


class PickDatumTestCase(SchedulerTestCase):
    async def test_empty_pool_returns_none(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        self.assertIsNone(await self._pick([], recipe))

    async def test_corpus_order_beats_closest_to_complete_group(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        await self._add([_pick_run("d1"), _pick_run("d2"), _pick_run("d2")])
        self.assertEqual(await self._pick(["d1", "d2"], recipe), "d1")

    async def test_fresh_earlier_datum_beats_started_later_datum(self):
        recipe = Recipe(group_size=8, max_off_policy_generation=4)
        await self._add([_pick_run("d2") for _ in range(4)])
        self.assertEqual(await self._pick(["d1", "d2"], recipe), "d1")

    async def test_started_group_can_fill_without_reward_wait(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=0)
        await self._add([_pick_run("d1")])
        for _ in range(3):
            self.assertEqual(await self._pick(["d1"], recipe), "d1")
            await self._add([_pick_run("d1")])
        self.assertIsNone(await self._pick(["d1"], recipe))

    async def test_fresh_datum_beats_saturated(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        await self._add([_pick_run("d1", reward=1.0), _pick_run("d1", reward=1.0)])
        self.assertEqual(await self._pick(["d1", "d2"], recipe), "d2")

    async def test_fresh_tiebreak_prefers_least_trained(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        await self._add([_pick_run("d1", reward=1.0, trained_count=1)])
        self.assertEqual(await self._pick(["d1", "d2"], recipe), "d2")

    async def test_saturated_dispatch_allowed_when_off_policy(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        await self._add(
            [
                _pick_run("d1", reward=1.0),
                _pick_run("d1", reward=1.0),
                _pick_run("d1", reward=1.0),
                _pick_run("d2", reward=1.0),
                _pick_run("d2", reward=1.0),
            ]
        )
        self.assertEqual(await self._pick(["d1", "d2"], recipe), "d2")

    async def test_off_policy_dispatch_stops_at_outstanding_run_budget(self):
        recipe = Recipe(
            group_size=2,
            num_groups_per_batch=2,
            max_off_policy_generation=1,
        )
        # (1 current + 1 off-policy generation) * 2 runs * 2 groups = 8.
        await self._add([_pick_run("d1", reward=1.0) for _ in range(8)])
        self.assertIsNone(await self._pick(["d1"], recipe))

        # Training one run drops it out of the outstanding tally, freeing a slot.
        await self._mark_one_trained()
        self.assertEqual(await self._pick(["d1"], recipe), "d1")

    async def test_budget_stall_is_reported_as_its_own_tier(self):
        """A budget stall must not be labelled saturation.

        They clear differently -- one when the trainer catches up, the other
        only with more datums -- so a preview that conflates them points the
        reader at the wrong fix.
        """
        recipe = Recipe(
            group_size=2,
            num_groups_per_batch=2,
            max_off_policy_generation=1,
        )
        await self._add([_pick_run("d1", reward=1.0) for _ in range(8)])
        scores = await self._scores(["d1"])
        tier, reason = pick_tier("d1", scores.score, recipe)
        self.assertEqual(tier, "budget")
        self.assertIn("8/8", reason)

    async def test_saturated_returns_none_when_strictly_on_policy(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=0)
        await self._add([_pick_run("d1", reward=1.0), _pick_run("d1", reward=1.0)])
        self.assertIsNone(await self._pick(["d1"], recipe))

    async def test_rejected_and_expired_runs_not_counted(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        await self._add(
            [
                _pick_run("d1", reward=1.0),
                _pick_run("d1", reward=1.0, rejected_count=1),
                _pick_run("d1", expires_in=-1.0),
            ]
        )
        self.assertEqual(await self._pick(["d1", "d2"], recipe), "d1")


class SchedulerScoresTestCase(SchedulerTestCase):
    """The tally rules themselves, now that they are expressed in SQL."""

    async def test_trained_runs_count_as_exposure_not_as_score(self):
        await self._add(
            [
                _pick_run("d1", reward=1.0, trained_count=2),
                _pick_run("d1", reward=1.0),
            ]
        )
        scores = await self._scores(["d1"])
        self.assertEqual(scores.score["d1"], 1)
        self.assertEqual(scores.trained["d1"], 2)

    async def test_expired_run_counts_only_once_rewarded(self):
        # An expired lease with a reward is still consumable by a train step;
        # an expired lease without one is lost work.
        await self._add(
            [
                _pick_run("d1", reward=1.0, expires_in=-1.0),
                _pick_run("d2", expires_in=-1.0),
            ]
        )
        scores = await self._scores(["d1", "d2"])
        self.assertEqual(scores.score["d1"], 1)
        self.assertEqual(scores.score["d2"], 0)

    async def test_datums_outside_the_pool_are_ignored(self):
        # Eval runs live in the same table; they must not leak into the
        # training pool's tallies.
        await self._add([_pick_run("held-out", reward=1.0)])
        scores = await self._scores(["d1"])
        self.assertEqual(scores.score, {"d1": 0})
        self.assertEqual(scores.trained, {"d1": 0})


class PickEvalDatumTestCase(SchedulerTestCase):
    async def test_least_covered_datum_wins(self):
        await self._add(
            [
                _pick_run("e1", reward=1.0, checkpoint_id="ckpt-1"),
                _pick_run("e2", reward=1.0, checkpoint_id="ckpt-1"),
                _pick_run("e2", reward=1.0, checkpoint_id="ckpt-1"),
            ]
        )
        covered = await self._coverage(["e1", "e2"], "ckpt-1")
        self.assertEqual(pick_eval_datum(["e1", "e2"], covered, 2), "e1")

    async def test_none_when_every_datum_is_covered(self):
        await self._add(
            [_pick_run("e1", reward=1.0, checkpoint_id="ckpt-1") for _ in range(2)]
        )
        covered = await self._coverage(["e1"], "ckpt-1")
        self.assertIsNone(pick_eval_datum(["e1"], covered, 2))

    async def test_attempts_against_another_checkpoint_do_not_count(self):
        await self._add(
            [_pick_run("e1", reward=1.0, checkpoint_id="ckpt-0") for _ in range(2)]
        )
        covered = await self._coverage(["e1"], "ckpt-1")
        self.assertEqual(covered["e1"], 0)
        self.assertEqual(pick_eval_datum(["e1"], covered, 2), "e1")

    async def test_expired_unrewarded_attempt_is_redispensed(self):
        await self._add([_pick_run("e1", checkpoint_id="ckpt-1", expires_in=-1.0)])
        covered = await self._coverage(["e1"], "ckpt-1")
        self.assertEqual(covered["e1"], 0)
        self.assertEqual(pick_eval_datum(["e1"], covered, 1), "e1")

    async def test_zero_group_size_disables_the_tier(self):
        self.assertIsNone(pick_eval_datum(["e1"], {"e1": 0}, 0))
        self.assertIsNone(pick_eval_datum([], {}, 2))


if __name__ == "__main__":
    unittest.main()
