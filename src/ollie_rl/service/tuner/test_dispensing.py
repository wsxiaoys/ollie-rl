"""Unit tests for the pure dispense scheduler."""

import unittest
from datetime import timedelta
from typing import Optional

from ollie_rl.cookbook import Recipe
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner.dispensing import pick_datum


def _pick_run(
    datum_id: str,
    *,
    reward: Optional[float] = None,
    trained_count: int = 0,
    rejected_count: int = 0,
    expires_in: float = 3600.0,
) -> RunModel:
    return RunModel(
        datum_id=datum_id,
        reward=reward,
        trained_count=trained_count,
        rejected_count=rejected_count,
        expires_at=utcnow() + timedelta(seconds=expires_in),
    )


class PickDatumTestCase(unittest.TestCase):
    def test_empty_pool_returns_none(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        self.assertIsNone(pick_datum([], [], recipe))

    def test_corpus_order_beats_closest_to_complete_group(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        runs = [_pick_run("d1"), _pick_run("d2"), _pick_run("d2")]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d1")

    def test_fresh_earlier_datum_beats_started_later_datum(self):
        recipe = Recipe(group_size=8, max_off_policy_generation=4)
        runs = [_pick_run("d2") for _ in range(4)]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d1")

    def test_started_group_can_fill_without_reward_wait(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=0)
        runs = [_pick_run("d1")]
        for _ in range(3):
            self.assertEqual(pick_datum(["d1"], runs, recipe), "d1")
            runs.append(_pick_run("d1"))
        self.assertIsNone(pick_datum(["d1"], runs, recipe))

    def test_fresh_datum_beats_saturated(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [_pick_run("d1", reward=1.0), _pick_run("d1", reward=1.0)]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_fresh_tiebreak_prefers_least_trained(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [_pick_run("d1", reward=1.0, trained_count=1)]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_saturated_dispatch_allowed_when_off_policy(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0),
            _pick_run("d2", reward=1.0),
            _pick_run("d2", reward=1.0),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_off_policy_dispatch_stops_at_outstanding_run_budget(self):
        recipe = Recipe(
            group_size=2,
            num_groups_per_batch=2,
            max_off_policy_generation=1,
        )
        # (1 current + 1 off-policy generation) * 2 runs * 2 groups = 8.
        runs = [_pick_run("d1", reward=1.0) for _ in range(8)]
        self.assertIsNone(pick_datum(["d1"], runs, recipe))

        runs[0].trained_count = 1
        self.assertEqual(pick_datum(["d1"], runs, recipe), "d1")

    def test_saturated_returns_none_when_strictly_on_policy(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=0)
        runs = [_pick_run("d1", reward=1.0), _pick_run("d1", reward=1.0)]
        self.assertIsNone(pick_datum(["d1"], runs, recipe))

    def test_rejected_and_expired_runs_not_counted(self):
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0, rejected_count=1),
            _pick_run("d1", expires_in=-1.0),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d1")


if __name__ == "__main__":
    unittest.main()
