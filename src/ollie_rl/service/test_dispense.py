"""Unit tests for the pure dispense / scheduling helpers.

Everything under test here is a pure function of its arguments (no service or
DB state), so these tests build plain in-memory inputs and assert on the
returned decision.
"""

import unittest
from datetime import timedelta
from typing import Optional

from ollie_rl.cookbook import Recipe
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.dispense import (
    RewardedRun,
    TerminalStats,
    pick_datum,
    quarantined_datums,
    terminal_stats,
)


def _rewarded(**runs: tuple[str, float]) -> dict[str, RewardedRun]:
    """Build a ``run_id -> RewardedRun`` map from ``run_id=(datum_id, reward)``."""
    return {
        run_id: RewardedRun(datum_id=datum_id, reward=reward)
        for run_id, (datum_id, reward) in runs.items()
    }


def _pick_run(
    datum_id: str,
    *,
    reward: Optional[float] = None,
    trained_count: int = 0,
    rejected_count: int = 0,
    expires_in: float = 3600.0,
) -> RunModel:
    """Build an in-memory (unpersisted) RunModel for pick_datum tests."""
    return RunModel(
        datum_id=datum_id,
        reward=reward,
        trained_count=trained_count,
        rejected_count=rejected_count,
        expires_at=utcnow() + timedelta(seconds=expires_in),
    )


class PickDatumTestCase(unittest.TestCase):
    """Unit tests for the pure, free-function pick_datum scheduler."""

    def test_empty_pool_returns_none(self):
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        self.assertIsNone(pick_datum([], [], recipe))

    def test_prefers_closest_to_complete_group(self):
        # d2 has more in-flight runs, so it is closer to completing its group.
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        runs = [
            _pick_run("d1"),
            _pick_run("d2"),
            _pick_run("d2"),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_started_group_beats_fresh_datum(self):
        # d1 has a partial group; d3 is fresh. Finish d1 first.
        recipe = Recipe(group_size=4, max_off_policy_generation=4)
        runs = [_pick_run("d1")]
        self.assertEqual(pick_datum(["d1", "d3"], runs, recipe), "d1")

    def test_fresh_datum_beats_saturated(self):
        # d1 is saturated (complete group), d2 is fresh -> start d2.
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_fresh_tiebreak_prefers_least_trained(self):
        # Both d1 and d2 have count == 0; d1 was trained before, d2 never was.
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [
            _pick_run("d1", reward=1.0, trained_count=1),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_saturated_dispatch_allowed_when_off_policy(self):
        # All datums saturated; off-policy allowed -> dispatch surplus to the
        # least-saturated datum (d2 has fewer runs than d1).
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0),
            _pick_run("d2", reward=1.0),
            _pick_run("d2", reward=1.0),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d2")

    def test_saturated_returns_none_when_strictly_on_policy(self):
        # All datums saturated and off-policy disabled -> nothing to dispatch.
        recipe = Recipe(group_size=2, max_off_policy_generation=0)
        runs = [
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0),
        ]
        self.assertIsNone(pick_datum(["d1"], runs, recipe))

    def test_rejected_and_expired_runs_not_counted(self):
        # d1 has 1 rewarded + 1 rejected + 1 expired-pending -> count == 1
        # (incomplete), so it still wins over the fresh d2.
        recipe = Recipe(group_size=2, max_off_policy_generation=4)
        runs = [
            _pick_run("d1", reward=1.0),
            _pick_run("d1", reward=1.0, rejected_count=1),
            _pick_run("d1", expires_in=-1.0),
        ]
        self.assertEqual(pick_datum(["d1", "d2"], runs, recipe), "d1")


class TerminalStatsTestCase(unittest.TestCase):
    """Unit tests for the pure terminal_stats tallier.

    Inputs are two maps: ``run_id -> RewardedRun`` (datum id + reward; the
    rewarded denominator, with ``reward == 1.0`` entries also feeding the
    success numerator) and ``run_id -> datum_id`` for expired, unrewarded runs
    (the expiration numerator). It returns a ``TerminalStats(expired, rewarded,
    succeeded)`` per datum.
    """

    def test_empty_maps_are_all_zero(self):
        stats = terminal_stats(["d1", "d2"], {}, {})
        self.assertEqual(
            stats,
            {
                "d1": TerminalStats(expired=0, rewarded=0, succeeded=0),
                "d2": TerminalStats(expired=0, rewarded=0, succeeded=0),
            },
        )

    def test_rewarded_counts_and_success_subset(self):
        # d1: 3 rewarded (2 of them reward==1.0) -> rewarded=3, succeeded=2.
        stats = terminal_stats(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0), r3=("d1", 0.0)),
            {},
        )
        self.assertEqual(stats["d1"], TerminalStats(expired=0, rewarded=3, succeeded=2))

    def test_expired_counts_toward_numerator_only(self):
        # Two expirations on d1 and nothing rewarded: expired=2, rewarded=0.
        stats = terminal_stats(["d1"], {}, {"r1": "d1", "r2": "d1"})
        self.assertEqual(stats["d1"], TerminalStats(expired=2, rewarded=0, succeeded=0))

    def test_mixed_rewarded_expired_and_succeeded(self):
        # d1: 3 rewarded (1 success) + 1 expired; d2: 2 expired only.
        stats = terminal_stats(
            ["d1", "d2"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 0.0), r3=("d1", 0.5)),
            {"r4": "d1", "r5": "d2", "r6": "d2"},
        )
        self.assertEqual(stats["d1"], TerminalStats(expired=1, rewarded=3, succeeded=1))
        self.assertEqual(stats["d2"], TerminalStats(expired=2, rewarded=0, succeeded=0))

    def test_runs_outside_pool_are_ignored(self):
        # Entries referencing datums not in the pool must not appear or crash.
        stats = terminal_stats(
            ["d1"],
            _rewarded(r1=("gone", 1.0)),
            {"r2": "gone"},
        )
        self.assertEqual(
            stats, {"d1": TerminalStats(expired=0, rewarded=0, succeeded=0)}
        )


class QuarantinedDatumsExpireTestCase(unittest.TestCase):
    """The expire filter of the pure quarantined_datums selector.

    A datum is quarantined when it holds at least ``min_samples`` terminal
    attempts (expired + rewarded) and the expiration rate (``expired /
    terminal``) is ``>= max_expire_rate``.
    """

    def test_no_datums_when_maps_empty(self):
        self.assertEqual(
            quarantined_datums(
                ["d1", "d2"], {}, {}, min_samples=2, max_expire_rate=0.5
            ),
            set(),
        )

    def test_quarantines_high_expire_rate(self):
        # d1: 3 expired / 3 terminal = 1.0 >= 0.5, samples 3 >= 2 -> quarantine.
        excluded = quarantined_datums(
            ["d1"],
            {},
            {"r1": "d1", "r2": "d1", "r3": "d1"},
            min_samples=2,
            max_expire_rate=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_not_quarantined_below_min_samples(self):
        # d1: 1 expired / 1 terminal = 1.0 rate, but only 1 sample < 2 samples.
        excluded = quarantined_datums(
            ["d1"], {}, {"r1": "d1"}, min_samples=2, max_expire_rate=0.5
        )
        self.assertEqual(excluded, set())

    def test_not_quarantined_below_rate(self):
        # d1: 1 expired / 4 terminal = 0.25 < 0.5, enough samples but low rate.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 0.0), r2=("d1", 0.0), r3=("d1", 0.0)),
            {"r4": "d1"},
            min_samples=2,
            max_expire_rate=0.5,
        )
        self.assertEqual(excluded, set())

    def test_rate_and_samples_thresholds_are_inclusive(self):
        # d1: 2 expired / 4 terminal = exactly 0.5 rate and exactly 4 samples,
        # matching both `>=` thresholds -> quarantined.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 0.0), r2=("d1", 0.0)),
            {"r3": "d1", "r4": "d1"},
            min_samples=4,
            max_expire_rate=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_only_flaky_datums_selected(self):
        # d1 is flaky (2/2 = 1.0); d2 is healthy (0/3 = 0.0).
        excluded = quarantined_datums(
            ["d1", "d2"],
            _rewarded(r1=("d2", 0.0), r2=("d2", 0.0), r3=("d2", 0.0)),
            {"r4": "d1", "r5": "d1"},
            min_samples=2,
            max_expire_rate=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_disabled_when_threshold_none(self):
        # No thresholds supplied -> nothing quarantined even at rate 1.0.
        excluded = quarantined_datums(
            ["d1"], {}, {"r1": "d1", "r2": "d1"}, min_samples=1
        )
        self.assertEqual(excluded, set())


class QuarantinedDatumsSucceedTestCase(unittest.TestCase):
    """The too-easy filter of the pure quarantined_datums selector.

    A datum is quarantined when it holds at least ``min_samples`` terminal
    attempts and the success ratio (``succeeded / terminal``, sharing the
    ``expired + rewarded`` denominator with the expire filter) is ``>
    max_succeed_ratio``.
    """

    def test_quarantines_high_success_ratio(self):
        # d1: 3/3 terminal succeed = 1.0 > 0.9, samples 3 >= 2 -> quarantine.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0), r3=("d1", 1.0)),
            {},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, {"d1"})

    def test_not_quarantined_below_min_samples(self):
        # d1: 1/1 = 1.0 ratio, but only 1 terminal sample < 2.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0)),
            {},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, set())

    def test_strictly_greater_than_threshold(self):
        # d1: 2/4 terminal = exactly 0.5, not > 0.5 -> not quarantined.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0), r3=("d1", 0.0), r4=("d1", 0.0)),
            {},
            min_samples=2,
            max_succeed_ratio=0.5,
        )
        self.assertEqual(excluded, set())

    def test_expired_runs_dilute_ratio(self):
        # d1: 2 succeed but terminal = 2 rewarded + 3 expired = 5, so
        # 2/5 = 0.4 <= 0.9 -> not too-easy (expirations share the denominator).
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0)),
            {"r3": "d1", "r4": "d1", "r5": "d1"},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, set())


class QuarantinedDatumsCombinedTestCase(unittest.TestCase):
    """Both filters together: a datum caught by either is excluded."""

    def test_union_of_both_filters(self):
        # d1 too flaky (2 expired / 2 terminal), d2 too easy (2/2 success),
        # d3 healthy (1/2 success, 0 expired).
        excluded = quarantined_datums(
            ["d1", "d2", "d3"],
            _rewarded(
                s1=("d2", 1.0),
                s2=("d2", 1.0),
                h1=("d3", 1.0),
                h2=("d3", 0.0),
            ),
            {"e1": "d1", "e2": "d1"},
            min_samples=2,
            max_expire_rate=0.5,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, {"d1", "d2"})


if __name__ == "__main__":
    unittest.main()
