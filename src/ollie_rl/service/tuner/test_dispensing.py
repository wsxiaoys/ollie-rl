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
from ollie_rl.service.tuner.dispensing import pick_datum
from ollie_rl.service.tuner.quarantine import quarantined_datums, terminal_stats
from ollie_rl.service.tuner.types import RewardedRun, TerminalStats


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

    Inputs are a map of rewarded runs and a ``run_id -> finish_reason`` map of
    behavior-penalty runs (``"length"`` / ``"content_filter"``). It returns a
    ``TerminalStats(rewarded, length, succeeded, content_filter)`` per datum.
    ``rewarded`` counts every rewarded run (including content-filtered ones) and
    is the shared quarantine denominator; the unhealthy-finish numerator
    (``length + content_filter``) is summed by ``quarantined_datums``.
    """

    def test_empty_maps_are_all_zero(self):
        stats = terminal_stats(["d1", "d2"], {}, {})
        self.assertEqual(
            stats,
            {
                "d1": TerminalStats(),
                "d2": TerminalStats(),
            },
        )

    def test_rewarded_counts_and_success_subset(self):
        # d1: 3 rewarded (2 of them reward==1.0) -> rewarded=3, succeeded=2.
        stats = terminal_stats(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0), r3=("d1", 0.0)),
            {},
        )
        self.assertEqual(stats["d1"], TerminalStats(rewarded=3, succeeded=2))

    def test_length_counts_rewarded_subset_only(self):
        # r2 has a length-limited completion but is not rewarded, so it is not
        # counted toward the length quarantine numerator.
        stats = terminal_stats(
            ["d1"],
            _rewarded(r1=("d1", 0.0), r3=("d1", 1.0)),
            {"r1": "length", "r2": "length"},
        )
        self.assertEqual(stats["d1"], TerminalStats(rewarded=2, length=1, succeeded=1))

    def test_mixed_rewarded_length_and_succeeded(self):
        # d1: 3 rewarded (1 success), 1 length; d2: no rewarded attempts.
        stats = terminal_stats(
            ["d1", "d2"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 0.0), r3=("d1", 0.5)),
            {"r2": "length"},
        )
        self.assertEqual(stats["d1"], TerminalStats(rewarded=3, length=1, succeeded=1))
        self.assertEqual(stats["d2"], TerminalStats())

    def test_content_filter_counted_in_rewarded_on_own_axis(self):
        # d1: r1/r2 length-limited, r3 content-filtered (malformed). The
        # content-filtered run counts in `rewarded` (consistent with batch/group
        # accounting) and is tracked on the `content_filter` axis; downstream the
        # unhealthy-finish numerator sums `length + content_filter`.
        stats = terminal_stats(
            ["d1"],
            _rewarded(r1=("d1", -10.0), r2=("d1", -10.0), r3=("d1", -1.0)),
            {"r1": "length", "r2": "length", "r3": "content_filter"},
        )
        self.assertEqual(
            stats["d1"],
            TerminalStats(rewarded=3, length=2, succeeded=0, content_filter=1),
        )

    def test_runs_outside_pool_are_ignored(self):
        # Entries referencing datums not in the pool must not appear or crash.
        stats = terminal_stats(
            ["d1"],
            _rewarded(r1=("gone", 1.0)),
            {"r1": "length"},
        )
        self.assertEqual(stats, {"d1": TerminalStats()})


class QuarantinedDatumsUnhealthyFinishTestCase(unittest.TestCase):
    """The unhealthy-finish filter of the pure quarantined_datums selector.

    The numerator sums both auto-penalty finish reasons (``length`` +
    ``content_filter``) over the full ``rewarded`` denominator; the
    ``min_samples`` gate is on ``rewarded`` too.
    """

    def test_no_datums_when_maps_empty(self):
        self.assertEqual(
            quarantined_datums(
                ["d1", "d2"], {}, {}, min_samples=2, max_unhealthy_finish_ratio=0.5
            ),
            set(),
        )

    def test_quarantines_high_unhealthy_finish_rate(self):
        # d1: 2 length / 3 rewarded = 0.67 >= 0.5, samples 3 >= 2.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 0.0), r2=("d1", 0.0), r3=("d1", 1.0)),
            {"r1": "length", "r2": "length"},
            min_samples=2,
            max_unhealthy_finish_ratio=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_content_filter_counts_toward_unhealthy_numerator(self):
        # d1: 2 length + 1 content-filtered (malformed). Both feed the
        # unhealthy-finish numerator: (2 + 1) / 3 rewarded = 1.0 (>= 1.0), and
        # rewarded 3 >= 2 min-samples -> quarantined. The malformed run no longer
        # rescues the datum; it counts as another unhealthy finish.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", -10.0), r2=("d1", -10.0), r3=("d1", -1.0)),
            {"r1": "length", "r2": "length", "r3": "content_filter"},
            min_samples=2,
            max_unhealthy_finish_ratio=1.0,
        )
        self.assertEqual(excluded, {"d1"})

    def test_not_quarantined_below_min_samples(self):
        # d1: 1 length / 1 rewarded = 1.0 rate, but only 1 sample < 2 samples.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 0.0)),
            {"r1": "length"},
            min_samples=2,
            max_unhealthy_finish_ratio=0.5,
        )
        self.assertEqual(excluded, set())

    def test_content_filter_counts_toward_min_samples(self):
        # d1: 2 length + 1 content-filtered. Every rewarded run is a sample, so
        # rewarded 3 >= 3 min-samples meets the gate and (2 + 1) / 3 = 1.0
        # (>= 1.0) -> quarantined. (content_filter is a genuine degenerate
        # rewarded attempt, so it counts both as a sample and in the numerator.)
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", -10.0), r2=("d1", -10.0), r3=("d1", -1.0)),
            {"r1": "length", "r2": "length", "r3": "content_filter"},
            min_samples=3,
            max_unhealthy_finish_ratio=1.0,
        )
        self.assertEqual(excluded, {"d1"})

    def test_content_filter_and_length_combine_to_quarantine(self):
        # Example: 3 length + 1 content-filtered run, min_samples=4. rewarded=4
        # meets the gate and the unhealthy-finish numerator sums both reasons:
        # (3 + 1) / 4 = 1.0 (>= 1.0) -> quarantined immediately (no extra
        # dispense needed, since content_filter is a real degenerate sample).
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", -10.0),
                r2=("d1", -10.0),
                r3=("d1", -10.0),
                r4=("d1", -1.0),
            ),
            {
                "r1": "length",
                "r2": "length",
                "r3": "length",
                "r4": "content_filter",
            },
            min_samples=4,
            max_unhealthy_finish_ratio=1.0,
        )
        self.assertEqual(excluded, {"d1"})

    def test_mixed_reasons_below_min_samples_not_quarantined(self):
        # d1: 1 length + 1 content-filtered, rewarded=2 < 3 min-samples -> gate
        # not met even though the unhealthy rate is 2/2 = 1.0.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", -10.0), r2=("d1", -1.0)),
            {"r1": "length", "r2": "content_filter"},
            min_samples=3,
            max_unhealthy_finish_ratio=1.0,
        )
        self.assertEqual(excluded, set())

    def test_not_quarantined_below_rate(self):
        # d1: 1 length / 4 rewarded = 0.25 < 0.5.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", 0.0),
                r2=("d1", 0.0),
                r3=("d1", 0.0),
                r4=("d1", 0.0),
            ),
            {"r4": "length"},
            min_samples=2,
            max_unhealthy_finish_ratio=0.5,
        )
        self.assertEqual(excluded, set())

    def test_combined_reasons_cross_rate_where_neither_alone_would(self):
        # d1: 1 length + 1 content-filtered over 4 rewarded. Neither reason alone
        # reaches 0.5 (each is 1/4 = 0.25), but summed the unhealthy-finish rate
        # is (1 + 1) / 4 = 0.5 (>= 0.5) -> quarantined. Demonstrates the combined
        # numerator.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", -10.0),
                r2=("d1", -1.0),
                r3=("d1", 0.0),
                r4=("d1", 0.0),
            ),
            {"r1": "length", "r2": "content_filter"},
            min_samples=4,
            max_unhealthy_finish_ratio=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_rate_and_samples_thresholds_are_inclusive(self):
        # d1: 2 length / 4 rewarded = exactly 0.5 and exactly 4 samples.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", 0.0),
                r2=("d1", 0.0),
                r3=("d1", 0.0),
                r4=("d1", 0.0),
            ),
            {"r3": "length", "r4": "length"},
            min_samples=4,
            max_unhealthy_finish_ratio=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_disabled_when_threshold_none(self):
        # No thresholds supplied -> nothing quarantined even at rate 1.0.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 0.0), r2=("d1", 0.0)),
            {"r1": "length", "r2": "length"},
            min_samples=1,
        )
        self.assertEqual(excluded, set())


class QuarantinedDatumsSucceedTestCase(unittest.TestCase):
    """The too-easy filter of the pure quarantined_datums selector.

    A datum is quarantined when it holds at least ``min_samples`` rewarded
    attempts and the success ratio (``succeeded / rewarded``) is ``>=``
    ``max_succeed_ratio``.
    """

    def test_quarantines_high_success_ratio(self):
        # d1: 3/3 rewarded succeed = 1.0 >= 0.9, samples 3 >= 2 -> quarantine.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0), r3=("d1", 1.0)),
            {},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, {"d1"})

    def test_not_quarantined_below_min_samples(self):
        # d1: 1/1 = 1.0 ratio, but only 1 rewarded sample < 2.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0)),
            {},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, set())

    def test_at_threshold_is_quarantined(self):
        # d1: 2/4 rewarded = exactly 0.5, >= 0.5 -> quarantined (inclusive).
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", 1.0),
                r2=("d1", 1.0),
                r3=("d1", 0.0),
                r4=("d1", 0.0),
            ),
            {},
            min_samples=2,
            max_succeed_ratio=0.5,
        )
        self.assertEqual(excluded, {"d1"})

    def test_below_threshold_not_quarantined(self):
        # d1: 1/4 rewarded = 0.25, < 0.5 -> not quarantined.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", 1.0),
                r2=("d1", 0.0),
                r3=("d1", 0.0),
                r4=("d1", 0.0),
            ),
            {},
            min_samples=2,
            max_succeed_ratio=0.5,
        )
        self.assertEqual(excluded, set())

    def test_expired_runs_do_not_dilute_ratio(self):
        # Expired runs are no longer part of quarantine math. With 2/2 rewarded
        # successes, this datum is too easy despite separate expired attempts.
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0)),
            {},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, {"d1"})

    def test_content_filter_not_counted_as_success(self):
        # d1: 2 successes + 1 content-filtered (malformed). The success ratio
        # uses the full `rewarded` denominator and the malformed run is NOT a
        # success: 2/3 = 0.67 < 0.9 -> NOT quarantined by the too-easy filter.
        # (With the unhealthy-finish filter disabled here, the malformed run only
        # affects the denominator, never the success numerator.)
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(r1=("d1", 1.0), r2=("d1", 1.0), r3=("d1", -1.0)),
            {"r3": "content_filter"},
            min_samples=2,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, set())

    def test_content_filter_keeps_datum_used_at_full_succeed_ratio(self):
        # Example: 3 successes + 1 content-filtered run, max_succeed_ratio=1.0.
        # The gate is met (rewarded = 4 >= 3) so the success filter runs over the
        # full `rewarded` denominator: 3/4 = 0.75 < 1.0 -> NOT quarantined, i.e.
        # the datum is still used. (The malformed run is not a success, so it
        # drags the success ratio below the threshold.)
        excluded = quarantined_datums(
            ["d1"],
            _rewarded(
                r1=("d1", 1.0),
                r2=("d1", 1.0),
                r3=("d1", 1.0),
                r4=("d1", -1.0),
            ),
            {"r4": "content_filter"},
            min_samples=3,
            max_succeed_ratio=1.0,
        )
        self.assertEqual(excluded, set())


class QuarantinedDatumsCombinedTestCase(unittest.TestCase):
    """Both filters together: a datum caught by either is excluded."""

    def test_union_of_both_filters(self):
        # d1 too many unhealthy finishes (2/2 length), d2 too easy (2/2 success),
        # d3 healthy (1/2 success, 0 unhealthy).
        excluded = quarantined_datums(
            ["d1", "d2", "d3"],
            _rewarded(
                l1=("d1", 0.0),
                l2=("d1", 0.0),
                s1=("d2", 1.0),
                s2=("d2", 1.0),
                h1=("d3", 1.0),
                h2=("d3", 0.0),
            ),
            {"l1": "length", "l2": "length"},
            min_samples=2,
            max_unhealthy_finish_ratio=0.5,
            max_succeed_ratio=0.9,
        )
        self.assertEqual(excluded, {"d1", "d2"})


if __name__ == "__main__":
    unittest.main()
