"""Pure datum-quarantine predicate for the tuner dispense algorithm.

This is a *leaf* module: it depends only on :mod:`ollie_rl.service.tuner.types`
and the standard library, so both the data-access core
(:mod:`ollie_rl.service.tuner.base`) and the dispenser
(:mod:`ollie_rl.service.tuner.dispensing`) can import it without forming a
cycle.

It holds the two pure functions behind datum quarantine -- deciding *which*
datums to skip, independent of *which* datum to dispense next (that scheduler
lives in :mod:`ollie_rl.service.tuner.dispensing`):

* :func:`terminal_stats` -- per-datum rewarded-attempt tallies.
* :func:`quarantined_datums` -- turns those tallies into the excluded set for
  the enabled recipe thresholds.

Both are plain functions of their arguments (no service/DB state), so they can
be read and unit-tested in isolation. :class:`TunerServiceBase` wraps them in
an async helper (``_quarantined_datums``) that loads the DB stat maps and calls
:func:`quarantined_datums`, so every *decision* site (train dispense, eval
dispense, rejected-count bump) stays in lock-step.

# ---------------------------------------------------------------------------
# Dispenser quarantine (opt-in filters)
#
# Quarantine accepts two optional thresholds (both from the tuner's recipe;
# omit/None to disable each):
#
#   * `max_unhealthy_finish_ratio` -- quarantine datums whose rewarded attempts
#     too often end on an unhealthy finish reason (length-limited or malformed).
#   * `max_succeed_ratio` -- quarantine datums that are solved too reliably, so
#     compute isn't wasted on datums the policy has already mastered (they no
#     longer produce a useful learning signal).
#
# Both are computed from per-datum terminal tallies (`TerminalStats`), built once
# by `terminal_stats` and consumed by `quarantined_datums`. The moving parts:
#
#   * `_rewarded_datums` (DB, in tuner_service.base): per *rewarded* run, its
#     `RewardedRun` (datum id + reward). This is the quarantine denominator for
#     both filters. The reward lets the success numerator (`reward == 1.0`) be
#     derived from this same map -- no separate query. Every rewarded run for
#     the tuner is counted (no recency window).
#   * `_finish_reason_datums` (DB, in tuner_service.base): per run with a
#     behavior-penalty terminal `finish_reason`, that reason (`"length"` or
#     `"content_filter"`). `terminal_stats` joins it to `_rewarded_datums`: both
#     `length` and `content_filter` runs are tallied (each on its own axis) and
#     both count in `rewarded`. The unhealthy-finish filter sums them
#     (`length + content_filter`) over the full `rewarded` denominator.
#   * Expired/lost status is tracked separately in tuner_service for progress
#     and run-list observability, but is intentionally not a quarantine metric
#     anymore.
#   * `terminal_stats` (pure): per-datum `TerminalStats(rewarded, length,
#     succeeded, content_filter)`, computed straight from the already-windowed
#     maps.
#   * `quarantined_datums` (pure): turns those tallies into the quarantine set
#     for whichever thresholds are supplied.
#   * `TunerServiceBase._quarantined_datums`: the async wrapper that loads the
#     maps and filters the quarantined datums out of a candidate pool.
#
# Definitions and the reasoning behind them:
#
#   * Length-limited. A rewarded run with at least one completion whose
#     `finish_reason` is `"length"`. This can come directly from the model or
#     from the local context-window guard rewriting an oversized completion, so
#     repeated length samples are a clear signal that the datum often exhausts
#     the available generation budget.
#   * Content-filtered (malformed). A run with at least one completion whose
#     `finish_reason` is `"content_filter"`, terminated with the recipe's
#     `content_filter_penalty`. It counts as a real reward (in `rewarded`, like
#     batch/group accounting) and, like `length`, as an unhealthy finish: both
#     are auto-penalty degenerate rollouts (no verifier grade), so the
#     unhealthy-finish filter sums them into one numerator.
#   * Succeeded. A rewarded run with `reward == 1.0`; a subset of the rewarded
#     runs, derived from `RewardedRun.reward`.
#   * Quarantine denominators (expired/lost attempts never counted). Both
#     filters share the full `rewarded` denominator and the `min_samples` gate
#     (`rewarded >= min_samples`):
#       - unhealthy-finish rate = (length + content_filter) / rewarded.
#       - success ratio         = succeeded / rewarded.
#     `content_filter` is no longer subtracted anywhere: a malformed run is a
#     genuine (degenerate) rewarded attempt, so it counts both as a sample and
#     in the unhealthy-finish numerator alongside `length`.
#   * All-time counting. Every rewarded attempt for a datum is counted -- there
#     is no recency window, so all signals span the datum's entire history. A
#     quarantined datum receives no new runs, so once it crosses a threshold
#     (with enough samples) it stays quarantined; the metric only moves when new
#     rewarded attempts are recorded.
#   * Quarantine conditions (dispense passes `min_samples =
#     recipe.quarantine_min_samples`). Both gate on rewarded >= min_samples,
#     then:
#       - unhealthy: rate  >= max_unhealthy_finish_ratio.
#       - succeed:   ratio >= max_succeed_ratio.
#
# Observability: `get_progress` reuses `terminal_stats` to populate
# `DatumProgress.length` / `rewarded` / `succeeded` (the raw tallies) so an
# operator can watch the numbers and pick sensible thresholds. It computes
# `DatumProgress.expired` separately for the expired/lost run-status split.
# ---------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Set

from ollie_rl.service.tuner.types import RewardedRun, TerminalStats


def terminal_stats(
    datum_pool: List[str],
    rewarded_by_run: Dict[str, RewardedRun],
    finish_reason_by_run: Optional[Dict[str, str]] = None,
) -> Dict[str, TerminalStats]:
    """Per-datum ``TerminalStats`` tallies.

    Computed straight from ``run_id ->`` maps, each spanning the datum's entire
    history (no recency window; see the block comment above):

    * ``rewarded_by_run`` -- one entry per rewarded run (a :class:`RewardedRun`
      carrying its datum id + reward). Drives the rewarded quarantine
      denominator; its ``reward == 1.0`` entries drive the ``succeeded``
      numerator.
    * ``finish_reason_by_run`` -- one entry per run with a behavior-penalty
      terminal ``finish_reason`` (``"length"`` or ``"content_filter"``). Only
      entries also present in ``rewarded_by_run`` count. A ``"length"`` run is
      tallied as a length-limited rewarded attempt; a ``"content_filter"``
      (malformed) run is tallied on its own ``content_filter`` axis.

    ``rewarded`` counts *every* rewarded run (including content-filtered ones),
    matching ``scheduler_scores`` and batch/group accounting. The unhealthy-finish
    quarantine filter sums ``length + content_filter`` over this full ``rewarded``
    denominator (see :func:`quarantined_datums`); ``content_filter`` is not
    subtracted anywhere.
    """
    finish_reason_by_run = finish_reason_by_run or {}
    rewarded: Dict[str, int] = {d: 0 for d in datum_pool}
    length: Dict[str, int] = {d: 0 for d in datum_pool}
    succeeded: Dict[str, int] = {d: 0 for d in datum_pool}
    content_filter: Dict[str, int] = {d: 0 for d in datum_pool}

    for run_id, run in rewarded_by_run.items():
        if run.datum_id not in rewarded:
            continue
        rewarded[run.datum_id] += 1
        reason = finish_reason_by_run.get(run_id)
        if reason == "content_filter":
            content_filter[run.datum_id] += 1
        elif reason == "length":
            length[run.datum_id] += 1
        if run.reward == 1.0:
            succeeded[run.datum_id] += 1

    return {
        d: TerminalStats(
            rewarded=rewarded[d],
            length=length[d],
            succeeded=succeeded[d],
            content_filter=content_filter[d],
        )
        for d in datum_pool
    }


def quarantined_datums(
    datum_pool: List[str],
    rewarded_by_run: Dict[str, RewardedRun],
    finish_reason_by_run: Dict[str, str],
    *,
    min_samples: float,
    max_unhealthy_finish_ratio: Optional[float] = None,
    max_succeed_ratio: Optional[float] = None,
) -> Set[str]:
    """Datums to exclude from dispense, per the enabled quarantine filters.

    Built on :func:`terminal_stats`. Both filters share the full ``rewarded``
    denominator and the ``min_samples`` gate (``rewarded >= min_samples``);
    content-filtered (malformed) runs count as real rewarded samples like any
    other. Once the gate is met, a datum is quarantined when *either* enabled
    filter fires:

    * unhealthy-finish (``max_unhealthy_finish_ratio``): rate ``(length +
      content_filter) / rewarded >= max_unhealthy_finish_ratio`` -- its rewarded
      attempts too often end on an unhealthy finish reason. ``length``
      (length-limited) and ``content_filter`` (malformed) are both auto-penalty
      degenerate rollouts (no verifier grade), so they're summed into one
      numerator.
    * too-easy (``max_succeed_ratio``): success ratio ``succeeded / rewarded >=
      max_succeed_ratio`` -- it is solved too reliably.

    Passing ``None`` for a threshold disables that filter. Counts span the
    datum's entire history (no recency window): once a datum crosses a threshold
    with enough samples it stays quarantined, since it receives no new runs to
    move the metric.
    """
    stats = terminal_stats(datum_pool, rewarded_by_run, finish_reason_by_run)
    excluded: Set[str] = set()
    for d, s in stats.items():
        # Every rewarded run is a sample (including content-filtered/malformed
        # ones); both filters divide by the full `rewarded` count.
        if s.rewarded <= 0 or s.rewarded < min_samples:
            continue
        # Unhealthy-finish filter: `length` (length-limited) and `content_filter`
        # (malformed) are both auto-penalty degenerate rollouts with no verifier
        # grade, so sum them into one numerator over the full `rewarded`.
        if (
            max_unhealthy_finish_ratio is not None
            and (s.length + s.content_filter) / s.rewarded >= max_unhealthy_finish_ratio
        ):
            excluded.add(d)
            continue
        # Too-easy filter: solved too reliably.
        if (
            max_succeed_ratio is not None
            and s.succeeded / s.rewarded >= max_succeed_ratio
        ):
            excluded.add(d)
    return excluded
