"""Pure dispense / scheduling algorithm for the tuner.

This module holds the *decision logic* behind ``POST /tuners/{id}/runs``,
factored out of :mod:`ollie_rl.service.tuner_service` so it can be read (and
unit-tested) in isolation from any service/DB state. Every function here is a
pure function of its arguments -- given the datum pool, the current runs, and
the recipe, it decides *which* datum to dispense next and *why*.

Two concerns live here:

1. **Which datum to dispense** -- :func:`pick_datum` (greedy most-full-first
   tiered scheduler) plus :func:`scheduler_scores` (the per-datum consumable
   score it ranks on) and :func:`pick_tier` (a human-readable label for a
   candidate, used by the progress preview).

2. **Which datums to skip** -- the opt-in quarantine filters:
   :func:`terminal_stats` and :func:`quarantined_datums`.

``tuner_service`` performs the DB reads (loading the pool, the runs, and the
per-run policy generations) and then feeds them to these helpers.
"""

from typing import Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel

from ollie_rl.cookbook import Recipe
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow


class RewardedRun(BaseModel):
    """A single rewarded run's outcome, keyed by ``run_id`` in the map the
    dispenser's quarantine logic consumes.

    Carries the run's ``datum_id`` (so per-datum tallies never need the full
    ``RunModel`` list) plus its ``reward``. The reward lets the success metric
    (``reward == 1.0``) be derived from the *same* rewarded map the length
    metric uses, so no separate "succeeded runs" query is needed.
    """

    datum_id: str
    reward: float


class TerminalStats(BaseModel):
    """Per-datum rewarded-attempt tallies over the datum's entire history.

    * ``rewarded`` -- runs that earned a reward; this is the quarantine
      denominator for length and success metrics.
    * ``length`` -- rewarded runs with at least one length-limited completion
      (the length numerator); a subset of ``rewarded``.
    * ``succeeded`` -- rewarded runs with ``reward == 1.0`` (the success
      numerator); a subset of ``rewarded``.
    """

    rewarded: int = 0
    length: int = 0
    succeeded: int = 0


def scheduler_scores(
    datum_pool: List[str],
    runs: List[RunModel],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Scheduler-view consumable score per datum (no staleness filter).

    Returns ``(score, trained)`` where ``score[datum]`` counts runs that
    are still *consumable* by a future train step from the scheduler's
    point of view (not yet trained, not requeued, and either rewarded or
    still pending/not expired) and ``trained[datum]`` accumulates prior
    training exposure for the fresh-tier tie-break.

    Shared by ``pick_datum`` (dispense decision) and the progress builder
    (``next_pick`` labeling) so the two never drift.
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
    return score, trained


# ---------------------------------------------------------------------------
# Dispenser quarantine (opt-in filters)
#
# `dispense_run` accepts two optional thresholds (both from `POST /runs` query
# params; omit/None to disable each):
#
#   * `max_length_rate` -- quarantine datums whose rewarded attempts too often
#     produce length-limited completions.
#   * `max_succeed_ratio` -- quarantine datums that are solved too reliably, so
#     compute isn't wasted on datums the policy has already mastered (they no
#     longer produce a useful learning signal).
#
# Both are computed from per-datum terminal tallies (`TerminalStats`), built once
# by `terminal_stats` and consumed by `quarantined_datums`. The moving parts:
#
#   * `_rewarded_datums` (DB, in tuner_service): per *rewarded* run, its
#     `RewardedRun` (datum id + reward). This is the quarantine denominator for
#     both filters. The reward lets the success numerator (`reward == 1.0`) be
#     derived from this same map -- no separate query. Every rewarded run for
#     the tuner is counted (no recency window).
#   * `_length_datums` (DB, in tuner_service): per run with at least one
#     length-limited completion, its datum id. `terminal_stats` intersects this
#     with `_rewarded_datums`, so the length numerator is a subset of the
#     rewarded denominator.
#   * Expired/lost status is tracked separately in `tuner_service` for progress
#     and run-list observability, but is intentionally not a quarantine metric
#     anymore.
#   * `terminal_stats` (pure): per-datum `TerminalStats(rewarded, length,
#     succeeded)`, computed straight from the already-windowed maps.
#   * `quarantined_datums` (pure): turns those tallies into the quarantine set
#     for whichever thresholds are supplied.
#   * `dispense_run`: filters the quarantined datums out of the candidate pool.
#
# Definitions and the reasoning behind them:
#
#   * Length-limited. A rewarded run with at least one completion whose
#     `finish_reason` is `"length"`. This can come directly from the model or
#     from the local context-window guard rewriting an oversized completion, so
#     repeated length samples are a clear signal that the datum often exhausts
#     the available generation budget.
#   * Succeeded. A rewarded run with `reward == 1.0`; a subset of the rewarded
#     runs, derived from `RewardedRun.reward`.
#   * Quarantine denominator. Both active quarantine metrics use rewarded
#     attempts only (expired/lost attempts are excluded because they no longer
#     drive quarantine):
#       - length rate   = length    / rewarded.
#       - success ratio = succeeded / rewarded.
#   * All-time counting. Every rewarded attempt for a datum is counted -- there
#     is no recency window, so all signals span the datum's entire history. A
#     quarantined datum receives no new runs, so once it crosses a threshold
#     (with enough samples) it stays quarantined; the metric only moves when new
#     rewarded attempts are recorded.
#   * Quarantine conditions (dispense passes `min_samples = 0.5 *
#     recipe.group_size`, i.e. half a group's worth of rewarded attempts).
#     Both gate on rewarded >= min_samples, then:
#       - length:  rate  >= max_length_rate.
#       - succeed: ratio >  max_succeed_ratio.
#
# Observability: `get_progress` reuses `terminal_stats` to populate
# `DatumProgress.length` / `rewarded` / `succeeded` (the raw tallies) so an
# operator can watch the numbers and pick sensible thresholds. It computes
# `DatumProgress.expired` separately for the expired/lost run-status split.
# ---------------------------------------------------------------------------
def terminal_stats(
    datum_pool: List[str],
    rewarded_by_run: Dict[str, RewardedRun],
    length_datum_by_run: Optional[Dict[str, str]] = None,
) -> Dict[str, TerminalStats]:
    """Per-datum ``TerminalStats`` tallies.

    Computed straight from ``run_id ->`` maps, each spanning the datum's entire
    history (no recency window; see the block comment above):

    * ``rewarded_by_run`` -- one entry per rewarded run (a :class:`RewardedRun`
      carrying its datum id + reward). Drives the rewarded quarantine
      denominator; its ``reward == 1.0`` entries drive the ``succeeded``
      numerator.
    * ``length_datum_by_run`` -- one entry per run with at least one
      length-limited completion. Only entries that are also present in
      ``rewarded_by_run`` count, keeping ``length`` a subset of ``rewarded``.
    """
    length_datum_by_run = length_datum_by_run or {}
    rewarded: Dict[str, int] = {d: 0 for d in datum_pool}
    length: Dict[str, int] = {d: 0 for d in datum_pool}
    succeeded: Dict[str, int] = {d: 0 for d in datum_pool}

    # Rewarded terminal attempts (denominator), plus their successes.
    for run in rewarded_by_run.values():
        if run.datum_id not in rewarded:
            continue
        rewarded[run.datum_id] += 1
        if run.reward == 1.0:
            succeeded[run.datum_id] += 1

    # Length-limited attempts: only rewarded runs count toward the quarantine
    # numerator, so intersect by run id with `rewarded_by_run`.
    for run_id, datum_id in length_datum_by_run.items():
        if run_id not in rewarded_by_run or datum_id not in length:
            continue
        length[datum_id] += 1

    return {
        d: TerminalStats(
            rewarded=rewarded[d],
            length=length[d],
            succeeded=succeeded[d],
        )
        for d in datum_pool
    }


def quarantined_datums(
    datum_pool: List[str],
    rewarded_by_run: Dict[str, RewardedRun],
    length_datum_by_run: Dict[str, str],
    *,
    min_samples: float,
    max_length_rate: Optional[float] = None,
    max_succeed_ratio: Optional[float] = None,
) -> Set[str]:
    """Datums to exclude from dispense, per the enabled quarantine filters.

    Built on :func:`terminal_stats`. Both filters share the same
    ``rewarded`` denominator and the ``min_samples`` gate; a datum is
    quarantined when it has at least ``min_samples`` rewarded attempts and
    *either* enabled filter fires:

    * length (``max_length_rate``): length rate ``length / rewarded >=
      max_length_rate`` -- it repeatedly produces length-limited completions.
    * too-easy (``max_succeed_ratio``): success ratio ``succeeded / rewarded >
      max_succeed_ratio`` -- it is solved too reliably.

    Passing ``None`` for a threshold disables that filter. Counts span the
    datum's entire history (no recency window): once a datum crosses a threshold
    with enough samples it stays quarantined, since it receives no new runs to
    move the metric.
    """
    stats = terminal_stats(datum_pool, rewarded_by_run, length_datum_by_run)
    excluded: Set[str] = set()
    for d, s in stats.items():
        if s.rewarded < min_samples:
            continue
        if max_length_rate is not None and s.length / s.rewarded >= max_length_rate:
            excluded.add(d)
            continue
        if (
            max_succeed_ratio is not None
            and s.succeeded / s.rewarded > max_succeed_ratio
        ):
            excluded.add(d)
    return excluded


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

    1. Started-but-incomplete groups (0 < count < group_size) come first,
       ordered by highest count, so the closest-to-complete group finishes
       ASAP. This minimizes the number of in-flight partial groups and gets
       complete groups ready for training as soon as possible.
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
    """
    if not datum_pool:
        return None

    group_size = recipe.group_size
    allow_surplus = recipe.max_off_policy_generation > 0

    score, trained = scheduler_scores(datum_pool, runs)

    def priority(datum: str) -> Tuple[int, int]:
        count = score[datum]
        if 0 < count < group_size:
            # Started but incomplete: finish closest-to-complete first.
            return (2, count)
        if count == 0:
            # Fresh: start new distinct groups before over-producing, and
            # prefer the least-trained datum so never-trained ones go first.
            return (1, -trained[datum])
        # Saturated (count >= group_size): the group is already complete, so
        # any further runs are surplus. Only dispatchable as off-policy
        # samples for a later train step; spread across the least-saturated.
        if allow_surplus:
            return (0, -count)
        # Strictly on-policy: surplus would be requeued, so don't dispatch.
        return (-1, 0)

    best = max(datum_pool, key=priority)
    if priority(best)[0] < 0:
        # All datums saturated and off-policy surplus is not allowed.
        return None
    return best
