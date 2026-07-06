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

2. **Which datums to skip** -- the opt-in *expiration quarantine* filter:
   :func:`expiration_stats` and :func:`expiring_datums`. See the block comment
   above :func:`expiration_stats` for the full rationale.

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
    (``reward == 1.0``) be derived from the *same* rewarded map the expiration
    metric uses, so no separate "succeeded runs" query is needed.
    """

    datum_id: str
    reward: float


class TerminalStats(BaseModel):
    """Per-datum terminal-attempt tallies over the datum's entire history.

    * ``expired`` -- expired, unrewarded runs (the expiration numerator).
    * ``rewarded`` -- runs that earned a reward; with ``expired`` it forms the
      shared ``terminal`` denominator (``expired + rewarded``) both metrics use.
    * ``succeeded`` -- rewarded runs with ``reward == 1.0`` (the success
      numerator); a subset of ``rewarded``.
    """

    expired: int
    rewarded: int
    succeeded: int


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
#   * `max_expire_rate` -- quarantine datums that genuinely keep expiring, so
#     compute isn't wasted on datums that never finish in time.
#   * `max_succeed_ratio` -- quarantine datums that are solved too reliably, so
#     compute isn't wasted on datums the policy has already mastered (they no
#     longer produce a useful learning signal).
#
# Both are computed from the same per-datum terminal tallies (`TerminalStats`),
# built once by `terminal_stats` and consumed by a single `quarantined_datums`
# helper. The moving parts:
#
#   * `_rewarded_datums` (DB, in tuner_service): per *rewarded* run, its
#     `RewardedRun` (datum id + reward). Drives the *rewarded* denominator for
#     both filters; the reward lets the success numerator (`reward == 1.0`) be
#     derived from this same map -- no separate query. Every rewarded run for
#     the tuner is counted (no recency window).
#   * `_expired_datums` (DB, in tuner_service): per *expired, unrewarded* run,
#     its datum id. A run is `expired` when it either still has a lingering
#     `InFlightChatCompletionModel` row (generation stalled) or its summed
#     completion `duration_ms` is at/past the expiration threshold
#     (`RUN_EXPIRE_GENERATION_BUDGET_MS`). The "expired and unrewarded" filter
#     lives in SQL, so ongoing runs (still within their lease) never appear.
#     Drives the *expiration* numerator (only fetched when `max_expire_rate`
#     is set).
#   * `terminal_stats` (pure): per-datum `TerminalStats(expired, rewarded,
#     succeeded)`, computed straight from the two already-windowed maps (each
#     entry carries its datum id, so no `RunModel` list and no recency filter
#     are needed).
#   * `quarantined_datums` (pure): turns those tallies into the quarantine set
#     for whichever thresholds are supplied.
#   * `dispense_run`: filters the quarantined datums out of the candidate pool.
#
# Definitions and the reasoning behind them:
#
#   * Expired vs lost. An expired, unrewarded run (`reward is None and
#     expires_at <= now`) is classified `expired` (the status the dispenser
#     quarantines on) when it matches either compute-waste signal: (a) it still
#     has a lingering in-flight op -- an op that timed out / was cancelled and is
#     still progressing on the backend, i.e. the *generation itself* was too slow
#     to finish within the lease; or (b) its summed completion `duration_ms` is
#     at/past the expiration threshold (`RUN_EXPIRE_GENERATION_BUDGET_MS`) -- it burned
#     real compute yet never finished. Both are the "never finishes in time"
#     waste quarantine targets.
#     Runs matching neither are `lost` and ignored: a crashed/abandoned worker,
#     or a run whose ops all completed but which was abandoned before a reward --
#     neither reflects a genuinely hard datum, so a flaky worker never poisons
#     it. The "expired and unrewarded" filter and each run's recency are resolved
#     when the two maps are built (the expired map is pre-filtered to expired,
#     unrewarded runs; the rewarded map holds the rewarded runs, which have no
#     lingering in-flight row since it is deleted on success).
#   * Succeeded. A rewarded run with `reward == 1.0`; a subset of the rewarded
#     runs, derived from `RewardedRun.reward`.
#   * Both metrics share the terminal denominator, where terminal = expired +
#     rewarded (in-flight runs are excluded -- outcome unknown):
#       - expire rate    = expired   / terminal.
#       - success ratio  = succeeded / terminal.
#   * All-time counting. Every terminal attempt for a datum is counted -- there
#     is no recency window, so all signals span the datum's entire history. A
#     quarantined datum receives no new runs, so once it crosses a threshold
#     (with enough samples) it stays quarantined; the metric only moves when new
#     terminal attempts are recorded.
#   * Quarantine conditions (dispense passes `min_samples = 0.5 *
#     recipe.group_size`, i.e. half a group's worth of terminal attempts).
#     Both gate on terminal >= min_samples, then:
#       - expire:  rate  >= max_expire_rate.
#       - succeed: ratio >  max_succeed_ratio.
#
# Observability: `get_progress` reuses `terminal_stats` to populate
# `DatumProgress.expired` / `rewarded` / `succeeded` (the raw tallies) so an
# operator can watch the numbers and pick sensible thresholds. It builds the
# same maps so the dashboard numbers match what the dispenser would quarantine
# on.
# ---------------------------------------------------------------------------
def terminal_stats(
    datum_pool: List[str],
    rewarded_by_run: Dict[str, RewardedRun],
    expired_datum_by_run: Dict[str, str],
) -> Dict[str, TerminalStats]:
    """Per-datum ``TerminalStats`` tallies.

    Computed straight from two ``run_id ->`` maps, each spanning the datum's
    entire history (no recency window; see the block comment above):

    * ``rewarded_by_run`` -- one entry per rewarded run (a :class:`RewardedRun`
      carrying its datum id + reward). Drives the ``rewarded`` denominator; its
      ``reward == 1.0`` entries drive the ``succeeded`` numerator.
    * ``expired_datum_by_run`` -- one entry per *expired, unrewarded* run (its
      datum id): a run that either kept a lingering in-flight op (the generation
      itself was too slow to finish within the lease) or ran past the
      total-duration expiration threshold. Because the "expired and unrewarded"
      filter is applied when the map is built, every entry here is an expiration
      (an `expired`, not `lost`, run); ongoing runs (still within their lease)
      and lost/abandoned runs never appear.

    Both maps span the datum's entire history, so this function just tallies.
    """
    expired: Dict[str, int] = {d: 0 for d in datum_pool}
    rewarded: Dict[str, int] = {d: 0 for d in datum_pool}
    succeeded: Dict[str, int] = {d: 0 for d in datum_pool}
    # Rewarded terminal attempts (denominator), plus their successes.
    for run in rewarded_by_run.values():
        if run.datum_id not in rewarded:
            continue
        rewarded[run.datum_id] += 1
        if run.reward == 1.0:
            succeeded[run.datum_id] += 1
    # Expirations: expired, unrewarded runs (lingering in-flight op or total
    # duration past the expiration threshold).
    for datum_id in expired_datum_by_run.values():
        if datum_id not in expired:
            continue
        expired[datum_id] += 1
    return {
        d: TerminalStats(
            expired=expired[d],
            rewarded=rewarded[d],
            succeeded=succeeded[d],
        )
        for d in datum_pool
    }


def quarantined_datums(
    datum_pool: List[str],
    rewarded_by_run: Dict[str, RewardedRun],
    expired_datum_by_run: Dict[str, str],
    *,
    min_samples: float,
    max_expire_rate: Optional[float] = None,
    max_succeed_ratio: Optional[float] = None,
) -> Set[str]:
    """Datums to exclude from dispense, per the enabled quarantine filters.

    Built on :func:`terminal_stats`. Both filters share the same
    ``terminal = expired + rewarded`` denominator and the ``min_samples`` gate;
    a datum is quarantined when it has at least ``min_samples`` terminal
    attempts and *either* enabled filter fires:

    * expiration (``max_expire_rate``): expire rate ``expired / terminal >=
      max_expire_rate`` -- it genuinely keeps expiring.
    * too-easy (``max_succeed_ratio``): success ratio ``succeeded / terminal >
      max_succeed_ratio`` -- it is solved too reliably.

    Passing ``None`` for a threshold disables that filter. Counts span the
    datum's entire history (no recency window): once a datum crosses a threshold
    with enough samples it stays quarantined, since it receives no new runs to
    move the metric.
    """
    stats = terminal_stats(datum_pool, rewarded_by_run, expired_datum_by_run)
    excluded: Set[str] = set()
    for d, s in stats.items():
        # Both filters share the terminal denominator (expired + rewarded) and
        # the same min-samples gate, so in-flight runs are the only ones
        # excluded from either metric.
        terminal = s.expired + s.rewarded
        if terminal < min_samples:
            continue
        if max_expire_rate is not None and s.expired / terminal >= max_expire_rate:
            excluded.add(d)
            continue
        if max_succeed_ratio is not None and s.succeeded / terminal > max_succeed_ratio:
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
