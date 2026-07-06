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

from ollie_rl.cookbook import Recipe
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow


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
# Expiration quarantine (opt-in dispenser filter)
#
# `dispense_run` accepts an optional `max_expire_rate` (from the `POST /runs`
# query param; omit/None to disable). When set, the dispenser *quarantines*
# datums that genuinely keep expiring, so compute isn't wasted on datums that
# never finish in time. The moving parts:
#
#   * `_recent_rewarded_datums` (DB, in tuner_service): per *rewarded* run
#     whose newest completion is within the recent generation window, its datum
#     id (a `run_id -> datum_id` map). Drives the *rewarded* side of the
#     denominator. The recency window is applied in SQL there
#     (`policy_generation >= min_generation`), so this map is already scoped to
#     the window (see the performance note there).
#   * `_expired_datums` (DB, in tuner_service): per *expired, unrewarded* run,
#     its datum id. A run is `expired` when it either still has a lingering
#     `InFlightChatCompletionModel` row (generation stalled) or its summed
#     completion `duration_ms` is at/past the expiration threshold
#     (`RUN_EXPIRE_DURATION_MS`). Both
#     signals honor the recency window, and the "expired and unrewarded" filter
#     lives in SQL, so ongoing runs (still within their lease) never appear.
#     Drives the *expiration* numerator.
#   * `expiration_stats` (pure): per-datum (expired, terminal) counts,
#     computed straight from the two already-windowed maps (each entry carries
#     its datum id, so no `RunModel` list and no recency filter are needed).
#   * `expiring_datums` (pure): turns those counts into the quarantine set.
#   * `dispense_run`: filters the flaky datums out of the candidate pool.
#
# Definitions and the reasoning behind them:
#
#   * Expired vs lost. An expired, unrewarded run (`reward is None and
#     expires_at <= now`) is classified `expired` (the status the dispenser
#     quarantines on) when it matches either compute-waste signal: (a) it still
#     has a lingering in-flight op -- an op that timed out / was cancelled and is
#     still progressing on the backend, i.e. the *generation itself* was too slow
#     to finish within the lease; or (b) its summed completion `duration_ms` is
#     at/past the expiration threshold (`RUN_EXPIRE_DURATION_MS`) -- it burned
#     real compute yet never finished. Both are the "never finishes in time"
#     waste quarantine targets.
#     Runs matching neither are `lost` and ignored: a crashed/abandoned worker,
#     or a run whose ops all completed but which was abandoned before a reward --
#     neither reflects a genuinely hard datum, so a flaky worker never poisons
#     it. The "expired and unrewarded" filter and each run's recency are resolved
#     when the two maps are built (the expired map is pre-filtered to expired,
#     unrewarded runs; the rewarded map holds the rewarded runs, which have no
#     lingering in-flight row since it is deleted on success).
#   * Rate = expired / terminal, where terminal = expired + rewarded.
#     In-flight runs are excluded (outcome unknown).
#   * Recency window == recovery mechanism. Only attempts within
#     `max_off_policy_generation` of the current policy generation are counted --
#     this applies to both expiration signals (in-flight op and duration past
#     the expiration threshold) as well as the rewarded denominator. That window is applied
#     upstream in SQL (the maps are built with `policy_generation >=
#     min_generation`, where `min_generation = generation -
#     max_off_policy_generation`), so by the time `expiration_stats` sees them
#     every entry is already in-window.
#     A quarantined datum receives no new runs, so a fixed all-time or
#     run-count window would trap it forever. Anchoring recency to the policy
#     generation fixes this: the generation clock advances as *other* datums
#     train, so a starved datum's stale expirations age out of the window, its
#     terminal count drops below `min_samples`, and it becomes dispensable
#     again (a "probe"). A successful probe lowers the rate; another expiry
#     re-quarantines it.
#   * Quarantine condition: terminal >= min_samples (dispense passes
#     `0.5 * recipe.group_size`, i.e. half a group's worth of terminal
#     attempts) AND rate >= max_expire_rate.
#
# Observability: `get_progress` reuses `expiration_stats` to populate
# `DatumProgress.expired_within_policy_generation_cutoff` /
# `rewarded_within_policy_generation_cutoff` (the raw numerator/denominator
# components, equivalent to the rate) so an operator can watch the numbers and
# pick a sensible `max_expire_rate`. It builds the same two maps
# (rewarded runs; expired-unrewarded runs -- lingering in-flight op or
# total duration past the expiration threshold) so the dashboard number matches
# what the dispenser would quarantine on.
# ---------------------------------------------------------------------------
def expiration_stats(
    datum_pool: List[str],
    rewarded_datum_by_run: Dict[str, str],
    expired_datum_by_run: Dict[str, str],
) -> Dict[str, Tuple[int, int]]:
    """Per-datum ``(expired, terminal)`` counts.

    Computed straight from two ``run_id -> datum_id`` maps, each already scoped
    to the recent-generation window when it was built (see the block comment
    above):

    * ``rewarded_datum_by_run`` -- one entry per rewarded run (its datum id).
      Drives the rewarded side of the denominator.
    * ``expired_datum_by_run`` -- one entry per *expired, unrewarded* run (its
      datum id): a run that either kept a lingering in-flight op (the generation
      itself was too slow to finish within the lease) or ran past the
      total-duration expiration threshold -- the compute-waste cases quarantine
      targets. Because the
      "expired and unrewarded" filter is applied when the map is built, every
      entry here is an expiration (an `expired`, not `lost`, run); ongoing runs
      (still within their lease) and lost/abandoned runs never appear. Drives
      both the numerator and its share of the denominator.

    The recency window is enforced upstream for both expiration signals (the map
    holds only runs within ``max_off_policy_generation`` of the current
    generation), so this function just tallies. ``terminal`` is the denominator
    (expired + rewarded); the caller derives the rate as ``expired / terminal``.
    """
    expired: Dict[str, int] = {d: 0 for d in datum_pool}
    terminal: Dict[str, int] = {d: 0 for d in datum_pool}
    # Rewarded terminal attempts (denominator only).
    for datum_id in rewarded_datum_by_run.values():
        if datum_id not in terminal:
            continue
        terminal[datum_id] += 1
    # Expirations: expired, unrewarded runs (lingering in-flight op or total
    # duration past the expiration threshold) -- numerator + their share of the
    # denominator.
    for datum_id in expired_datum_by_run.values():
        if datum_id not in expired:
            continue
        expired[datum_id] += 1
        terminal[datum_id] += 1
    return {d: (expired[d], terminal[d]) for d in datum_pool}


def expiring_datums(
    datum_pool: List[str],
    rewarded_datum_by_run: Dict[str, str],
    expired_datum_by_run: Dict[str, str],
    max_expire_rate: float,
    min_samples: float,
) -> Set[str]:
    """Datums to quarantine because they genuinely keep expiring.

    Built on :func:`expiration_stats`: a datum is quarantined when its recent
    window holds at least ``min_samples`` terminal attempts and the
    expiration rate within it is ``>= max_expire_rate``.

    Anchoring recency to the policy generation (the window applied upstream when
    the two maps are built) is what lets a quarantined datum recover: the global
    generation clock keeps advancing as *other* datums train, so a starved
    datum's stale expirations eventually age out of the window and it becomes
    dispensable again.
    """
    stats = expiration_stats(
        datum_pool,
        rewarded_datum_by_run,
        expired_datum_by_run,
    )
    flaky: Set[str] = set()
    for d, (expired, terminal) in stats.items():
        if terminal < min_samples:
            continue
        if expired / terminal >= max_expire_rate:
            flaky.add(d)
    return flaky


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
