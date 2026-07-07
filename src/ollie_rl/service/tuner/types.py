"""Pure data types for the tuner dispense / quarantine algorithm.

These are plain pydantic value objects (no service/DB state) shared between the
data-access core (:mod:`ollie_rl.service.tuner.base`), the dispenser
(:mod:`ollie_rl.service.tuner.dispensing`), and the read-only query builders
(:mod:`ollie_rl.service.tuner.queries`). They live in their own leaf module so
those importers don't form a cycle.
"""

from typing import Dict

from pydantic import BaseModel


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

    * ``rewarded`` -- every run that earned a reward, *including* both
      length-limited and content-filtered (malformed) ones (consistent with
      batch/group accounting): ``length`` and ``content_filter`` are subsets of
      it. It is the shared denominator for both quarantine filters and the
      ``min_samples`` sample gate.
    * ``length`` -- rewarded runs with at least one length-limited completion; a
      subset of ``rewarded`` and part of the unhealthy-finish numerator.
    * ``succeeded`` -- rewarded runs with ``reward == 1.0`` (the success
      numerator); a subset of ``rewarded``.
    * ``content_filter`` -- rewarded runs whose completion was content-filtered
      (malformed); a subset of ``rewarded`` carrying the
      ``content_filter_penalty`` reward. Summed with ``length`` into the
      unhealthy-finish numerator; both are auto-penalty degenerate rollouts.
    """

    rewarded: int = 0
    length: int = 0
    succeeded: int = 0
    content_filter: int = 0


class SchedulerScores(BaseModel):
    """Per-datum scheduler tallies produced by ``scheduler_scores``.

    * ``score`` -- runs still *consumable* by a future train step from the
      scheduler's point of view (not yet trained, not requeued, and either
      rewarded or still pending/not expired).
    * ``trained`` -- accumulated prior training exposure, used for the
      fresh-tier tie-break.
    * ``rewarded`` -- the consumable runs whose reward has already returned (a
      subset of ``score``). This is what the two-phase probe gate in
      ``pick_datum`` checks against ``0.5 * group_size``: the same denominator
      the quarantine filters evaluate, so a datum's second half is only
      dispensed once enough probe results are back to clear quarantine.
    """

    score: Dict[str, int]
    trained: Dict[str, int]
    rewarded: Dict[str, int]
