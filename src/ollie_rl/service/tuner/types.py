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
