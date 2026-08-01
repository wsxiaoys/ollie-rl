"""Pure data types for the tuner dispense scheduler.

These are plain pydantic value objects (no service/DB state) shared between the
dispenser and read-only query builders. They live in their own leaf module so
those importers don't form a cycle.
"""

from typing import Dict

from pydantic import BaseModel


class SchedulerScores(BaseModel):
    """Per-datum scheduler tallies produced by ``scheduler_scores``.

    * ``score`` -- runs still consumable by a future train step from the
      scheduler's point of view (not yet trained, not requeued, and either
      rewarded or still pending/not expired).
    * ``trained`` -- accumulated prior training exposure, used for the
      fresh-tier tie-break.
    """

    score: Dict[str, int]
    trained: Dict[str, int]
