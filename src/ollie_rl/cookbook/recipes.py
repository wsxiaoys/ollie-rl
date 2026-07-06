from typing import Literal
from pydantic import BaseModel

Scheduler = Literal["fifo_epoch", "random"]


class Recipe(BaseModel, frozen=True):
    """
    Declarative algorithm-level knobs the TunerService needs to schedule
    runs and form training batches. Pure data; knows nothing about backends.
    """

    # ---- Batch formation (GRPO-style grouping) --------------------------
    group_size: int = 16
    num_groups_per_batch: int = 32
    max_off_policy_generation: int = 4

    # ---- Behavior penalties ----
    content_filter_penalty: float = -1.0
    length_penalty: float = -1.0

    # ---- Context window guard ------------------------------------------
    # Hard cap on prompt + completion + reasoning tokens. Samples that
    # exceed this are overridden to the `length` finish reason and have their
    # response cleared.
    max_context_window: int = 60_000

    # ---- Datum quarantine ----------------------------------------------
    # Skip datums that no longer yield a useful learning signal. Both ratios
    # are measured over the datum's rewarded attempts (all-time, no recency
    # window) and only take effect once the datum has accumulated at least
    # `quarantine_min_samples` rewarded attempts. The default ratios (1.0) are
    # permissive: `max_succeed_ratio = 1.0` never fires (a success ratio can't
    # exceed 1.0) and `max_length_ratio = 1.0` fires only when *every* rewarded
    # attempt is length-limited.
    #
    #   * `max_length_ratio`: quarantine when the fraction of rewarded attempts
    #     with a length-limited completion is >= this value.
    #   * `max_succeed_ratio`: quarantine when the success ratio (reward == 1.0
    #     over rewarded attempts) is > this value (solved too reliably).
    max_length_ratio: float = 1.0
    max_succeed_ratio: float = 1.0
    # Number of rewarded attempts a datum must accumulate before the quarantine
    # verdict is trusted. Also sets the two-phase probe size: a started group is
    # dispensed up to `quarantine_min_samples` runs, then held for their rewards
    # before the rest is filled -- so a datum that will be quarantined wastes at
    # most this many rollouts.
    quarantine_min_samples: int = 4


# ---- Named recipe instances --------------------------------------------

GRPO_16x32 = Recipe(
    group_size=16,
    num_groups_per_batch=32,
)

GRPO_4x8 = Recipe(
    group_size=4,
    num_groups_per_batch=8,
)
