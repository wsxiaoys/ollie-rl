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
    max_context_window: int = 40_000


# ---- Named recipe instances --------------------------------------------

GRPO_16x32 = Recipe(
    group_size=16,
    num_groups_per_batch=32,
)

GRPO_4x8 = Recipe(
    group_size=4,
    num_groups_per_batch=8,
)
