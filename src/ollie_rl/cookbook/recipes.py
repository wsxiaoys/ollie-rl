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


# ---- Named recipe instances --------------------------------------------

GRPO_16x32 = Recipe(
    group_size=16,
    num_groups_per_batch=32,
)
