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

    # ---- Sampler promotion cadence -------------------------------------
    # Publish a fresh sampler snapshot every N train steps. On steps that
    # don't promote, backends that support it skip the (expensive) weight
    # sync to the sampler/serving path (e.g. Gemini's `skipWeightSync`,
    # Tinker's `save_weights_for_sampler`). 1 = promote on every step.
    sampler_promotion_every: int = 4

    # ---- Evaluation ----------------------------------------------------
    # Rollouts to dispense per eval datum per checkpoint. Averaging K attempts
    # smooths per-checkpoint eval variance the same way `group_size` does for a
    # training group. Only matters when the tuner has eval datums; 1 = a single
    # attempt per datum per checkpoint, 0 disables the eval dispense tier.
    eval_group_size: int = 4

    # ---- Behavior penalties ----
    content_filter_penalty: float = -1.0
    length_penalty: float = -10.0

    # ---- Context window guard ------------------------------------------
    # Hard cap on prompt + completion + reasoning tokens. Samples that
    # exceed this are overridden to the `length` finish reason and have their
    # response cleared.
    max_context_window: int = 60_000


# ---- Named recipe instances --------------------------------------------

GRPO_16x32 = Recipe(
    group_size=16,
    num_groups_per_batch=32,
)

GRPO_4x8 = Recipe(
    group_size=4,
    num_groups_per_batch=8,
)
