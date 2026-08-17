from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
RewardNormalizer = Literal["grpo", "centering"]


class Recipe(BaseModel, frozen=True):
    """
    Declarative algorithm-level knobs the TunerService needs to schedule
    runs and form training batches. Pure data; knows nothing about backends.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ---- Batch formation (GRPO-style grouping) --------------------------
    group_size: PositiveInt = 16
    num_groups_per_batch: PositiveInt = 32
    max_off_policy_generation: NonNegativeInt = 4
    reward_normalizer: RewardNormalizer = "grpo"

    # ---- Sampler promotion cadence -------------------------------------
    # Publish a fresh sampler snapshot every N train steps. On steps that
    # don't promote, backends that support it skip the (expensive) weight
    # sync to the sampler/serving path (e.g. Gemini's `skipWeightSync`,
    # Tinker's `save_weights_for_sampler`). 1 = promote on every step.
    sampler_promotion_every: PositiveInt = 4

    # ---- Evaluation ----------------------------------------------------
    # Evaluate every Nth persisted checkpoint, numbered from 1 in policy-
    # generation order. 1 evaluates every checkpoint; larger values reduce
    # eval overhead when each train step is small.
    eval_every_n_checkpoints: PositiveInt = 1

    # Rollouts to dispense per eval datum per eligible checkpoint. Averaging K
    # attempts smooths checkpoint eval variance the same way `group_size` does
    # for a training group. Only matters when the tuner has eval datums; 1 = a
    # single attempt per datum, 0 disables the eval dispense tier.
    eval_group_size: NonNegativeInt = 4

    # ---- Behavior penalties ----
    content_filter_penalty: FiniteFloat = -1.0
    length_penalty: FiniteFloat = -10.0

    # ---- Context window guard ------------------------------------------
    # Hard cap on prompt + completion + reasoning tokens. Samples that
    # exceed this are overridden to the `length` finish reason, causing
    # `length_penalty` to be applied, and have their response cleared.
    max_context_window: PositiveInt = 60_000


class RecipeInput(BaseModel):
    """Tuner-specific recipe fields layered over the Recipe defaults."""

    model_config = ConfigDict(extra="forbid", strict=True)

    group_size: Optional[PositiveInt] = None
    num_groups_per_batch: Optional[PositiveInt] = None
    max_off_policy_generation: Optional[NonNegativeInt] = None
    reward_normalizer: Optional[RewardNormalizer] = None
    sampler_promotion_every: Optional[PositiveInt] = None
    eval_every_n_checkpoints: Optional[PositiveInt] = None
    eval_group_size: Optional[NonNegativeInt] = None
    content_filter_penalty: Optional[FiniteFloat] = None
    length_penalty: Optional[FiniteFloat] = None
    max_context_window: Optional[PositiveInt] = None

    @model_validator(mode="after")
    def reject_explicit_null_overrides(self) -> "RecipeInput":
        null_fields = sorted(
            field for field in self.model_fields_set if getattr(self, field) is None
        )
        if null_fields:
            raise ValueError(f"Recipe overrides cannot be null: {null_fields}")
        return self


# ---- Named recipe instances --------------------------------------------

GRPO_16x32 = Recipe(
    group_size=16,
    num_groups_per_batch=32,
)

GRPO_4x8 = Recipe(
    group_size=4,
    num_groups_per_batch=8,
)
