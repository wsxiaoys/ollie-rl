from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionTool,
    ChatCompletionMessageParam,
)
from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from ollie_rl.cookbook import Cookbook, Recipe, RecipeInput


class TunerStatus(str, Enum):
    """Dynamic lifecycle status of a tuner's backend training resource.

    The value is deliberately not persisted: each trainer maps its backend's
    authoritative state into this small service-level lifecycle.
    """

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    CANCELLED = "CANCELLED"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[SerializeAsAny[ChatCompletionMessageParam]]
    max_tokens: Optional[int] = None
    tools: Optional[List[ChatCompletionFunctionTool]] = None
    stream: Optional[bool] = None

    @field_validator("messages", mode="wrap")
    @classmethod
    def preserve_extra_content(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> List[ChatCompletionMessageParam]:
        """Retain OpenAI-compatible vendor extensions after SDK validation."""
        messages = handler(value)
        if not isinstance(value, list):
            return messages

        for raw_message, message in zip(value, messages):
            if not isinstance(raw_message, dict):
                continue

            raw_message_dict = cast(Dict[str, Any], raw_message)
            message_dict = cast(Dict[str, Any], message)
            if "extra_content" in raw_message_dict:
                message_dict["extra_content"] = raw_message_dict["extra_content"]

            raw_tool_calls = raw_message_dict.get("tool_calls")
            tool_calls = message_dict.get("tool_calls")
            if not isinstance(raw_tool_calls, list) or tool_calls is None:
                continue

            validated_tool_calls = list(tool_calls)
            for raw_tool_call, tool_call in zip(raw_tool_calls, validated_tool_calls):
                if not isinstance(raw_tool_call, dict):
                    continue
                raw_tool_call_dict = cast(Dict[str, Any], raw_tool_call)
                if "extra_content" in raw_tool_call_dict:
                    cast(Dict[str, Any], tool_call)["extra_content"] = (
                        raw_tool_call_dict["extra_content"]
                    )
            message_dict["tool_calls"] = validated_tool_calls

        return messages

    @model_validator(mode="after")
    def materialize_tool_calls(self) -> "ChatCompletionRequest":
        """Make OpenAI's iterable tool calls safe for repeated consumption."""
        for message in self.messages:
            message_dict = cast(Dict[str, Any], message)
            tool_calls = message_dict.get("tool_calls")
            if tool_calls is not None:
                message_dict["tool_calls"] = list(tool_calls)
        return self


class CreateTunerRequest(BaseModel):
    name: str
    recipe: str | RecipeInput
    trainer: str
    # Datums to train on (dispensed into GRPO groups, rewarded, consumed by a
    # train_step). Must be non-empty.
    train_datum_ids: List[str]
    # Held-out datums scored per checkpoint but never trained on. Empty disables
    # eval. Must not overlap `train_datum_ids`.
    eval_datum_ids: List[str] = Field(default_factory=list)
    trainer_params: Optional[Dict[str, Any]] = None

    @field_validator("recipe")
    @classmethod
    def validate_recipe(cls, recipe: str | RecipeInput) -> str | RecipeInput:
        # Validate named recipe lookup at the HTTP boundary so malformed
        # creation requests consistently produce FastAPI's 422 response.
        Cookbook.resolve(recipe)
        return recipe


class CreateTunerResponse(BaseModel):
    tuner_id: str
    name: str
    recipe: Recipe
    status: TunerStatus


class PutRewardRequest(BaseModel):
    reward: float


class PutRewardResponse(BaseModel):
    run_id: str
    reward: float


class DispenseRun(BaseModel):
    run_id: str
    datum_id: str
    expires_at: datetime


class RolloutRun(BaseModel):
    id: str
    reward: float
    advantage: float


class Rollout(BaseModel):
    runs: List[RolloutRun]


class RunProgress(BaseModel):
    """Aggregate run counts across all datums for a tuner (dynamic)."""

    total: int
    in_flight: int  # reward is None, lease not expired
    # reward is None, lease expired, and a compute-waste signal fired -- either a
    # lingering in-flight op remains (the generation itself stalled past the
    # lease) or the run's total generation time crossed the expiration
    # threshold. Matches the `expired` run status; re-dispensable.
    expired: int
    # reward is None, lease expired, and *no* expiration signal fired (a
    # crashed/abandoned worker, or ops finished but no reward was ever posted).
    # Matches the `lost` run status; re-dispensable.
    lost: int
    rewarded: int  # reward set (any trained/rejected state)
    trained: int  # trained_count > 0
    rejected: int  # rejected_count > 0


class DatumProgress(BaseModel):
    """Per-datum group ('Rollout') coverage, trainer view (dynamic)."""

    datum_id: str
    # One-based position in the exact corpus-ordered group set required by the
    # next train step. None means this datum cannot enter the next batch yet.
    next_batch_position: Optional[int]
    consumable: int  # rewarded runs counting toward this group's group_size
    in_flight: int  # runs awaiting a reward (reward None, lease not expired)
    # All-time count of `expired` runs: expired, unrewarded runs that either
    # still have a lingering in-flight op or crossed the total-duration
    # expiration threshold. Lost runs are excluded, matching RunProgress.
    expired: int
    trained: int  # prior training exposure (scheduler's primary ordering)


class NextPick(BaseModel):
    """What pick_datum would dispense next, with reasoning (dynamic)."""

    datum_id: Optional[str]
    tier: Literal["incomplete", "fresh", "saturated", "budget", "none"]
    reason: str


class BatchProgress(BaseModel):
    """Readiness of the strict corpus-ordered groups for the next train_step."""

    groups_ready: int  # required next-batch groups already at group_size
    groups_in_progress: int  # required groups with consumable or in-flight runs


class DatumCoverage(BaseModel):
    """How the datum pool is being exercised."""

    in_progress: int  # datums with >=1 consumable or in-flight run
    trained: int  # datums with >=1 trained run
    never_trained: int  # datums with no trained run yet


class DatumPool(BaseModel):
    """Datum-pool coverage summary plus per-datum detail."""

    coverage: DatumCoverage
    items: List[DatumProgress]  # per-datum detail (non-empty only)


class TrainingProgress(BaseModel):
    """Dynamic snapshot. Thresholds live on the recipe, not here."""

    batch: BatchProgress
    runs: RunProgress
    data: DatumPool
    next_pick: NextPick


class EvalDatumProgress(BaseModel):
    """Held-out status for a single eval datum against the latest checkpoint.

    Both counts are scoped to the newest checkpoint the eval tier is currently
    targeting (`EvalProgress.latest_checkpoint_generation`) -- mirroring how a
    training datum's `consumable` counts toward the *current* group rather than
    all-time -- so the numbers describe progress toward scoring that checkpoint.
    """

    datum_id: str
    in_flight: int  # eval runs pending a reward (lease unexpired)
    completed: int  # eval runs that have been rewarded


class CheckpointInfo(BaseModel):
    """Persisted checkpoint metadata exposed in the eval progress view."""

    id: str
    ref: str
    policy_generation: int
    created_at: datetime


class EvalProgress(BaseModel):
    """Per-eval-datum held-out status rollup for a tuner.

    `latest_checkpoint_generation` is the newest checkpoint the eval tier is
    currently targeting (None before any checkpoint exists); every per-datum
    count in `items` is scoped to it. `eval_group_size` is the target number of
    eval attempts per datum per checkpoint (the progress-bar denominator, from
    the recipe). `items` covers every registered eval datum, including ones with
    no runs against the latest checkpoint yet. `checkpoints` lists the persisted
    checkpoint records newest-first.
    """

    latest_checkpoint_generation: Optional[int]
    eval_group_size: int  # target attempts per datum per checkpoint
    total: int  # number of registered eval datums
    items: List[EvalDatumProgress]
    checkpoints: List[CheckpointInfo]


class TunerProgress(BaseModel):
    """Optional progress snapshots attached to a tuner detail response.

    Each field is populated only when its kind was requested via the tuner
    detail endpoint's `progress` query param (a comma-separated list of
    `train`/`eval`); un-requested kinds stay `None`.
    """

    train: Optional[TrainingProgress] = None
    eval: Optional[EvalProgress] = None


class GetTunerResponse(BaseModel):
    tuner_id: str
    name: str
    recipe: Recipe
    trainer: str
    status: TunerStatus
    policy_generation: int
    trainer_state: Optional[Any] = None
    # Optional progress snapshots, each populated only when requested via the
    # `progress` query param (comma-separated `train`/`eval`).
    progress: TunerProgress = Field(default_factory=TunerProgress)
    # True while an asynchronous train op is in flight. Backends that train
    # inline (or don't track ops) always report False. The completed step is
    # already available via `policy_generation`, so we only expose the
    # in-flight flag here.
    is_training: bool = False
    # Wall-clock execution time (seconds) of the most recent *completed* train
    # op, derived from its LRO `updateTime - createTime`. None when the backend
    # doesn't track op timing or no train op has completed yet.
    last_train_op_duration_seconds: Optional[float] = None


class TunerItem(BaseModel):
    tuner_id: str
    name: str
    trainer: str
    status: TunerStatus
    policy_generation: int


class ListTunersResponse(BaseModel):
    tuners: List[TunerItem]


class ListDatumsResponse(BaseModel):
    """The datum-id pool registered for a tuner (for filter dropdowns).

    Scoped to a single split when the request passes `split=train`/`eval`,
    otherwise the full pool.
    """

    datum_ids: List[str]


# Lifecycle status of a run, derived from its bookkeeping columns (plus, for
# the expired/lost split, whether a lingering in-flight op remains or the run's
# total duration crossed the expiration threshold). The labels are mutually
# exclusive and assigned by priority in `TunerService`:
# trained > rejected > length > content_filter > rewarded > in_flight > expired
# > lost.
#
# `length` means at least one recorded completion exceeded the recipe's
# `max_context_window` (prompt + completion + reasoning tokens) and was converted
# to a cleared length sample. `content_filter` means at least one completion was
# content-filtered (a malformed model output the server terminated with the
# recipe's `content_filter_penalty`). `expired` and `lost` both mean "reward is
# None and
# the lease has passed"; they differ on *why*. `expired` means a compute-waste
# signal fired: the run either still has a lingering `InFlightChatCompletionModel`
# row (the generation itself stalled past the lease) or its summed completion
# duration crossed the expiration threshold. `lost` is the residual case
# (crashed/abandoned worker, or ops all finished but no reward was ever posted).
# Both are surfaced as their own aggregate counts (`RunProgress.expired` /
# `RunProgress.lost`) and per-datum (`DatumProgress.expired`).
RunStatus = Literal[
    "in_flight",
    "expired",
    "lost",
    "length",
    "content_filter",
    "rewarded",
    "trained",
    "rejected",
]


class RunItem(BaseModel):
    """Summary of a single run (one attempt at a datum) under a tuner."""

    run_id: str
    datum_id: str
    status: RunStatus
    reward: Optional[float]
    # The run's policy generation, derived as the max `policy_generation`
    # across its chat completions. `None` when the run has no recorded
    # completions yet. Lets clients bucket rewards by generation (e.g. a
    # reward-distribution view) without an extra per-run fetch.
    policy_generation: Optional[int]
    trained_count: int
    rejected_count: int
    completion_count: int
    # Sum of generation latency (milliseconds) across the run's chat
    # completions. `None` when the run has no recorded completions (or none
    # carry a duration, e.g. only legacy rows).
    duration_ms_total: Optional[int] = None
    # Maximum context-window length observed across the run's chat completions,
    # measured as prompt + completion + reasoning tokens. `None` when no
    # completion reports token usage.
    context_window_tokens_max: Optional[int] = None
    created_at: datetime
    expires_at: datetime


class ListRunsResponse(BaseModel):
    runs: List[RunItem]
    # Opaque forward cursor for the next page (cursor-based pagination). Pass it
    # back as the `cursor` query param to fetch the runs immediately after the
    # last item in `runs`. `None` when there are no more runs (or the caller
    # requested every run unbounded).
    next_cursor: Optional[str] = None


class InFlightChatCompletionItem(BaseModel):
    """A resumable backend chat-completion operation tracked for a tuner."""

    run_id: str
    datum_id: str
    request_hash: str
    kind: Literal["train", "eval"]
    checkpoint_generation: Optional[int] = None
    created_at: datetime
    run_expires_at: datetime
    lease_expired: bool
    recorded_completion_count: int


class ListInFlightChatCompletionsResponse(BaseModel):
    items: List[InFlightChatCompletionItem]
    total: int
    active_lease_count: int
    past_lease_count: int
    oldest_created_at: Optional[datetime] = None
    next_cursor: Optional[str] = None


class GenerationRewardStats(BaseModel):
    """Reward summary for all rewarded runs at a single policy generation."""

    generation: int
    count: int
    mean: float
    std: float  # population standard deviation
    min: float
    max: float
    # Per-bin reward counts, aligned to the response's shared `bin_edges`.
    bins: List[int]


class RewardDistributionResponse(BaseModel):
    """Reward distribution bucketed by policy generation, computed server-side.

    Replaces the former client-side aggregation over an *unbounded* run fetch:
    the dashboard used to download every run just to bucket rewards by
    generation. The server now reads only `(reward, max policy_generation)` per
    rewarded run -- two scalars, no JSON blobs and no full run transfer -- and
    returns the finished histogram. A run contributes only when it has both a
    reward and at least one recorded completion (so a derived generation).
    """

    # Per-generation rows, ascending by generation.
    rows: List[GenerationRewardStats]
    # Shared lower edges of each histogram bin (length matches the bin count).
    bin_edges: List[float]
    bin_width: float
    # Global reward range across all contributing rewarded runs.
    reward_min: float
    reward_max: float
    # Total rewarded runs that contributed (reward + generation present).
    total: int


class ChatCompletionItem(BaseModel):
    """A single recorded LLM request/response inside a run."""

    id: str
    policy_generation: int
    created_at: datetime
    # Wall-clock generation latency in milliseconds. `None` only for legacy
    # rows written before this column existed.
    duration_ms: Optional[int] = None
    request: ChatCompletionRequest
    response: ChatCompletion


class RunDetailResponse(BaseModel):
    run: RunItem
    completions: List[ChatCompletionItem]


class ChatCompletionDetailResponse(BaseModel):
    """Full detail of a single recorded chat completion for inspection.

    Extends the summary fields of `ChatCompletionItem` with the owning
    tuner/run/datum identifiers and the optional sample-time tensors
    (`tokens`/`logprobs`) so a single completion can be inspected in
    isolation.
    """

    id: str
    tuner_id: str
    run_id: str
    datum_id: str
    policy_generation: int
    created_at: datetime
    # Wall-clock generation latency in milliseconds. `None` only for legacy
    # rows written before this column existed.
    duration_ms: Optional[int] = None
    request: ChatCompletionRequest
    response: ChatCompletion
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
