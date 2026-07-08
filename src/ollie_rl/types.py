from datetime import datetime

from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionTool,
    ChatCompletionMessageParam,
)

from ollie_rl.cookbook import Recipe


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    max_tokens: Optional[int] = None
    tools: Optional[List[ChatCompletionFunctionTool]] = None
    stream: Optional[bool] = None


class CreateTunerRequest(BaseModel):
    name: str
    recipe: str
    trainer: str
    # Datums to train on (dispensed into GRPO groups, rewarded, consumed by a
    # train_step). Must be non-empty.
    train_datum_ids: List[str]
    # Held-out datums scored per checkpoint but never trained on nor counted
    # toward datum quarantine. Empty disables eval. Must not overlap
    # `train_datum_ids`.
    eval_datum_ids: List[str] = []
    trainer_params: Optional[Dict[str, Any]] = None


class CreateTunerResponse(BaseModel):
    tuner_id: str
    name: str
    recipe: str


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
    consumable: int  # rewarded runs counting toward this group's group_size
    in_flight: int  # runs awaiting a reward (reward None, lease not expired)
    # Per-datum terminal-attempt tallies over the datum's entire history (no
    # recency window). Both POST /runs quarantine filters share the full
    # `rewarded` denominator and the `min_samples` gate:
    #   * unhealthy-finish rate = (length + content_filter) / rewarded
    #   * success ratio         = succeeded / rewarded
    #
    # `expired`: all-time count of `expired` runs -- expired, unrewarded runs
    # that either still have a lingering in-flight op (the generation itself
    # stalled past the lease) or crossed the total-duration expiration
    # threshold. The `lost` runs are excluded, matching the aggregate
    # `RunProgress.expired`. This is observability-only and is not used for
    # quarantine.
    expired: int
    # `rewarded`: every run that earned a reward, *including* both length-limited
    # and content-filtered (malformed) runs (consistent with batch/group
    # accounting). It is the shared denominator for both quarantine filters and
    # the `min_samples` gate.
    # `length`: the subset of rewarded runs with a length finish reason; part of
    # the unhealthy-finish numerator.
    # `succeeded`: the `reward == 1.0` subset of `rewarded`.
    # `content_filter`: the subset of rewarded runs whose completion was
    # content-filtered (malformed); carries the `content_filter_penalty` reward.
    # Summed with `length` into the unhealthy-finish numerator (both are
    # auto-penalty degenerate rollouts).
    length: int
    rewarded: int
    succeeded: int
    content_filter: int
    trained: int  # prior training exposure (fresh-tier tie-break)


class NextPick(BaseModel):
    """What pick_datum would dispense next, with reasoning (dynamic)."""

    datum_id: Optional[str]
    tier: Literal["incomplete", "fresh", "saturated", "none"]
    reason: str


class BatchProgress(BaseModel):
    """Readiness toward the next train_step."""

    groups_ready: int  # groups already at group_size
    groups_in_progress: int  # not-yet-ready groups with >=1 consumable or in-flight run


class DatumCoverage(BaseModel):
    """How the datum pool is being exercised."""

    in_progress: int  # datums with >=1 consumable or in-flight run
    trained: int  # datums with >=1 trained run
    never_trained: int  # datums with no trained run yet
    excluded: int  # datums quarantined out of the dispensable pool


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


class GetTunerResponse(BaseModel):
    tuner_id: str
    name: str
    recipe: Recipe
    trainer: str
    policy_generation: int
    trainer_state: Optional[Any] = None
    progress: Optional[TrainingProgress] = None
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
    policy_generation: int


class ListTunersResponse(BaseModel):
    tuners: List[TunerItem]


class ListDatumsResponse(BaseModel):
    """The full datum-id pool registered for a tuner (for filter dropdowns)."""

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
# recipe's `content_filter_penalty`); like `length`, it is an unhealthy finish
# reason, and both are summed into the `max_unhealthy_finish_ratio` quarantine
# numerator. `expired` and `lost` both mean "reward is None and
# the lease has passed"; they differ on *why*. `expired` means a compute-waste
# signal fired: the run either still has a lingering `InFlightChatCompletionModel`
# row (the generation itself stalled past the lease) or its summed completion
# duration crossed the expiration threshold. `lost` is the residual case
# (crashed/abandoned worker, or ops all finished but no reward was ever posted).
# Both are surfaced as their own aggregate counts (`RunProgress.expired` /
# `RunProgress.lost`) and per-datum (`DatumProgress.expired`). Quarantine uses
# rewarded unhealthy-finish (length + content_filter) samples instead of expired
# runs.
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
