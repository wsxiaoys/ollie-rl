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
    datum_ids: List[str]
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
    consumable: int  # rewarded & trained<=0 & rejected<=0 & not stale
    trained: int  # trained_count > 0
    rejected: int  # rejected_count > 0


class DatumProgress(BaseModel):
    """Per-datum group ('Rollout') coverage, trainer view (dynamic)."""

    datum_id: str
    consumable: int  # rewarded runs counting toward this group's group_size
    in_flight: int  # runs awaiting a reward (reward None, lease not expired)
    # All-time count of `expired` runs for this datum: expired, unrewarded
    # runs that either still have a lingering in-flight op (the generation itself
    # stalled past the lease) or crossed the total-duration expiration
    # threshold, regardless of policy generation. The headline "how flaky is
    # this datum" number, not
    # clipped by the recency window the quarantine rate uses. This is the
    # per-datum tally of runs the run-status `expired` label counts (`lost` runs
    # are excluded), matching the aggregate `RunProgress.expired`.
    expired: int
    trained: int  # prior training exposure (fresh-tier tie-break)
    # Recent expiration signal, matching the dispenser's quarantine logic:
    # the raw per-datum terminal-attempt counts within `max_off_policy_generation`
    # of the current generation. We pass the two components directly (rather than a
    # pre-computed rate) since together they are equivalent: the expire rate is
    # `expired / (expired + rewarded)` and the sample size is their sum. Use these
    # to pick a sensible `max_expire_rate` threshold for POST /runs.
    expired_within_policy_generation_cutoff: int  # expirations (numerator)
    rewarded_within_policy_generation_cutoff: int  # rewarded terminal attempts


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
# trained > rejected > rewarded > in_flight > expired > lost.
#
# `expired` and `lost` both mean "reward is None and the lease has passed"; they
# differ on *why*. `expired` means a compute-waste signal fired: the run either
# still has a lingering `InFlightChatCompletionModel` row (the generation itself
# stalled past the lease) or its summed completion duration crossed the
# expiration threshold (the same cases the dispenser quarantines on). `lost` is
# the residual case
# (crashed/abandoned worker, or ops all finished but no reward was ever posted).
# Both are surfaced as their own aggregate counts (`RunProgress.expired` /
# `RunProgress.lost`) and per-datum (`DatumProgress.expired`).
RunStatus = Literal["in_flight", "expired", "lost", "rewarded", "trained", "rejected"]


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
    created_at: datetime
    expires_at: datetime


class ListRunsResponse(BaseModel):
    runs: List[RunItem]
    # Opaque forward cursor for the next page (cursor-based pagination). Pass it
    # back as the `cursor` query param to fetch the runs immediately after the
    # last item in `runs`. `None` when there are no more runs (or the caller
    # requested every run unbounded).
    next_cursor: Optional[str] = None


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
