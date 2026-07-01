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
    expired: int  # reward is None, lease expired (re-dispensable)
    rewarded: int  # reward set (any trained/rejected state)
    consumable: int  # rewarded & trained<=0 & rejected<=0 & not stale
    trained: int  # trained_count > 0
    rejected: int  # rejected_count > 0


class DatumProgress(BaseModel):
    """Per-datum group ('Rollout') coverage, trainer view (dynamic)."""

    datum_id: str
    consumable: int  # rewarded runs counting toward this group's group_size
    in_flight: int  # runs awaiting a reward (reward None, lease not expired)
    trained: int  # prior training exposure (fresh-tier tie-break)


class NextPick(BaseModel):
    """What _pick_datum would dispense next, with reasoning (dynamic)."""

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


# Lifecycle status of a run, derived from its bookkeeping columns. The labels
# are mutually exclusive and assigned by priority in `TunerService`:
# trained > rejected > rewarded > in_flight > expired.
RunStatus = Literal["in_flight", "expired", "rewarded", "trained", "rejected"]


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
    created_at: datetime
    expires_at: datetime


class ListRunsResponse(BaseModel):
    runs: List[RunItem]


class ChatCompletionItem(BaseModel):
    """A single recorded LLM request/response inside a run."""

    id: str
    policy_generation: int
    created_at: datetime
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
    request: ChatCompletionRequest
    response: ChatCompletion
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None
