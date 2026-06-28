from datetime import datetime

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from openai.types.chat import (
    ChatCompletionFunctionTool,
    ChatCompletionMessageParam,
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    max_tokens: Optional[int] = None
    tools: Optional[List[ChatCompletionFunctionTool]] = None


class CreateTunerRequest(BaseModel):
    name: str
    recipe: str = "grpo_16x32"
    trainer: str = "gemini_msrl"
    datum_ids: List[str]
    # Trainer-specific bootstrap overrides forwarded as **kwargs to
    # `TrainerFactory.open(...)`. Lets callers adopt an existing backend
    # resource (e.g. a pre-warmed Vertex tuning job) instead of provisioning
    # a fresh one. Schema is intentionally trainer-defined.
    bootstrap: Dict[str, Any] = Field(default_factory=dict)


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
