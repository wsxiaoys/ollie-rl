from datetime import datetime

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai.types.chat import (
    ChatCompletionFunctionTool,
    ChatCompletionMessageParam,
)


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


class GetTunerResponse(BaseModel):
    tuner_id: str
    name: str
    recipe: str
    trainer: str
    policy_generation: int
    trainer_state: Optional[Any] = None


class TunerItem(BaseModel):
    tuner_id: str
    name: str
    recipe: str
    trainer: str
    policy_generation: int


class ListTunersResponse(BaseModel):
    tuners: List[TunerItem]
