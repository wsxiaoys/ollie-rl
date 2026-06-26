from pydantic import BaseModel
from typing import List, Optional
from openai.types.chat import (
    ChatCompletionFunctionTool,
    ChatCompletionMessage,
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = None
    tools: Optional[List[ChatCompletionFunctionTool]] = None


class CreateTunerRequest(BaseModel):
    name: str
    recipe: str
    datum_ids: List[str]
    hparams: Optional[dict] = None


class PutRewardRequest(BaseModel):
    reward: float


class RolloutRun(BaseModel):
    id: str
    reward: float
    advantage: float


class Rollout(BaseModel):
    runs: List[RolloutRun]
