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


class SetValueRequest(BaseModel):
    value: float


class RolloutRun(BaseModel):
    id: str
    datum_id: str
    reward: float
    advantage: float


class Rollout(BaseModel):
    datum_id: str
    runs: List[RolloutRun]
