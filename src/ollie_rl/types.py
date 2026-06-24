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
    model_id: str
    recipe: str
