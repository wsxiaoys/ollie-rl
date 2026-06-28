from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Protocol, TypeVar
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

from ollie_rl.types import ChatCompletionRequest

T = TypeVar("T")


class Example(BaseModel):
    chat_completion_id: str
    advantage: float
    policy_generation: str


class Sample(BaseModel):
    completion: ChatCompletion
    policy_generation: str


class StateStore(Protocol):
    """
    Bi-directional opaque-blob persistence handle owned by a Trainer.
    Read-your-writes semantics required.
    """

    async def load(self) -> Optional[str]: ...

    async def save(self, state: str) -> None: ...


class Op(ABC, Generic[T]):
    @abstractmethod
    async def wait(self) -> T: ...

    @abstractmethod
    async def peek(self) -> bool: ...


class TrainOp(Op[None]):
    pass


class SampleOp(Op[Sample]):
    pass


class Trainer(ABC):
    """
    A single, live training job against some backend.

    The Trainer owns its own persistence cadence via its StateStore.
    """

    @abstractmethod
    async def sample(self, request: ChatCompletionRequest) -> SampleOp: ...

    @abstractmethod
    async def train_step(self, examples: List[Example]) -> TrainOp: ...

    @abstractmethod
    async def in_flight_train_op(self) -> Optional[TrainOp]: ...

    async def is_training(self) -> bool:
        op = await self.in_flight_train_op()
        if op:
            return not await op.peek()
        return False


class TrainerFactory(ABC):
    """
    Async factory that bootstraps or restores a Trainer against a StateStore.

    Has no knowledge of recipes or scheduling. May accept backend-specific
    bootstrap kwargs (base_model, adapter_size, …) via `open(...)`.
    """

    @abstractmethod
    async def open(
        self,
        name: str,
        state_store: StateStore,
        **bootstrap,
    ) -> Trainer: ...
