from abc import ABC, abstractmethod
from typing import List, Generic, Optional, Protocol, TypeVar
from dataclasses import dataclass
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from ollie_rl.types import ChatCompletionRequest


T = TypeVar("T")


class Example(BaseModel):
    chat_completion_id: str
    advantage: float


class Sample(BaseModel):
    completion: ChatCompletion
    policy_generation: str


@dataclass
class DispenseContext:
    is_training: bool
    datum_pool: List[str]


@dataclass
class RunAssignment:
    run_id: str
    datum_id: str


class StateStore(Protocol):
    """
    Bi-directional, opaque-blob persistence handle owned by a Tuner.

    The Tuner controls *when* to persist its state by calling `save`.
    On startup, the Tuner decides whether it is bootstrapping or resuming
    by inspecting the result of `load`:
      - `None` means no prior state exists (fresh creation).
      - A non-None string is the most recent successfully-saved blob.

    Implementations must provide read-your-writes semantics: a `load`
    following a successful `save` must return that saved value (or a newer
    one if another save happened in between).
    """

    async def load(self) -> Optional[str]:
        """Return the last saved state blob, or None if none exists yet."""
        ...

    async def save(self, state: str) -> None:
        """Persist the given opaque state blob durably."""
        ...


class Op(ABC, Generic[T]):
    """
    Represents an active, asynchronous operation.
    """

    @abstractmethod
    async def wait(self) -> T:
        """Block and wait for the operation to complete."""
        pass

    @abstractmethod
    async def peek(self) -> bool:
        """Return True iff the op has reached a terminal state. Cheap;
        OK to call on every request."""
        pass


class TrainOp(Op[None]):
    """
    Represents an active, asynchronous training step operation.
    """

    pass


class SampleOp(Op[Sample]):
    """
    Represents an active, asynchronous sampling operation.
    """

    pass


class Tuner(ABC):
    """
    Abstract base class representing an active RL tuner/training job.

    A Tuner owns its own persistence cadence: it receives a `StateStore`
    at construction time (via `Recipe.open`) and calls `state_store.save`
    whenever its internal state has meaningfully changed and should be
    durable.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Return the identifier string of the recipe template."""
        pass

    @abstractmethod
    async def sample(self, request: ChatCompletionRequest) -> SampleOp:
        """
        Initiate a chat completion request.
        Returns a SamplingOp immediately after the request is received by the backend.
        """
        pass

    @abstractmethod
    async def train_step(self, examples: List[Example]) -> TrainOp:
        """
        Initiate a single RL training step.
        Returns a TrainingOp immediately after the request is received by the backend.
        """
        pass


class Recipe(ABC):
    """
    Abstract base class for all RL recipes.

    A Recipe is a factory that opens a Tuner against a backing `StateStore`.
    The recipe decides, based on the contents of the store, whether to
    bootstrap a fresh tuner or resume an existing one. Once initialized,
    the Tuner is responsible for calling `state_store.save` at appropriate
    moments in its own lifecycle.
    """

    @abstractmethod
    async def open(self, name: str, state_store: StateStore) -> Tuner:
        """
        Open a Tuner instance backed by the given state store.

        If `await state_store.load()` returns None, the recipe should
        bootstrap a fresh tuner and persist its initial state via
        `state_store.save` before returning. Otherwise, it should restore
        a tuner from the loaded blob.
        """
        pass
