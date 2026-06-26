from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from ollie_rl.types import ChatCompletionRequest


T = TypeVar("T")


class Example(BaseModel):
    chat_completion_id: str
    advantage: float


class Sample(BaseModel):
    completion: ChatCompletion
    step_id: str


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

    @abstractmethod
    async def save_state(self) -> str:
        """Save the current state of the tuner to an opaque string."""
        pass


class Recipe(ABC):
    """
    Abstract base class for all RL recipes.
    Acts as a factory to create or restore Tuner instances.
    """

    @abstractmethod
    async def create(self, name: str) -> Tuner:
        """
        Create and asynchronously initialize a new Tuner instance for a model.
        """
        pass

    @abstractmethod
    async def restore(self, state: str) -> Tuner:
        """
        Restore a Tuner instance from a saved state string.
        """
        pass
