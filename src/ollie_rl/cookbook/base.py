from abc import ABC, abstractmethod
from typing import List
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from ollie_rl.types import ChatCompletionRequest


class Example(BaseModel):
    chat_completion_id: str
    advantage: float


class Tuner(ABC):
    """
    Abstract base class representing an active RL tuner/training job.
    """

    @property
    @abstractmethod
    def tuner_id(self) -> str:
        """Return the identifier string of the recipe template."""
        pass

    @property
    @abstractmethod
    def kind(self) -> str:
        """Return the identifier string of the recipe template."""
        pass

    @abstractmethod
    async def sample(self, request: ChatCompletionRequest) -> ChatCompletion:
        """
        Handle a chat completion request.
        Can perform standard inference or a multi-step rollout with environment feedback.
        """
        pass

    @abstractmethod
    async def train_step(self, examples: List[Example]) -> None:
        """Run a single RL training step (e.g., PPO/GRPO) using Tinker's TrainingClient."""
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
    async def create(self, tuner_id: str) -> Tuner:
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
