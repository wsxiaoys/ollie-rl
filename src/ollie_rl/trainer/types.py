from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Protocol, TypeVar
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

from ollie_rl.types import ChatCompletionRequest

T = TypeVar("T")


class Example(BaseModel):
    chat_completion_id: str
    advantage: float
    policy_generation: int
    # Optional cached sample-time data needed by trainers that train on
    # raw tokens/logprobs (e.g. Tinker). Layout convention:
    #   tokens   : full sequence (prompt + completion)
    #   logprobs : per-completion-token logprobs
    # Hence prompt_len = len(tokens) - len(logprobs). Backends that do not
    # need these (e.g. gemini_msrl, fake) ignore them.
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None


class Sample(BaseModel):
    completion: ChatCompletion
    policy_generation: int
    malformed: bool = False
    # Optional cached sample-time data. Same layout convention as Example.
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None


class StateStore(Protocol):
    """
    Bi-directional opaque-blob persistence handle owned by a Trainer.
    Read-your-writes semantics required.
    """

    async def load(self) -> Optional[str]: ...

    async def save(self, trainer_state: str) -> None: ...


class Op(ABC, Generic[T]):
    @abstractmethod
    async def wait(self) -> T: ...

    @abstractmethod
    async def peek(self) -> bool: ...

    def save_state(self) -> Optional[str]:
        """Serialize this op's resume state (e.g. an LRO op_name).

        Persist the returned string and later pass it back to the *same public
        entry point* that produced the op (e.g.
        ``Trainer.sample(request, restore_state=...)``) to reconstruct an
        equivalent op that keeps waiting on the same backend operation.
        ``None`` = this op is not resumable and must always be produced fresh.
        """
        return None


class TrainOp(Op[None]):
    pass


class SampleOp(Op[Sample]):
    pass


class Trainer(ABC):
    """
    A single, live training job against some backend.

    The Trainer owns its own persistence cadence via its StateStore.
    """

    @property
    @abstractmethod
    def policy_generation(self) -> int: ...

    @abstractmethod
    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ) -> SampleOp:
        """Submit a fresh sample op, or -- when ``restore_state`` is given --
        re-attach to that already-submitted backend op instead of submitting a
        new one. Backends that don't support resumption ignore it."""
        ...

    @abstractmethod
    async def train_step(self, examples: List[Example]) -> TrainOp: ...

    async def is_training(self) -> bool:
        """Whether an asynchronous train op is currently in flight.

        Default is ``False`` so backends that train inline (e.g. Tinker,
        Fake) need not implement it. Backends with an asynchronous, pollable
        train op should override this to surface an in-flight indicator to
        the dashboard. May poll the backend to confirm the op's terminal
        state, so callers should treat it as potentially I/O-bound.
        """
        return False


class TrainerFactory(ABC):
    """
    Async factory that bootstraps or restores a Trainer against a StateStore.

    Has no knowledge of recipes or scheduling.
    """

    @abstractmethod
    async def create(
        self,
        name: str,
        state_store: StateStore,
        trainer_params: Optional[dict] = None,
    ) -> Trainer: ...

    @abstractmethod
    async def restore(
        self,
        name: str,
        state_store: StateStore,
    ) -> Trainer: ...
