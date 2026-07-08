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
    # Optional cached sample-time data. Same layout convention as Example.
    tokens: Optional[List[int]] = None
    logprobs: Optional[List[float]] = None


# Reserved ref: the step produced no usable frozen checkpoint (e.g. gemini's
# TunedModelCheckpoint is null), so "sampling this checkpoint" means replay the
# live policy rather than address frozen weights.
LIVE_POLICY_CHECKPOINT = "live"


class Checkpoint(BaseModel):
    """A frozen snapshot produced by one train step, or the live-policy
    sentinel when the backend emits no usable handle.

    `ref` is the backend's opaque handle, or `LIVE_POLICY_CHECKPOINT` when the
    step produced none. `policy_generation` is the attribution tag eval buckets
    by.
    """

    ref: str = LIVE_POLICY_CHECKPOINT  # backend handle, or live-policy sentinel
    policy_generation: int  # monotonic; ordering / display / bucketing


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


class TrainOp(Op[Optional[Checkpoint]]):
    """wait() returns the Checkpoint the step produced, or None if the backend
    does not emit a generation for the completed step."""

    pass


class SampleOp(Op[Sample]):
    pass


class Sampler(ABC):
    """Sampling surface scoped to one policy snapshot -- a single frozen
    checkpoint, or the live policy.

    Owns ``SampleOp`` submission plus ``restore_state`` re-attach, exactly like
    :meth:`Trainer.sample`, so a caller can treat a checkpoint-scoped sampler
    and the live trainer uniformly. A frozen checkpoint is a distinct, reusable
    resource (the backend loads its weights into a serving slot), so modeling
    it as an object lets the service create it once per checkpoint and cache it
    -- amortizing the load across the many eval rollouts that target one
    checkpoint.
    """

    @abstractmethod
    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ) -> SampleOp: ...


class Trainer(Sampler):
    """
    A single, live training job against some backend.

    The Trainer owns its own persistence cadence via its StateStore.

    A ``Trainer`` *is* a :class:`Sampler`: its :meth:`sample` already satisfies
    the sampler surface, so a live trainer doubles as the live-policy sampler
    (see :meth:`create_sampler`) with no wrapper on the hot path.
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
    async def train_step(
        self,
        examples: List[Example],
        *,
        sampler_promotion_every: int = 1,
    ) -> TrainOp:
        """Run one train step over ``examples``.

        ``sampler_promotion_every`` is the sampler-promotion cadence (from the
        recipe): a fresh sampler snapshot is published only every N steps, and
        on the other steps backends that support it skip the weight sync to the
        sampler/serving path. Backends compute the promote/skip decision from
        their own (post-increment) step counter.
        """
        ...

    async def pending_train_op(self) -> Optional[TrainOp]:
        """The in-flight train op, if one is running, else None.

        Serves two callers: a truthiness check ("is this trainer training?")
        and reconcile -- the service awaits the returned op to drive it to
        completion (e.g. after a restart). Backends that train inline
        (Tinker, Fake) have no reattachable op and return None.

        Must be cheap / non-I/O: it only reconstructs a handle, it does not
        poll the backend.
        """
        return None

    @abstractmethod
    async def create_sampler(self, checkpoint: "Checkpoint") -> "Sampler":
        """Return a :class:`Sampler` scoped to ``checkpoint``.

        Callers resolve the ``LIVE_POLICY_CHECKPOINT`` sentinel to the live
        trainer *before* calling this, so ``create_sampler`` is only invoked
        for a checkpoint with a real backend ref. A backend that publishes a
        real, reusable checkpoint handle (Tinker's sampler ``path``) loads it
        into frozen weights -- pinning eval to exactly that snapshot and making
        it immune to later sampler promotions. Backends that never publish a
        real ref (fake, gemini) are never reached and simply return ``self``
        (the live policy, since a ``Trainer`` is already a :class:`Sampler`).
        """
        ...


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
