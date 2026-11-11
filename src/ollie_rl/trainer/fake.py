import json
import uuid
import logging
from typing import List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from ollie_rl.types import ChatCompletionRequest, TunerStatus
from ollie_rl.trainer.types import (
    LIVE_POLICY_CHECKPOINT,
    Checkpoint,
    Trainer,
    TrainerFactory,
    Example,
    Sample,
    Sampler,
    TrainOp,
    SampleOp,
    StateStore,
)
from ollie_rl.trainer import factory

logger = logging.getLogger(__name__)


class FakeTrainOp(TrainOp):
    def __init__(self, policy_generation: int):
        self._policy_generation = policy_generation

    async def wait(self) -> Optional[Checkpoint]:
        # No usable frozen checkpoint: tag the completed step with the
        # live-policy sentinel so eval (if any) samples the live policy.
        return Checkpoint(
            ref=LIVE_POLICY_CHECKPOINT,
            policy_generation=self._policy_generation,
        )

    async def peek(self) -> bool:
        return True


class FakeSampleOp(SampleOp):
    def __init__(self, value: Sample):
        self.value = value

    async def wait(self) -> Sample:
        return self.value

    async def peek(self) -> bool:
        return True


class FakeTrainer(Trainer):
    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        train_step: int = 0,
    ):
        # Monotonic step counter advanced on each `train_step`, so the fake
        # backend emits a checkpoint per completed step (with the live-policy
        # sentinel ref, since it has no frozen weights to address).
        self._train_step = train_step
        # Optional: when present, every step is mirrored into the DB state blob
        # so `policy_generation` survives a process restart.
        self._state_store = state_store

    @property
    def policy_generation(self) -> int:
        return self._train_step

    async def _persist(self) -> None:
        """Write the (trivial) fake trainer state blob.

        A non-null `trainer_state` is the service's "backend is provisioned"
        marker: `list_tuners` and `_train_all_pending` filter on it, and
        `_materialize` refuses to restore without it. Persisting here keeps
        fake tuners visible, trainable and restart-surviving instead of
        looking like unprovisioned zombie rows.
        """
        if self._state_store is None:
            return
        await self._state_store.save(
            json.dumps(
                {
                    "trainer": "fake",
                    "version": 1,
                    "train_step": self._train_step,
                }
            )
        )

    async def get_status(self) -> TunerStatus:
        return TunerStatus.IN_PROGRESS

    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ) -> SampleOp:
        # FakeTrainer has no long-running op to re-attach to, so
        # `restore_state` is accepted (for interface parity) and ignored.
        completion_id = f"cmpl_{uuid.uuid4().hex}"
        completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="This is a fake completion response from ollie-rl fake trainer.",
                    ),
                )
            ],
            created=1234567890,
            model="fake-model",
            object="chat.completion",
        )
        sample = Sample(
            completion=completion,
            policy_generation=0,
        )
        return FakeSampleOp(sample)

    async def train_step(
        self,
        examples: List[Example],
        *,
        sampler_promotion_every: int = 1,
    ) -> TrainOp:
        logger.info(f"FakeTrainer training step with {len(examples)} examples.")
        self._train_step += 1
        await self._persist()
        return FakeTrainOp(self._train_step)

    async def create_sampler(self, checkpoint: Checkpoint) -> Sampler:
        # FakeTrainer only ever emits the live-policy sentinel ref, so the
        # service never routes a frozen checkpoint here. Return the live policy
        # (self) to satisfy the Sampler contract.
        return self


class FakeTrainerFactory(TrainerFactory):
    async def create(
        self,
        name: str,
        state_store: StateStore,
        trainer_params: Optional[dict] = None,
    ) -> Trainer:
        trainer = FakeTrainer(state_store=state_store)
        await trainer._persist()
        return trainer

    async def restore(
        self,
        name: str,
        state_store: StateStore,
    ) -> Trainer:
        trainer = FakeTrainer(
            state_store=state_store,
            train_step=_load_train_step(await state_store.load()),
        )
        # Re-save so a row left with a null blob (pre-fix tuners) is repaired
        # on first materialize rather than staying permanently invisible.
        await trainer._persist()
        return trainer


def _load_train_step(raw: Optional[str]) -> int:
    """Best-effort read of the persisted step counter; 0 when absent/garbled.

    A corrupt or legacy blob must not make the tuner unrestorable -- the step
    counter is cosmetic for the fake backend (it only advances checkpoint
    numbering), so falling back to 0 is safe.
    """
    if not raw:
        return 0
    try:
        value = json.loads(raw).get("train_step", 0)
    except AttributeError, TypeError, ValueError:
        return 0
    return value if isinstance(value, int) and value >= 0 else 0


factory.register("fake", FakeTrainerFactory())
