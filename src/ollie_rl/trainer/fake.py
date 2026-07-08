import uuid
import logging
from typing import List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from ollie_rl.types import ChatCompletionRequest
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
    def __init__(self):
        # Monotonic step counter advanced on each `train_step`, so the fake
        # backend emits a checkpoint per completed step (with the live-policy
        # sentinel ref, since it has no frozen weights to address).
        self._train_step = 0

    @property
    def policy_generation(self) -> int:
        return self._train_step

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

    async def train_step(self, examples: List[Example]) -> TrainOp:
        logger.info(f"FakeTrainer training step with {len(examples)} examples.")
        self._train_step += 1
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
        return FakeTrainer()

    async def restore(
        self,
        name: str,
        state_store: StateStore,
    ) -> Trainer:
        return FakeTrainer()


factory.register("fake", FakeTrainerFactory())
