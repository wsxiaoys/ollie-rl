import uuid
import logging
from typing import List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from ollie_rl.types import ChatCompletionRequest
from ollie_rl.trainer.types import (
    Trainer,
    TrainerFactory,
    Example,
    Sample,
    TrainOp,
    SampleOp,
    StateStore,
)
from ollie_rl.trainer import factory

logger = logging.getLogger(__name__)


class FakeTrainOp(TrainOp):
    async def wait(self) -> None:
        return None

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
        pass

    @property
    def policy_generation(self) -> int:
        return 0

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
        return FakeTrainOp()


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
