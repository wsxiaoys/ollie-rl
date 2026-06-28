from __future__ import annotations
import asyncio
import logging
import os
import time
import uuid
from typing import List, Optional, Any

from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

import tinker
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, Message

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


class TinkerTrainerConfig(BaseModel):
    service_url: Optional[str] = None
    api_key: Optional[str] = None
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    kl_penalty_coef: float = 0.0
    loss_fn: str = "importance_sampling"
    max_steps_off_policy: int = 4
    sampler_promotion_every: int = 1


class TinkerTrainerState(BaseModel):
    # Tinker-side identity
    sampler_path: str  # latest saved sampler weights (also seeds restore)
    optimizer_path: Optional[str] = None  # latest full checkpoint (weights + opt state)

    # Async-RL bookkeeping
    train_step: int  # monotonically increasing; mirrors AsyncConfig step
    sampler_step: int  # train_step at which `sampler_path` was published

    # Backend config (frozen at create-time)
    base_model: str
    lora_rank: int
    learning_rate: float
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    kl_penalty_coef: float
    loss_fn: str  # e.g. "importance_sampling"


class TinkerTrainOp(TrainOp):
    async def wait(self) -> None:
        return None

    async def peek(self) -> bool:
        return True


class TinkerSampleOp(SampleOp):
    def __init__(self, task: asyncio.Task[Sample]):
        self._task = task

    async def wait(self) -> Sample:
        return await self._task

    async def peek(self) -> bool:
        return self._task.done()


class TinkerTrainer(Trainer):
    config: TinkerTrainerConfig
    service_client: tinker.ServiceClient
    state: TinkerTrainerState
    state_store: StateStore
    _sampling_client: tinker.SamplingClient
    _training_client: tinker.TrainingClient
    _train_op: Optional[TrainOp] = None

    def __init__(
        self,
        config: TinkerTrainerConfig,
        service_client: tinker.ServiceClient,
        state: TinkerTrainerState,
        state_store: StateStore,
        sampling_client: tinker.SamplingClient,
        training_client: tinker.TrainingClient,
    ):
        self.config = config
        self.service_client = service_client
        self.state = state
        self.state_store = state_store
        self._sampling_client = sampling_client
        self._training_client = training_client
        self._train_op = None

        # Set up renderer and tokenizer
        tokenizer = self._sampling_client.get_tokenizer()
        try:
            renderer_name = get_recommended_renderer_name(self.state.base_model)
        except Exception:
            renderer_name = "role_colon"
        self.renderer = get_renderer(renderer_name, tokenizer)

    async def _persist_state(self) -> None:
        await self.state_store.save(self.state.model_dump_json())

    async def sample(self, request: ChatCompletionRequest) -> SampleOp:
        task = asyncio.create_task(self._run_sample(request))
        return TinkerSampleOp(task)

    async def _run_sample(self, request: ChatCompletionRequest) -> Sample:
        messages = []
        for msg in request.messages:
            content_val = msg.get("content") or ""
            if not isinstance(content_val, str):
                content_val = str(content_val)
            messages.append(
                Message(
                    role=msg["role"],
                    content=content_val,
                )
            )
        model_input = self.renderer.build_generation_prompt(messages)

        max_tokens = request.max_tokens or self.config.max_tokens or 1024
        temperature = self.config.temperature or 1.0
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        response = await self._sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

        if not response.sequences:
            raise RuntimeError("Sampling returned no sequences")

        sequence = response.sequences[0]
        parsed_message, parse_success = self.renderer.parse_response(sequence.tokens)
        if not parse_success and sequence.stop_reason != "length":
            raise NotImplementedError("Malformed assistant response or function call")
        text_content = parsed_message.get("content") or ""

        tool_calls: list[Any] = []
        if "tool_calls" in parsed_message and parsed_message["tool_calls"]:
            for tc in parsed_message["tool_calls"]:
                call_id = tc.id if tc.id else f"call_{uuid.uuid4().hex}"
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=call_id,
                        type="function",
                        function=Function(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )

        completion_id = f"cmpl-{uuid.uuid4()}"
        finish_reason = sequence.stop_reason
        if tool_calls:
            finish_reason = "tool_calls"
        elif finish_reason == "stop":
            finish_reason = "stop"
        elif finish_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    message=ChatCompletionMessage(
                        content=text_content if text_content else None,
                        role="assistant",
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=model_input.length,
                completion_tokens=len(sequence.tokens),
                total_tokens=model_input.length + len(sequence.tokens),
            ),
        )

        return Sample(
            completion=completion,
            policy_generation=str(self.state.sampler_step),
        )

    async def train_step(self, examples: List[Example]) -> TrainOp:
        logger.info(
            f"TinkerTrainer train_step called with {len(examples)} examples (no-op in Phase 2)."
        )
        self.state.train_step += 1
        if self.state.train_step % self.config.sampler_promotion_every == 0:
            self.state.sampler_step = self.state.train_step
        await self._persist_state()
        op = TinkerTrainOp()
        self._train_op = op
        return op

    async def in_flight_train_op(self) -> Optional[TrainOp]:
        return self._train_op


class TinkerTrainerFactory(TrainerFactory):
    async def open(
        self,
        name: str,
        state_store: StateStore,
    ) -> TinkerTrainer:
        service_url = os.environ.get("TINKER_SERVICE_URL") or os.environ.get(
            "TINKER_BASE_URL"
        )
        api_key = os.environ.get("TINKER_API_KEY")

        config_kwargs: dict[str, Any] = {}
        if service_url:
            config_kwargs["service_url"] = service_url
        if api_key:
            config_kwargs["api_key"] = api_key

        config = TinkerTrainerConfig(**config_kwargs)

        client_kwargs: dict[str, Any] = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.service_url:
            client_kwargs["base_url"] = config.service_url

        service_client = tinker.ServiceClient(**client_kwargs)

        raw_state = await state_store.load()
        if raw_state is None:
            logger.info(
                f"Bootstrapping Tinker training client for model: {config.base_model}"
            )
            training_client = await service_client.create_lora_training_client_async(
                base_model=config.base_model,
                rank=config.lora_rank,
            )

            logger.info("Saving initial weights for sampler...")
            initial_sampler_name = f"sampler-init-{uuid.uuid4().hex}"
            future = await training_client.save_weights_for_sampler_async(
                name=initial_sampler_name
            )
            save_response = await future
            sampler_path = save_response.path

            sampling_client = await service_client.create_sampling_client_async(
                model_path=sampler_path
            )

            state = TinkerTrainerState(
                sampler_path=sampler_path,
                optimizer_path=None,
                train_step=0,
                sampler_step=0,
                base_model=config.base_model,
                lora_rank=config.lora_rank,
                learning_rate=config.learning_rate,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                kl_penalty_coef=config.kl_penalty_coef,
                loss_fn=config.loss_fn,
            )

            instance = TinkerTrainer(
                config=config,
                service_client=service_client,
                state=state,
                state_store=state_store,
                sampling_client=sampling_client,
                training_client=training_client,
            )
            await instance._persist_state()
        else:
            state = TinkerTrainerState.model_validate_json(raw_state)
            logger.info(
                f"Restoring Tinker trainer from state. Sampler path: {state.sampler_path}"
            )

            if state.optimizer_path:
                training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                    path=state.optimizer_path
                )
            else:
                training_client = (
                    await service_client.create_training_client_from_state_async(
                        path=state.sampler_path
                    )
                )

            sampling_client = await service_client.create_sampling_client_async(
                model_path=state.sampler_path
            )

            instance = TinkerTrainer(
                config=config,
                service_client=service_client,
                state=state,
                state_store=state_store,
                sampling_client=sampling_client,
                training_client=training_client,
            )

        return instance


factory.register("tinker", TinkerTrainerFactory())
