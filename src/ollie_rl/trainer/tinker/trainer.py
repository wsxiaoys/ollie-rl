from __future__ import annotations
import asyncio
import logging
import os
import time
import uuid
from typing import Any, List, Optional

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
from .accumulator import examples_to_data

logger = logging.getLogger(__name__)


class StaleBatchError(RuntimeError):
    """
    Raised when too many examples in a `train_step` batch fail or are empty.
    """


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
    sampler_promotion_every: int = 1


class TinkerTrainerState(BaseModel):
    # Tinker-side identity
    sampler_path: str  # latest saved sampler weights (also seeds restore)
    optimizer_path: Optional[str] = None  # latest full checkpoint (weights + opt state)

    # Async-RL bookkeeping
    train_step: int  # monotonically increasing; mirrors AsyncConfig step
    sampler_step: int  # train_step at which `sampler_path` was published

    # Backend config (frozen at create-time)
    config: TinkerTrainerConfig


class TinkerTrainOp(TrainOp):
    """
    Completed-op sentinel for :class:`TinkerTrainer`.

    Unlike LRO-style backends (e.g. Gemini MSRL), Tinker's training
    primitives are exposed as locally-awaited futures rather than
    remote long-running operations we can poll for. We therefore drive
    the entire pipeline (``forward_backward_async`` →
    ``optim_step_async`` → optional sampler promotion → state
    persist) inline inside :meth:`TinkerTrainer.train_step` and only
    return once everything has been awaited.

    Consequently the returned op is *already finished* by
    construction:

    - :meth:`peek` always returns ``True`` (the work is done).
    - :meth:`wait` is a no-op.

    This is intentional. Callers that use ``await train_op.wait()``
    as a uniform completion barrier across backends will simply fall
    through.
    """

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

        # Set up renderer and tokenizer
        tokenizer = self._sampling_client.get_tokenizer()
        try:
            renderer_name = get_recommended_renderer_name(self.state.config.base_model)
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
        # Snapshot the prompt + completion tokens (and per-completion-token
        # logprobs) so train_step can replay them through
        # forward_backward without re-sampling. Layout matches
        # Sample.tokens / Sample.logprobs:
        #   tokens   = prompt_tokens + completion_tokens
        #   logprobs = completion logprobs only
        # (Hence prompt_len = len(tokens) - len(logprobs).)
        prompt_tokens = model_input.to_ints()
        completion_tokens = list(sequence.tokens)
        completion_logprobs = (
            list(sequence.logprobs) if sequence.logprobs is not None else []
        )
        full_tokens = prompt_tokens + completion_tokens
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
            policy_generation=self.state.sampler_step,
            tokens=full_tokens,
            logprobs=completion_logprobs,
        )

    @property
    def policy_generation(self) -> int:
        return self.state.train_step

    def _examples_to_data(self, examples: List[Example]) -> List[tinker.Datum]:
        """
        Groups examples into trajectory-level tinker.Datums by reconstructing
        the prefix-tree structure of their tokens.
        """
        return examples_to_data(examples)

    async def _promote_sampler(self) -> None:
        """
        Snapshot the current weights into a new sampler checkpoint and
        swap `self._sampling_client` to it. Updates `state.sampler_path`
        and `state.sampler_step`. Caller is responsible for persisting
        state.
        """
        sampler_name = f"sampler-step-{self.state.train_step}-{uuid.uuid4().hex[:8]}"
        future = await self._training_client.save_weights_for_sampler_async(
            name=sampler_name
        )
        save_response = await future
        new_sampler_path = save_response.path
        new_sampling_client = await self.service_client.create_sampling_client_async(
            model_path=new_sampler_path
        )
        self._sampling_client = new_sampling_client
        self.state.sampler_path = new_sampler_path
        self.state.sampler_step = self.state.train_step
        logger.info(
            f"TinkerTrainer promoted sampler to step {self.state.sampler_step} "
            f"({new_sampler_path})"
        )

    async def train_step(self, examples: List[Example]) -> TrainOp:
        logger.info(f"TinkerTrainer.train_step called with {len(examples)} examples")

        if not examples:
            # Empty batch is a structural caller bug, not a staleness
            # problem; surface it the same way (consistent with the
            # plan's "too few survive → fail loudly" branch).
            raise StaleBatchError("TinkerTrainer.train_step received an empty batch")

        # 2. Build Datum list (skipping rows missing cached tokens and grouping into trajectories).
        data = self._examples_to_data(examples)

        if not data:
            raise StaleBatchError(
                "TinkerTrainer.train_step rejecting batch: no surviving "
                "examples carry cached tokens/logprobs. "
                "Verify sample() is writing tokens/logprobs into "
                "ChatCompletionModel."
            )

        # 3. forward_backward + optim_step.
        loss_fn_config = {"kl_penalty_coef": self.config.kl_penalty_coef}
        # `loss_fn` is a tinker Literal type; we hold it as `str` on the
        # config (lets us accept future loss kinds without bumping the
        # config schema) and cast at the call boundary.
        loss_fn: Any = self.config.loss_fn
        fb_future = await self._training_client.forward_backward_async(
            data,
            loss_fn,
            loss_fn_config=loss_fn_config,
        )
        await fb_future
        opt_future = await self._training_client.optim_step_async(
            tinker.AdamParams(learning_rate=self.config.learning_rate)
        )
        await opt_future

        # 4. Advance step counter; promote sampler at cadence.
        self.state.train_step += 1
        if self.state.train_step % self.config.sampler_promotion_every == 0:
            await self._promote_sampler()
        await self._persist_state()

        return TinkerTrainOp()


class TinkerTrainerFactory(TrainerFactory):
    async def create(
        self,
        name: str,
        state_store: StateStore,
        trainer_params: Optional[dict] = None,
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

        if trainer_params:
            config_kwargs.update(trainer_params)

        config = TinkerTrainerConfig(**config_kwargs)

        client_kwargs: dict[str, Any] = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.service_url:
            client_kwargs["base_url"] = config.service_url

        service_client = tinker.ServiceClient(**client_kwargs)

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
            config=config,
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
        return instance

    async def restore(
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

        raw_state = await state_store.load()
        if raw_state is None:
            raise ValueError(
                f"Cannot restore Tinker trainer for {name}: no persisted state found."
            )

        state = TinkerTrainerState.model_validate_json(raw_state)
        logger.info(
            f"Restoring Tinker trainer from state. Sampler path: {state.sampler_path}"
        )

        # Override config fields with frozen values from state
        config = state.config

        client_kwargs: dict[str, Any] = {}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.service_url:
            client_kwargs["base_url"] = config.service_url

        service_client = tinker.ServiceClient(**client_kwargs)

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
