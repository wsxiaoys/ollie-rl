from __future__ import annotations
import asyncio
import logging
import os
import time
import uuid
from typing import Any, List, Optional, Tuple

import torch
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


class StaleBatchError(RuntimeError):
    """
    Raised when too many examples in a `train_step` batch fail the
    client-side staleness filter (more than
    `TinkerTrainerConfig.max_stale_fraction`).

    Signals that the async-RL pipeline is running too off-policy for the
    configured tolerance — operators should tighten `max_steps_off_policy`,
    raise `sampler_promotion_every` cadence, or reduce sampler throughput.
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
    max_steps_off_policy: int = 4
    sampler_promotion_every: int = 1
    # Maximum fraction of a `train_step` batch allowed to be filtered out
    # as stale before `train_step` raises `StaleBatchError`. Default 0.4
    # mirrors the rule-of-thumb that an off-policy pipeline running more
    # than ~40% stale is mis-tuned, not just noisy.
    max_stale_fraction: float = 0.4


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
            policy_generation=str(self.state.sampler_step),
            tokens=full_tokens,
            logprobs=completion_logprobs,
        )

    def _filter_stale(self, examples: List[Example]) -> Tuple[List[Example], int]:
        """
        Drop examples whose `policy_generation` is more than
        `config.max_steps_off_policy` steps behind `state.train_step`.

        This is the client-side analogue of tinker-cookbook's
        `filter_stale_trajectory_group`. The cutoff is inclusive: an
        example sampled at generation `g` survives iff
        `state.train_step - g <= max_steps_off_policy`. Examples whose
        `policy_generation` is not a parseable int are conservatively
        counted as stale (with a warning) — every shipping backend
        stamps it as a stringified int (we do too), so this should
        never trip in practice.

        Returns `(survivors, dropped_count)`.
        """
        cutoff = self.state.train_step - self.config.max_steps_off_policy
        survivors: List[Example] = []
        dropped = 0
        for ex in examples:
            try:
                gen = int(ex.policy_generation)
            except ValueError:
                logger.warning(
                    "TinkerTrainer dropping example with unparseable "
                    f"policy_generation={ex.policy_generation!r}"
                )
                dropped += 1
                continue
            if gen < cutoff:
                dropped += 1
                continue
            survivors.append(ex)
        if dropped:
            logger.info(
                f"TinkerTrainer staleness filter dropped {dropped}/{len(examples)} "
                f"examples (train_step={self.state.train_step}, "
                f"max_steps_off_policy={self.config.max_steps_off_policy})"
            )
        return survivors, dropped

    def _example_to_datum(self, example: Example) -> Optional[tinker.Datum]:
        """
        Convert an `Example` to a single `tinker.Datum` for
        `forward_backward`. Returns None if the example is missing the
        cached tokens/logprobs needed to replay it.

        Layout mirrors `tinker_cookbook.rl.data_processing.trajectory_to_data`
        for the single-turn case:

            full_seq      = prompt + completion
            sampled_lp    = [0.0]*prompt_len + completion_logprobs
            advantages    = [0.0]*prompt_len + [advantage]*completion_len
            mask          = [0.0]*prompt_len + [1.0]*completion_len

            model_input   = full_seq[:-1]   (next-token-prediction inputs)
            target_tokens = full_seq[1:]
            (loss_fn_inputs are all sliced [1:] to align with targets)
        """
        if example.tokens is None or example.logprobs is None:
            logger.warning(
                f"TinkerTrainer skipping example {example.chat_completion_id}: "
                "no cached tokens/logprobs"
            )
            return None

        full_tokens = example.tokens
        completion_logprobs = example.logprobs
        completion_len = len(completion_logprobs)
        prompt_len = len(full_tokens) - completion_len
        if prompt_len < 1 or completion_len < 1:
            logger.warning(
                f"TinkerTrainer skipping example {example.chat_completion_id}: "
                f"degenerate prompt_len={prompt_len}, completion_len={completion_len}"
            )
            return None

        sampled_logprobs = [0.0] * prompt_len + list(completion_logprobs)
        advantages = [0.0] * prompt_len + [example.advantage] * completion_len
        mask = [0.0] * prompt_len + [1.0] * completion_len

        # Drop the leading position so loss inputs align with target_tokens.
        target_tokens = full_tokens[1:]
        sampled_logprobs = sampled_logprobs[1:]
        advantages = advantages[1:]
        mask = mask[1:]

        input_tokens = tinker.ModelInput.from_ints(tokens=full_tokens[:-1])
        return tinker.Datum(
            model_input=input_tokens,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.int64)
                ),
                "logprobs": tinker.TensorData.from_torch(
                    torch.tensor(sampled_logprobs, dtype=torch.float32)
                ),
                "advantages": tinker.TensorData.from_torch(
                    torch.tensor(advantages, dtype=torch.float32)
                ),
                "mask": tinker.TensorData.from_torch(
                    torch.tensor(mask, dtype=torch.float32)
                ),
            },
        )

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

        # 1. Client-side staleness filter. If too large a fraction of
        # the batch is stale, refuse to advance and surface the problem
        # to the operator instead of silently no-op'ing.
        survivors, dropped = self._filter_stale(examples)
        stale_fraction = dropped / len(examples)
        if stale_fraction > self.config.max_stale_fraction:
            raise StaleBatchError(
                f"TinkerTrainer.train_step rejecting batch: "
                f"{dropped}/{len(examples)} examples stale "
                f"(fraction={stale_fraction:.2f} > "
                f"max_stale_fraction={self.config.max_stale_fraction:.2f}). "
                f"Tighten max_steps_off_policy, raise sampler_promotion_every "
                "cadence, or slow sampler throughput."
            )

        # 2. Build Datum list (skipping rows missing cached tokens).
        data: List[tinker.Datum] = []
        for ex in survivors:
            datum = self._example_to_datum(ex)
            if datum is not None:
                data.append(datum)

        if not data:
            raise StaleBatchError(
                "TinkerTrainer.train_step rejecting batch: no surviving "
                "examples carry cached tokens/logprobs after staleness "
                "filter. Verify sample() is writing tokens/logprobs into "
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

        op = TinkerTrainOp()
        self._train_op = op
        return op

    async def in_flight_train_op(self) -> Optional[TrainOp]:
        return self._train_op


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
