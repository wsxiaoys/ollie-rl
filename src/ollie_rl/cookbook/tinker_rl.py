from __future__ import annotations
import time
import uuid
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

import torch
import tinker
from tinker import TensorData

from tinker_cookbook.renderers import Renderer, get_renderer, Message
from tinker_cookbook.model_info import get_recommended_renderer_name

from .types import Recipe, Example, Tuner
from ollie_rl.types import ChatCompletionRequest

logger = logging.getLogger(__name__)


class TinkerRlRecipeConfig(BaseModel):
    base_url: Optional[str] = None
    base_model: str = "Qwen/Qwen3.5-9B-Base"
    renderer_name: Optional[str] = None
    lora_rank: int = 32
    learning_rate: float = 1e-5
    max_tokens: int = 512
    temperature: float = 1.0


class TinkerRlRecipeState(BaseModel):
    model_id: str
    config: TinkerRlRecipeConfig
    state_path: str
    sampler_path: str


class TinkerRlTuner(Tuner):
    """
    Stateful tuner wrapping Tinker's TrainingClient and SamplingClient.
    """

    model_id: str
    config: TinkerRlRecipeConfig
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    renderer: Renderer
    state_path: str
    sampler_path: str
    _rollout_cache: Dict[str, Dict[str, Any]]

    def __init__(
        self,
        model_id: str,
        config: TinkerRlRecipeConfig,
        training_client: tinker.TrainingClient,
        sampling_client: tinker.SamplingClient,
        renderer: Any,
        state_path: str,
        sampler_path: str,
    ):
        self.model_id = model_id
        self.config = config
        self.training_client = training_client
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.state_path = state_path
        self.sampler_path = sampler_path
        self._rollout_cache = {}

    @property
    def kind(self) -> str:
        return "tinker_rl"

    async def sample(self, request: ChatCompletionRequest) -> ChatCompletion:
        # Convert request messages to tinker format
        tinker_messages: list[Message] = []
        for msg in request.messages:
            tinker_messages.append({"role": msg.role, "content": msg.content or ""})

        # Render conversation to prompt tokens
        model_input = self.renderer.build_generation_prompt(tinker_messages)
        prompt_token_ids = model_input.to_ints()

        # Sample from active policy
        stop_seq = self.renderer.get_stop_sequences()
        sample_response = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.config.temperature,
                max_tokens=request.max_tokens or self.config.max_tokens,
                stop=stop_seq,
            ),
        )

        seq = sample_response.sequences[0]
        completion_token_ids = seq.tokens
        logprobs = seq.logprobs

        # Decode/parse generated response
        parsed_message, termination = self.renderer.parse_response(completion_token_ids)
        text_content = parsed_message.get("content", "")

        # Generate unique completion ID and cache rollout details
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        self._rollout_cache[completion_id] = {
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "logprobs": logprobs,
        }

        return ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason="stop" if termination else "length",
                    index=0,
                    message=ChatCompletionMessage(
                        content=text_content,
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=len(completion_token_ids),
                prompt_tokens=len(prompt_token_ids),
                total_tokens=len(prompt_token_ids) + len(completion_token_ids),
            ),
        )

    async def train_step(self, examples: List[Example]) -> None:
        datums = []
        for example in examples:
            cache_entry = self._rollout_cache.get(example.chat_completion_id)
            if not cache_entry:
                logger.warning(
                    f"Rollout cache miss for completion ID: {example.chat_completion_id}. Skipping example."
                )
                continue

            prompt_token_ids = cache_entry["prompt_token_ids"]
            completion_token_ids = cache_entry["completion_token_ids"]
            logprobs = cache_entry["logprobs"]
            advantage = example.advantage

            L_p = len(prompt_token_ids)
            L_r = len(completion_token_ids)
            full_sequence = prompt_token_ids + completion_token_ids

            # Input tokens (N-1)
            model_input = tinker.ModelInput(
                chunks=[tinker.EncodedTextChunk(tokens=full_sequence[:-1])]
            )

            # Targets (N-1)
            targets = full_sequence[1:]

            # Mask, advantages, and logprobs alignment
            mask = [0.0] * (L_p - 1) + [1.0] * L_r
            advantages = [0.0] * (L_p - 1) + [advantage] * L_r

            if not logprobs:
                logprobs = [0.0] * L_r
            aligned_logprobs = [0.0] * (L_p - 1) + logprobs

            assert len(targets) == len(mask) == len(advantages) == len(aligned_logprobs)

            # Construct clean datum (without mask in loss_fn_inputs as Tinker's forward_backward doesn't expect it)
            datums.append(
                tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(targets, dtype=torch.long)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(aligned_logprobs, dtype=torch.float32)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(advantages, dtype=torch.float32)
                        ),
                    },
                )
            )

        if not datums:
            logger.warning(
                "No valid training datums constructed in train_step. Skipping policy update."
            )
            return

        # Execute forward-backward and optimization step in Tinker
        adam_params = tinker.AdamParams(
            learning_rate=self.config.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )

        logger.info(
            f"Running Tinker policy update with {len(datums)} training datums..."
        )
        fwd_bwd_future = await self.training_client.forward_backward_async(
            datums,
            loss_fn="importance_sampling",
        )
        optim_future = await self.training_client.optim_step_async(adam_params)

        # Wait for operations to complete
        await fwd_bwd_future.result_async()
        await optim_future.result_async()

        # Update sampling policy in-place to ensure subsequent rollouts use the updated weights
        logger.info("Updating SamplingClient weights in-place...")
        self.sampling_client = (
            await self.training_client.save_weights_and_get_sampling_client_async()
        )

        # Evict cache to prevent memory leaks
        self._rollout_cache.clear()
        logger.info(
            "Tinker RL train_step completed successfully and rollout cache cleared."
        )

    async def save_state(self) -> str:
        # Save both training state and sampler weights to Tinker storage
        state_future = await self.training_client.save_state_async(
            f"checkpoint-{int(time.time())}"
        )
        sampler_future = await self.training_client.save_weights_for_sampler_async(
            f"sampler-{int(time.time())}"
        )

        state_res = await state_future.result_async()
        sampler_res = await sampler_future.result_async()

        self.state_path = state_res.path
        self.sampler_path = sampler_res.path

        return TinkerRlRecipeState(
            model_id=self.model_id,
            config=self.config,
            state_path=self.state_path,
            sampler_path=self.sampler_path,
        ).model_dump_json()


class TinkerRlRecipe(Recipe):
    """
    Recipe factory for Tinker RL tuners.
    """

    async def create(self, model_id: str) -> TinkerRlTuner:
        config = TinkerRlRecipeConfig()

        # Initialize service client
        service_client = tinker.ServiceClient(
            base_url=config.base_url,
            user_metadata={"recipe_name": "tinker_rl_create"},
        )

        logger.info(
            f"Initializing LoRA training client for model '{config.base_model}'..."
        )
        training_client = await service_client.create_lora_training_client_async(
            config.base_model,
            rank=config.lora_rank,
        )

        # Export initial weights and get sampling client
        logger.info("Creating initial SamplingClient...")
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )

        # Get tokenizer and renderer
        tokenizer = training_client.get_tokenizer()
        renderer_name = config.renderer_name or get_recommended_renderer_name(
            config.base_model
        )
        renderer = get_renderer(renderer_name, tokenizer)

        # Save initial checkpoint paths to populate state_path and sampler_path
        state_future = await training_client.save_state_async("init-checkpoint")
        sampler_future = await training_client.save_weights_for_sampler_async(
            "init-sampler"
        )

        state_res = await state_future.result_async()
        sampler_res = await sampler_future.result_async()

        return TinkerRlTuner(
            model_id=model_id,
            config=config,
            training_client=training_client,
            sampling_client=sampling_client,
            renderer=renderer,
            state_path=state_res.path,
            sampler_path=sampler_res.path,
        )

    async def restore(self, state: str) -> TinkerRlTuner:
        state_data = TinkerRlRecipeState.model_validate_json(state)

        # Initialize service client
        service_client = tinker.ServiceClient(
            base_url=state_data.config.base_url,
            user_metadata={"recipe_name": "tinker_rl_restore"},
        )

        # Restore training client with optimizer state
        logger.info(
            f"Restoring LoRA training client from state path '{state_data.state_path}'..."
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                state_data.state_path
            )
        )

        # Restore sampling client from sampler_path
        logger.info(
            f"Restoring SamplingClient from sampler path '{state_data.sampler_path}'..."
        )
        sampling_client = training_client.create_sampling_client(
            state_data.sampler_path
        )

        # Get tokenizer and renderer
        tokenizer = training_client.get_tokenizer()
        renderer_name = (
            state_data.config.renderer_name
            or get_recommended_renderer_name(state_data.config.base_model)
        )
        renderer = get_renderer(renderer_name, tokenizer)

        return TinkerRlTuner(
            model_id=state_data.model_id,
            config=state_data.config,
            training_client=training_client,
            sampling_client=sampling_client,
            renderer=renderer,
            state_path=state_data.state_path,
            sampler_path=state_data.sampler_path,
        )
