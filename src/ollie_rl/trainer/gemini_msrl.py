from __future__ import annotations
import logging
import time
import uuid
from typing import List, Optional, Any

from pydantic import BaseModel

from gemini_msrl.types import (
    ContentGenerationParameters,
    CreateTuningJobRequest,
    GenerateContentTuningScopeRequest,
    GenerateContentTuningScopeResponse,
    GenerationConfig,
    MultiStepReinforcementTuningHyperParameters,
    MultiStepReinforcementTuningSpec,
    ReinforcementTuningTrainingData,
    ReinforcementTuningTrainingDataBatch,
    TrainStepRequest,
    TrainStepResponse,
)
from gemini_msrl import (
    GeminiMsrlClient,
)
from google.genai.types import (
    Content,
    Part,
    FunctionDeclaration,
    Schema,
    FinishReason,
    Tool,
)

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

import json
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat.chat_completion import Choice
from openai.types import CompletionUsage

from ollie_rl.types import ChatCompletionRequest

logger = logging.getLogger(__name__)


# JSON-Schema keywords that google.genai.types.Schema does NOT accept
# (its model has extra="forbid"). OpenAI-compatible clients often include
# these in tool parameter schemas; strip them recursively before handing the
# schema to Schema.model_validate.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    [
        "$schema",
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "additionalProperties",
        "patternProperties",
        "unevaluatedProperties",
        "dependentRequired",
        "dependentSchemas",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "const",
        "examples",
        "default",
        "readOnly",
        "writeOnly",
        "deprecated",
        "title",
    ]
)


def _sanitize_json_schema(node):
    """Best-effort recursive strip of JSON-Schema keywords not supported by
    google.genai.types.Schema. Returns a new structure (does not mutate)."""
    if isinstance(node, dict):
        return {
            k: _sanitize_json_schema(v)
            for k, v in node.items()
            if k not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_sanitize_json_schema(v) for v in node]
    return node


class GeminiMsrlTrainerConfig(BaseModel):
    base_model: str = "gemini-3.5-flash"
    adapter_size: str = "ADAPTER_SIZE_SIXTEEN"
    checkpoint_interval: int = 10
    poll_interval: float = 2.0
    timeout_seconds: float = 3600.0


class GeminiMsrlTrainerState(BaseModel):
    tuning_job_name: str
    last_train_op: Optional[str] = None
    config: GeminiMsrlTrainerConfig


class GeminiMsrlOp:
    def __init__(
        self,
        client: GeminiMsrlClient,
        op_name: str,
    ):
        self.client = client
        self.op_name = op_name

    async def peek(self) -> bool:
        """Return True iff the op has reached a terminal state."""
        operation = await self.client.get_operation(self.op_name)
        return bool(operation.done)


class GeminiMsrlSamplingOp(GeminiMsrlOp, SampleOp):
    def __init__(
        self,
        client: GeminiMsrlClient,
        op_name: str,
        model_name: str,
    ):
        super().__init__(client, op_name)
        self.model_name = model_name

    async def wait(self) -> Sample:
        completed_op = await self.client.wait_for_operation(
            self.op_name,
        )

        response = completed_op.get_response_as(GenerateContentTuningScopeResponse)
        if not response or not response.candidates:
            raise RuntimeError(
                "Failed to retrieve generated candidates from tuning scope response"
            )

        # 4. Pick the first candidate and format as ChatCompletion
        candidate_id, candidate = list(response.candidates.items())[0]

        text_parts = []
        tool_calls: list[
            ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
        ] = []
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    call_id = fc.id if fc.id else f"call_{uuid.uuid4().hex}"
                    args_str = json.dumps(fc.args) if fc.args is not None else "{}"
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=call_id,
                            type="function",
                            function=Function(
                                name=fc.name or "",
                                arguments=args_str,
                            ),
                        )
                    )

        text_content = "\n".join(text_parts) if text_parts else None

        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif candidate.finish_reason:
            finish_reason = candidate.finish_reason
            if finish_reason == FinishReason.STOP:
                finish_reason = "stop"
            elif finish_reason == FinishReason.MAX_TOKENS:
                finish_reason = "length"
            elif finish_reason == FinishReason.MALFORMED_FUNCTION_CALL:
                raise NotImplementedError(
                    "Malformed assistant response or function call"
                )
            elif finish_reason in (
                FinishReason.SAFETY,
                FinishReason.RECITATION,
                FinishReason.BLOCKLIST,
                FinishReason.PROHIBITED_CONTENT,
                FinishReason.IMAGE_SAFETY,
                FinishReason.IMAGE_PROHIBITED_CONTENT,
            ):
                finish_reason = "content_filter"
            else:
                finish_reason = "stop"

        completion = ChatCompletion(
            id=candidate_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    message=ChatCompletionMessage(
                        content=text_content,
                        role="assistant",
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=self.model_name,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=(response.usage_metadata.candidates_token_count or 0)
                if response.usage_metadata
                else 0,
                prompt_tokens=(response.usage_metadata.prompt_token_count or 0)
                if response.usage_metadata
                else 0,
                total_tokens=(response.usage_metadata.total_token_count or 0)
                if response.usage_metadata
                else 0,
            ),
        )

        return Sample(completion=completion, policy_generation=int(response.train_step_id))


class GeminiMsrlTrainingOp(GeminiMsrlOp, TrainOp):
    async def wait(self) -> None:
        completed_op = await self.client.wait_for_operation(
            self.op_name,
        )

        response = completed_op.get_response_as(TrainStepResponse)
        if not response:
            raise RuntimeError("Failed to retrieve train step response")


class GeminiMsrlTrainer(Trainer):
    """
    Trainer wrapping the Gemini MSRL tuning client.

    The Trainer's persistable state lives directly on `self.state`
    (a `GeminiMsrlTrainerState`). Mutate that object in place and then
    call `_persist_state()` to push it to the backing store.
    """

    config: GeminiMsrlTrainerConfig
    client: GeminiMsrlClient
    state: GeminiMsrlTrainerState
    state_store: StateStore

    def __init__(
        self,
        config: GeminiMsrlTrainerConfig,
        client: GeminiMsrlClient,
        state: GeminiMsrlTrainerState,
        state_store: StateStore,
    ):
        self.config = config
        self.client = client
        self.state = state
        self.state_store = state_store

    @property
    def tuning_job_name(self) -> str:
        return self.state.tuning_job_name

    async def _persist_state(self) -> None:
        await self.state_store.save(self.state.model_dump_json())

    async def sample(self, request: ChatCompletionRequest) -> GeminiMsrlSamplingOp:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        # 1. Translate ChatCompletionRequest to GenerateContentTuningScopeRequest
        system_messages = [msg for msg in request.messages if msg["role"] == "system"]
        other_messages = [msg for msg in request.messages if msg["role"] != "system"]

        system_instruction = None
        if system_messages:
            system_content = "\n".join(
                [str(msg["content"]) for msg in system_messages if msg.get("content")]
            )
            if system_content:
                system_instruction = Content(parts=[Part(text=system_content)])

        contents = []
        for msg in other_messages:
            contents.append(
                Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[Part(text=str(msg.get("content", "")))],
                )
            )

        gemini_tools = None
        if request.tools:
            function_declarations = []
            for tool in request.tools:
                if tool.type == "function" and tool.function:
                    func = tool.function
                    # Strip JSON-Schema meta keywords (e.g. "$schema",
                    # "additionalProperties") and a handful of unsupported
                    # constraints that OpenAI-style tool definitions often
                    # include but google.genai.types.Schema rejects under
                    # extra='forbid'. Keep the conversion best-effort.
                    parameters = _sanitize_json_schema(func.parameters)
                    decl = FunctionDeclaration(
                        name=func.name,
                        description=func.description,
                        parameters=Schema.model_validate(parameters),
                    )
                    function_declarations.append(decl)

            if function_declarations:
                gemini_tools = [Tool(function_declarations=function_declarations)]

        # Vertex caps max_output_tokens at 32768 for tuning-scope generations.
        # OpenAI clients (e.g. cloudcode) often send larger values (64000+);
        # clamp to the documented max so we don't 400 on perfectly valid
        # OpenAI-style requests.
        VERTEX_MAX_OUTPUT_TOKENS = 32768
        max_tokens = request.max_tokens
        if max_tokens is None or max_tokens > VERTEX_MAX_OUTPUT_TOKENS:
            max_tokens = VERTEX_MAX_OUTPUT_TOKENS

        tuning_job_id = self.tuning_job_name.split("/")[-1]
        scope_req = GenerateContentTuningScopeRequest(
            content_generation_parameters=ContentGenerationParameters(
                contents=contents,
                generation_config=GenerationConfig(
                    max_output_tokens=max_tokens,
                ),
                system_instruction=system_instruction,
                tools=gemini_tools,
            )
        )

        # 2. Trigger Generation LRO
        op = await self.client.generate_content_tuning_scope(tuning_job_id, scope_req)

        return GeminiMsrlSamplingOp(self.client, op.name, request.model)

    async def train_step(self, examples: List[Example]) -> GeminiMsrlTrainingOp:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        if self.state.last_train_op:
            last_op = GeminiMsrlTrainingOp(
                self.client,
                self.state.last_train_op,
            )
            if not await last_op.peek():
                raise RuntimeError(
                    f"Last training step {self.state.last_train_op} is still active"
                )

        # 1. Translate examples to TrainStepRequest
        rt_examples = []
        for item in examples:
            rt_examples.append(
                ReinforcementTuningTrainingData(
                    candidate_id=item.chat_completion_id,
                    advantage=item.advantage,
                )
            )

        tuning_job_id = self.tuning_job_name.split("/")[-1]
        train_req = TrainStepRequest(
            reinforcement_tuning_training_data_batch=ReinforcementTuningTrainingDataBatch(
                examples=rt_examples
            )
        )

        # 2. Trigger TrainStep LRO
        op = await self.client.train_step(tuning_job_id, train_req)

        self.state.last_train_op = op.name
        await self._persist_state()

        return GeminiMsrlTrainingOp(
            self.client,
            op.name,
        )


class GeminiMsrlTrainerFactory(TrainerFactory):
    """
    Trainer factory for Gemini MSRL trainers.
    """

    async def create(
        self,
        name: str,
        state_store: StateStore,
        trainer_params: Optional[dict] = None,
    ) -> GeminiMsrlTrainer:
        config_kwargs: dict[str, Any] = {}

        if trainer_params:
            config_kwargs.update(trainer_params)

        config = GeminiMsrlTrainerConfig(**config_kwargs)
        # If GEMINI_MSRL_ENV_FILE is set, the client will re-read the auth
        # token from that file (by mtime) on every outgoing request. This lets
        # us refresh tokens externally (e.g. `gcloud auth application-default
        # print-access-token > .env`) without restarting the server.
        client = GeminiMsrlClient()

        # Bootstrap path: create a fresh tuning job and persist its name.
        req = CreateTuningJobRequest(
            tuned_model_display_name=name,
            base_model=config.base_model,
            multi_step_reinforcement_tuning_spec=MultiStepReinforcementTuningSpec(
                hyper_parameters=MultiStepReinforcementTuningHyperParameters(
                    adapter_size=config.adapter_size,
                    checkpoint_interval=config.checkpoint_interval,
                )
            ),
        )

        logger.info(f"Creating Gemini MSRL tuning job for model display name: {name}")
        job = await client.create_tuning_job(req)

        instance = GeminiMsrlTrainer(
            config=config,
            client=client,
            state=GeminiMsrlTrainerState(tuning_job_name=job.name, config=config),
            state_store=state_store,
        )

        # Persist initial state as soon as we have a tuning_job_name, so
        # we can recover even if TPU warm-up is interrupted below.
        await instance._persist_state()

        # 2. Wait for TPU allocation and initialization (JOB_STATE_RUNNING).
        logger.info(
            f"Waiting for tuning job '{instance.tuning_job_name}' to enter RUNNING state..."
        )
        await instance.client.wait_for_tuning_job_running(
            instance.tuning_job_name,
            timeout_seconds=instance.config.timeout_seconds * 2,
            poll_interval=instance.config.poll_interval * 2,
        )

        logger.info("Gemini MSRL Tuning Job is successfully running.")
        return instance

    async def restore(
        self,
        name: str,
        state_store: StateStore,
    ) -> GeminiMsrlTrainer:
        raw_state = await state_store.load()
        if raw_state is None:
            raise ValueError(
                f"Cannot restore Gemini MSRL trainer for {name}: no persisted state found."
            )

        state = GeminiMsrlTrainerState.model_validate_json(raw_state)
        config = state.config

        # If GEMINI_MSRL_ENV_FILE is set, the client will re-read the auth
        # token from that file (by mtime) on every outgoing request. This lets
        # us refresh tokens externally (e.g. `gcloud auth application-default
        # print-access-token > .env`) without restarting the server.
        client = GeminiMsrlClient()

        logger.info(
            f"Restoring Gemini MSRL tuning job from state: {state.tuning_job_name}"
        )

        instance = GeminiMsrlTrainer(
            config=config,
            client=client,
            state=state,
            state_store=state_store,
        )

        # 2. Wait for TPU allocation and initialization (JOB_STATE_RUNNING).
        logger.info(
            f"Waiting for tuning job '{instance.tuning_job_name}' to enter RUNNING state..."
        )
        await instance.client.wait_for_tuning_job_running(
            instance.tuning_job_name,
            timeout_seconds=instance.config.timeout_seconds * 2,
            poll_interval=instance.config.poll_interval * 2,
        )

        logger.info("Gemini MSRL Tuning Job is successfully running.")
        return instance


# Register the factory
factory.register("gemini_msrl", GeminiMsrlTrainerFactory())
