from __future__ import annotations
import logging
import os
import time
import uuid
from typing import List

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

from .types import Recipe, Example, Tuner

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


class GeminiMsrlRecipeConfig(BaseModel):
    auth_token: str
    project_id: str
    location: str = "us-central1"
    base_model: str = "gemini-3.5-flash"
    adapter_size: str = "ADAPTER_SIZE_SIXTEEN"
    checkpoint_interval: int = 10
    poll_interval: float = 2.0
    timeout_seconds: float = 300.0


class GeminiMsrlRecipeState(BaseModel):
    tuner_id: str
    tuning_job_name: str


class GeminiMsrlTuner(Tuner):
    """
    Tuner wrapping the Gemini MSRL tuning client.
    """

    config: GeminiMsrlRecipeConfig
    client: GeminiMsrlClient
    tuning_job_name: str

    def __init__(
        self,
        tuner_id: str,
        config: GeminiMsrlRecipeConfig,
        client: GeminiMsrlClient,
        tuning_job_name: str = "",
    ):
        self._tuner_id = tuner_id
        self.config = config
        self.client = client
        self.tuning_job_name = tuning_job_name

    @property
    def tuner_id(self) -> str:
        return self._tuner_id

    @property
    def kind(self) -> str:
        return "gemini_msrl"

    async def sample(self, request: ChatCompletionRequest) -> ChatCompletion:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        # 1. Translate ChatCompletionRequest to GenerateContentTuningScopeRequest
        system_messages = [msg for msg in request.messages if msg.role == "system"]
        other_messages = [msg for msg in request.messages if msg.role != "system"]

        system_instruction = None
        if system_messages:
            system_content = "\n".join(
                [msg.content for msg in system_messages if msg.content]
            )
            if system_content:
                system_instruction = Content(parts=[Part(text=system_content)])

        contents = []
        for msg in other_messages:
            contents.append(
                Content(
                    role="user" if msg.role == "user" else "model",
                    parts=[Part(text=msg.content)],
                )
            )

        gemini_tools = None
        if request.tools:
            function_declarations = []
            for tool in request.tools:
                if tool.type == "function" and tool.function:
                    func = tool.function
                    decl = FunctionDeclaration(
                        name=func.name,
                        description=func.description,
                        parameters=Schema.model_validate(func.parameters),
                    )
                    function_declarations.append(decl)

            if function_declarations:
                gemini_tools = [Tool(function_declarations=function_declarations)]

        tuning_job_id = self.tuning_job_name.split("/")[-1]
        scope_req = GenerateContentTuningScopeRequest(
            content_generation_parameters=ContentGenerationParameters(
                contents=contents,
                generation_config=GenerationConfig(
                    max_output_tokens=request.max_tokens,
                ),
                system_instruction=system_instruction,
                tools=gemini_tools,
            )
        )

        # 2. Trigger Generation LRO
        op = await self.client.generate_content_tuning_scope(tuning_job_id, scope_req)

        # 3. Poll LRO to completion
        completed_op = await self.client.wait_for_operation(
            op.name,
            timeout_seconds=self.config.timeout_seconds,
            poll_interval=self.config.poll_interval,
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
                raise NotImplementedError()
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

        return ChatCompletion(
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
            model=request.model,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=response.usage_metadata.candidates_token_count
                if response.usage_metadata
                else 0,
                prompt_tokens=response.usage_metadata.prompt_token_count
                if response.usage_metadata
                else 0,
                total_tokens=response.usage_metadata.total_token_count
                if response.usage_metadata
                else 0,
            ),
        )

    async def train_step(self, examples: List[Example]) -> None:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

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

        # 3. Poll LRO to completion
        completed_op = await self.client.wait_for_operation(
            op.name,
            timeout_seconds=self.config.timeout_seconds,
            poll_interval=self.config.poll_interval,
        )

        response = completed_op.get_response_as(TrainStepResponse)
        if not response:
            raise RuntimeError("Failed to retrieve train step response")

    async def save_state(self) -> str:
        if not self.tuning_job_name:
            raise RuntimeError("Tuning job not initialized, cannot save state")
        return GeminiMsrlRecipeState(
            tuner_id=self.tuner_id,
            tuning_job_name=self.tuning_job_name,
        ).model_dump_json()


class GeminiMsrlRecipe(Recipe):
    """
    Recipe factory for Gemini MSRL tuners.
    """

    async def create(self, tuner_id: str) -> GeminiMsrlTuner:
        config = GeminiMsrlRecipeConfig(
            auth_token=os.environ.get("GEMINI_MSRL_AUTH_TOKEN", "dummy-auth-token"),
            project_id=os.environ.get("GEMINI_MSRL_PROJECT_ID", "dummy-project-id"),
        )
        client = GeminiMsrlClient(
            auth_token=config.auth_token,
            project_id=config.project_id,
            location=config.location,
        )

        # 1. Create the Tuning Job
        req = CreateTuningJobRequest(
            tuned_model_display_name=tuner_id,
            base_model=config.base_model,
            multi_step_reinforcement_tuning_spec=MultiStepReinforcementTuningSpec(
                hyper_parameters=MultiStepReinforcementTuningHyperParameters(
                    adapter_size=config.adapter_size,
                    checkpoint_interval=config.checkpoint_interval,
                )
            ),
        )

        logger.info(
            f"Creating Gemini MSRL tuning job for model display name: {tuner_id}"
        )
        job = await client.create_tuning_job(req)
        tuning_job_name = job.name

        instance = GeminiMsrlTuner(
            tuner_id=tuner_id,
            config=config,
            client=client,
            tuning_job_name=tuning_job_name,
        )

        # 2. Wait for TPU allocation and initialization (JOB_STATE_RUNNING)
        logger.info(
            f"Waiting for tuning job '{instance.tuning_job_name}' to enter RUNNING state..."
        )
        await instance.client.wait_for_tuning_job_running(
            instance.tuning_job_name,
            timeout_seconds=instance.config.timeout_seconds * 2,
            poll_interval=instance.config.poll_interval * 2,
        )

        logger.info("Gemini MSRL Tuning Job is successfully initialized and RUNNING.")
        return instance

    async def restore(self, state: str) -> GeminiMsrlTuner:
        state_data = GeminiMsrlRecipeState.model_validate_json(state)
        tuner_id = state_data.tuner_id
        tuning_job_name = state_data.tuning_job_name

        config = GeminiMsrlRecipeConfig(
            auth_token=os.environ.get("GEMINI_MSRL_AUTH_TOKEN", "dummy-auth-token"),
            project_id=os.environ.get("GEMINI_MSRL_PROJECT_ID", "dummy-project-id"),
        )
        client = GeminiMsrlClient(
            auth_token=config.auth_token,
            project_id=config.project_id,
            location=config.location,
        )

        instance = GeminiMsrlTuner(
            tuner_id=tuner_id,
            config=config,
            client=client,
            tuning_job_name=tuning_job_name,
        )

        logger.info(
            f"Restoring Gemini MSRL tuning job from state: {instance.tuning_job_name}"
        )
        await instance.client.wait_for_tuning_job_running(
            instance.tuning_job_name,
            timeout_seconds=instance.config.timeout_seconds * 2,
            poll_interval=instance.config.poll_interval * 2,
        )

        logger.info("Gemini MSRL Tuning Job is successfully restored and RUNNING.")
        return instance
