from __future__ import annotations
import asyncio
import base64
import logging
import time
import uuid
from typing import List, Optional, Any, cast

from pydantic import BaseModel, model_validator

from gemini_msrl.types import (
    ContentGenerationParameters,
    CreateTuningJobRequest,
    GenerateContentTuningScopeRequest,
    GenerateContentTuningScopeResponse,
    GenerationConfig,
    GenericMetadata,
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
    FunctionCall,
    FunctionResponse,
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
from openai.types.completion_usage import CompletionTokensDetails

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
        "exclusiveMinimum",
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
    tuning_job_name: Optional[str] = None


class CompletedTrainOp(BaseModel):
    """The most recent *completed* train-step LRO and its bookkeeping.

    Bundles the train-step response (source of truth for `policy_generation`)
    together with the operation's resource name and timing metadata so the
    exact Vertex operation is traceable and its execution time
    (`metadata.update_time - metadata.create_time`) is recoverable.
    """

    # The completed train-step response payload.
    response: TrainStepResponse
    # Operation resource name of the completed train-step LRO.
    name: Optional[str] = None
    # `genericMetadata` (createTime / updateTime) of the completed LRO.
    # `update_time - create_time` gives the train op's execution time.
    metadata: Optional[GenericMetadata] = None


class GeminiMsrlTrainerState(BaseModel):
    tuning_job_name: str
    # The most recent *completed* train-step op. This is the source of truth
    # for `policy_generation` and is NEVER overwritten by an in-flight op name,
    # so the generation can't regress to 0 while a new train step is running
    # (that is tracked separately by `pending_train_op`).
    last_train_op: Optional[CompletedTrainOp] = None
    # Op name of an in-flight train-step LRO, if any. Set the moment a train
    # step is submitted and cleared once `GeminiMsrlTrainOp.wait()` observes
    # its completion. Used to detect in-flight training and, via
    # `pending_train_op()`, to reconcile an op that was submitted before a
    # restart.
    pending_train_op: Optional[str] = None
    config: GeminiMsrlTrainerConfig

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_last_train_op(cls, data: Any) -> Any:
        """Backwards-compat for `last_train_op` shape changes:

        1. Oldest format stored the in-flight op name as a `str`; migrate it
           into `pending_train_op`.
        2. Previous format stored the bare `TrainStepResponse` dict; wrap it
           into the new `CompletedTrainOp` container under `response`.
        """
        if not isinstance(data, dict):
            return data
        legacy = data.get("last_train_op")
        if isinstance(legacy, str):
            data = dict(data)
            data.setdefault("pending_train_op", legacy)
            data["last_train_op"] = None
        elif isinstance(legacy, dict) and "response" not in legacy:
            data = dict(data)
            data["last_train_op"] = {"response": legacy}
        return data

    @property
    def train_step(self) -> int:
        if self.last_train_op and self.last_train_op.response.completed_train_step_id:
            return int(self.last_train_op.response.completed_train_step_id)
        return 0


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

    def save_state(self) -> Optional[str]:
        """Serialize this op's resume state: the LRO operation resource name.

        Persisting this the moment the op is submitted lets a later retry
        re-attach to the *same* in-flight Gemini operation (via
        ``sample(request, restore_state=...)``) instead of spawning a fresh op.
        """
        return self.op_name


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
        message_thought_sig = None

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                part_sig = getattr(part, "thought_signature", None)
                part_sig_str = None
                if part_sig:
                    if isinstance(part_sig, bytes):
                        part_sig_str = base64.b64encode(part_sig).decode("utf-8")
                    elif isinstance(part_sig, str):
                        part_sig_str = part_sig

                if part_sig_str and not message_thought_sig:
                    message_thought_sig = part_sig_str

                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    call_id = fc.id if fc.id else f"call_{uuid.uuid4().hex}"
                    args_str = json.dumps(fc.args) if fc.args is not None else "{}"
                    tc = ChatCompletionMessageToolCall(
                        id=call_id,
                        type="function",
                        function=Function(
                            name=fc.name or "",
                            arguments=args_str,
                        ),
                    )
                    if part_sig_str:
                        if tc.model_extra is None:
                            try:
                                tc.__pydantic_extra__ = {
                                    "extra_content": {
                                        "google": {"thought_signature": part_sig_str}
                                    }
                                }
                            except Exception:
                                pass
                        else:
                            tc.model_extra["extra_content"] = {
                                "google": {"thought_signature": part_sig_str}
                            }
                    tool_calls.append(tc)

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
                finish_reason = "content_filter"
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

        message = ChatCompletionMessage(
            content=text_content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
        )
        if message_thought_sig:
            if message.model_extra is None:
                try:
                    message.__pydantic_extra__ = {
                        "extra_content": {
                            "google": {"thought_signature": message_thought_sig}
                        }
                    }
                except Exception:
                    pass
            else:
                message.model_extra["extra_content"] = {
                    "google": {"thought_signature": message_thought_sig}
                }

        completion = ChatCompletion(
            id=candidate_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    message=message,
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
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=response.usage_metadata.thoughts_token_count,
                )
                if response.usage_metadata
                else None,
            ),
        )

        return Sample(
            completion=completion,
            policy_generation=int(response.train_step_id),
        )


class GeminiMsrlTrainOp(GeminiMsrlOp, TrainOp):
    """The single, restart-surviving completion path for a train-step LRO.

    Holds a reference to its :class:`GeminiMsrlTrainer` so ``wait()`` can
    mutate and persist trainer state as it drives the op to completion. This
    is *the* authoritative waiter: the service awaits it (either the fresh op
    returned by ``train_step`` or the one handed back by ``pending_train_op``
    on reconcile) and it records ``last_train_op``, clears ``pending_train_op``
    and persists.
    """

    def __init__(self, trainer: "GeminiMsrlTrainer", op_name: str):
        super().__init__(trainer.client, op_name)
        self.trainer = trainer

    async def wait(self) -> None:
        """Wait for the train-step LRO to terminate, record the completed
        ``TrainStepResponse`` in ``last_train_op`` and clear ``pending_train_op``.

        A train step can legitimately run much longer than a single
        ``wait_for_operation`` budget, so we keep retrying on timeouts /
        transient errors rather than giving up. Bailing out early is what
        previously left ``pending_train_op`` permanently stuck even though the
        underlying Vertex op had finished. Terminal errors are logged and
        swallowed (never raised out of ``wait()``) so the service reconcile
        loop is not killed.
        """
        # A short backoff between retries after a transient (non-timeout)
        # failure so we don't spin tightly on a persistent error.
        _RETRY_BACKOFF_SECONDS = 5.0

        trainer = self.trainer
        op_name = self.op_name

        completed_op = None
        while True:
            try:
                completed_op = await self.client.wait_for_operation(op_name)
                break
            except TimeoutError:
                # Op is still running past the poll budget; keep waiting so the
                # in-flight pointer eventually gets cleared once it completes.
                logger.warning(
                    "Train op %s still running; continuing to poll",
                    op_name,
                )
                continue
            except Exception as e:  # noqa: BLE001
                # Transient failure (e.g. token refresh / network blip). Back
                # off and retry rather than abandoning the poll, which would
                # leave `pending_train_op` stuck.
                logger.warning(
                    "Error polling train op %s: %s; retrying in %.0fs",
                    op_name,
                    e,
                    _RETRY_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                continue

        try:
            response = completed_op.get_response_as(TrainStepResponse)
            if asyncio.iscoroutine(response):
                response = await response
            if (
                isinstance(response, TrainStepResponse)
                and response.completed_train_step_id
            ):
                # Record the completed response monotonically so
                # `policy_generation` never regresses (guards against an older
                # op's poll landing after a newer one).
                completed = int(response.completed_train_step_id)
                if completed >= trainer.state.train_step:
                    # Persist the response together with the op name + its
                    # timing metadata so the exact operation is traceable and
                    # the train-step execution time (updateTime - createTime)
                    # is recoverable without re-querying Vertex.
                    generic_metadata = (completed_op.metadata or {}).get(
                        "genericMetadata"
                    )
                    trainer.state.last_train_op = CompletedTrainOp(
                        response=response,
                        name=completed_op.name,
                        metadata=(
                            GenericMetadata.model_validate(generic_metadata)
                            if generic_metadata
                            else None
                        ),
                    )

            # Clear the in-flight pointer if it still refers to this op; a newer
            # train_step may have already replaced it.
            if trainer.state.pending_train_op == op_name:
                trainer.state.pending_train_op = None

            await trainer._persist_state()
        except Exception as e:
            logger.error(f"Error polling train step or updating state: {e}")


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
        # Lazily-populated one-shot guard that resolves once the tuning job has
        # entered the RUNNING state. TPU allocation/warm-up can take a long
        # time, so we don't block create()/restore() on it; instead the first
        # operation that needs a running job awaits `_ensure_running()`.
        self._running_ready: Optional[asyncio.Task[None]] = None

    async def _ensure_running(self) -> None:
        """Wait (at most once) for the tuning job to enter RUNNING state.

        The wait is deferred out of create()/restore() so those return quickly.
        Concurrent callers share the same underlying wait; on failure the guard
        is reset so a later call can retry.
        """
        if self._running_ready is None:
            self._running_ready = asyncio.create_task(self._wait_for_running())
        try:
            await self._running_ready
        except Exception:
            self._running_ready = None
            raise

    async def _wait_for_running(self) -> None:
        logger.info(
            f"Waiting for tuning job '{self.tuning_job_name}' to enter RUNNING state..."
        )
        await self.client.wait_for_tuning_job_running(
            self.tuning_job_name,
            timeout_seconds=self.config.timeout_seconds * 2,
            poll_interval=self.config.poll_interval * 2,
        )
        logger.info("Gemini MSRL Tuning Job is successfully running.")

    @property
    def policy_generation(self) -> int:
        return self.state.train_step

    @property
    def tuning_job_name(self) -> str:
        return self.state.tuning_job_name

    async def pending_train_op(self) -> Optional[GeminiMsrlTrainOp]:
        """The in-flight train-step LRO, if one is running, else None.

        `train_step` records the in-flight op name in `pending_train_op` the
        moment it submits the LRO, and `GeminiMsrlTrainOp.wait()` clears it
        once the op completes. A non-None `pending_train_op` therefore reliably
        tracks an in-flight op without an extra backend round-trip; we just
        wrap the op name in a fresh (authoritative) `GeminiMsrlTrainOp` handle.
        """
        if self.state.pending_train_op is not None:
            return GeminiMsrlTrainOp(self, self.state.pending_train_op)
        return None

    async def _persist_state(self) -> None:
        await self.state_store.save(self.state.model_dump_json())

    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ) -> GeminiMsrlSamplingOp:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        if restore_state is not None:
            # Re-attach to the already-submitted op instead of submitting a new
            # one. This is the exact inline reconstruction that `train_step` /
            # `restore` already do for the train op. `model_name` only shapes
            # the returned ChatCompletion envelope and comes from the request.
            return GeminiMsrlSamplingOp(self.client, restore_state, request.model)

        # Ensure the tuning job is running before submitting work to it.
        await self._ensure_running()

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

        # OpenAI's assistant-message schema declares ``tool_calls`` as
        # ``Iterable[...]`` which makes pydantic v2 expose it as a lazy
        # ``ValidatorIterator`` -- safe to consume only once. Materialise it
        # in place on every message we touch so subsequent passes (and the
        # lookup we build below) see the same data.
        for msg in other_messages:
            if msg.get("role") == "assistant":
                msg_any = cast(dict[str, Any], msg)
                if msg_any.get("tool_calls") is not None:
                    msg_any["tool_calls"] = list(msg_any["tool_calls"])

        # Build a lookup of tool_call_id -> function name from prior
        # assistant messages so that we can rehydrate `tool` role messages
        # (which only carry `tool_call_id`) into Gemini FunctionResponse parts
        # (which require the function name).
        tool_call_name_by_id: dict[str, str] = {}
        for msg in other_messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    tc_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )
                    fn = (
                        tc.get("function")
                        if isinstance(tc, dict)
                        else getattr(tc, "function", None)
                    )
                    if fn is None:
                        continue
                    fn_name = (
                        fn.get("name")
                        if isinstance(fn, dict)
                        else getattr(fn, "name", None)
                    )
                    if tc_id and fn_name:
                        tool_call_name_by_id[tc_id] = fn_name

        contents = []
        for msg in other_messages:
            role = msg.get("role")
            if role == "tool":
                tool_call_id = msg.get("tool_call_id") or ""
                fn_name = tool_call_name_by_id.get(tool_call_id, "")
                raw_content = msg.get("content", "")
                content_str = (
                    raw_content if isinstance(raw_content, str) else str(raw_content)
                )
                # Try to parse as JSON object; fall back to wrapping it in a
                # {"result": ...} envelope since FunctionResponse.response
                # must be a dict.
                response_obj: dict[str, Any]
                try:
                    parsed = json.loads(content_str)
                    response_obj = (
                        parsed if isinstance(parsed, dict) else {"result": parsed}
                    )
                except ValueError, TypeError:
                    response_obj = {"result": content_str}
                contents.append(
                    Content(
                        role="user",
                        parts=[
                            Part(
                                function_response=FunctionResponse(
                                    # Gemini doesn't support id
                                    # id=tool_call_id or None,
                                    name=fn_name,
                                    response=response_obj,
                                )
                            )
                        ],
                    )
                )
                continue

            parts: list[Part] = []
            content_val = msg.get("content")

            # Extract thought signature from the assistant message if present
            msg_sig = None
            if role == "assistant":
                extra_content = msg.get("extra_content")
                if isinstance(extra_content, dict):
                    msg_sig = extra_content.get("google", {}).get("thought_signature")

            if content_val:
                sig_bytes = None
                if msg_sig:
                    if isinstance(msg_sig, str):
                        try:
                            sig_bytes = base64.b64decode(msg_sig)
                        except Exception:
                            sig_bytes = msg_sig.encode("utf-8")
                    elif isinstance(msg_sig, bytes):
                        sig_bytes = msg_sig
                parts.append(Part(text=str(content_val), thought_signature=sig_bytes))

            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    tc_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )
                    fn = (
                        tc.get("function")
                        if isinstance(tc, dict)
                        else getattr(tc, "function", None)
                    )
                    if fn is None:
                        continue
                    fn_name = (
                        fn.get("name")
                        if isinstance(fn, dict)
                        else getattr(fn, "name", None)
                    )
                    fn_args = (
                        fn.get("arguments")
                        if isinstance(fn, dict)
                        else getattr(fn, "arguments", None)
                    )
                    args_obj: dict[str, Any]
                    if isinstance(fn_args, dict):
                        args_obj = fn_args
                    elif isinstance(fn_args, str) and fn_args:
                        try:
                            parsed_args = json.loads(fn_args)
                            args_obj = (
                                parsed_args
                                if isinstance(parsed_args, dict)
                                else {"value": parsed_args}
                            )
                        except ValueError, TypeError:
                            args_obj = {}
                    else:
                        args_obj = {}

                    # Extract thought signature from the tool call if present,
                    # falling back to the message-level signature.
                    tc_sig = None
                    if isinstance(tc, dict):
                        tc_extra = tc.get("extra_content")
                        if isinstance(tc_extra, dict):
                            tc_sig = tc_extra.get("google", {}).get("thought_signature")
                    else:
                        tc_extra = getattr(tc, "extra_content", None)
                        if isinstance(tc_extra, dict):
                            tc_sig = tc_extra.get("google", {}).get("thought_signature")

                    actual_sig = tc_sig or msg_sig or "skip_thought_signature_validator"
                    sig_bytes = None
                    if actual_sig:
                        if isinstance(actual_sig, str):
                            try:
                                sig_bytes = base64.b64decode(actual_sig)
                            except Exception:
                                sig_bytes = actual_sig.encode("utf-8")
                        elif isinstance(actual_sig, bytes):
                            sig_bytes = actual_sig

                    parts.append(
                        Part(
                            function_call=FunctionCall(
                                # Gemini doesn't support id
                                # id=tc_id or None,
                                name=fn_name or "",
                                args=args_obj,
                            ),
                            thought_signature=sig_bytes,
                        )
                    )

            if not parts:
                # Preserve original behaviour: emit an empty text part rather
                # than skip the message entirely.
                parts = [Part(text="")]

            contents.append(
                Content(
                    role="model" if role == "assistant" else "user",
                    parts=parts,
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

    async def train_step(self, examples: List[Example]) -> GeminiMsrlTrainOp:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        # Ensure the tuning job is running before submitting work to it.
        await self._ensure_running()

        if self.state.pending_train_op:
            last_op = GeminiMsrlTrainOp(
                self,
                self.state.pending_train_op,
            )
            if not await last_op.peek():
                raise RuntimeError(
                    f"Last training step {self.state.pending_train_op} is still active"
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

        self.state.pending_train_op = op.name
        await self._persist_state()

        # The returned op is the single authoritative waiter: the service
        # awaits `wait()`, which drives the op to completion, records
        # `last_train_op`, clears `pending_train_op` and persists. No
        # background poll is spawned here (nor re-spawned on restart) --
        # reconcile flows through `pending_train_op()` instead.
        return GeminiMsrlTrainOp(self, op.name)


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
        if config.tuning_job_name:
            tuning_job_name = config.tuning_job_name
            if "/" not in tuning_job_name:
                tuning_job_name = f"projects/{client.project_id}/locations/{client.location}/tuningJobs/{tuning_job_name}"
            logger.info(f"Using pre-created Gemini MSRL tuning job: {tuning_job_name}")
        else:
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

            logger.info(
                f"Creating Gemini MSRL tuning job for model display name: {name}"
            )
            job = await client.create_tuning_job(req)
            tuning_job_name = job.name

        instance = GeminiMsrlTrainer(
            config=config,
            client=client,
            state=GeminiMsrlTrainerState(
                tuning_job_name=tuning_job_name,
                config=config,
            ),
            state_store=state_store,
        )

        # Persist initial state as soon as we have a tuning_job_name, so
        # we can recover even if TPU warm-up is interrupted below.
        await instance._persist_state()

        # TPU allocation and initialization (JOB_STATE_RUNNING) can take a long
        # time. We don't block creation on it; the wait is deferred to the
        # first operation that needs a running job via `_ensure_running()`.
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

        # A non-None `pending_train_op` means we never observed the op's
        # completion before the restart. We no longer re-spawn a poll here:
        # the service reconciles via `pending_train_op()` -> `wait()`, which is
        # the single restart-surviving completion path.

        # TPU allocation and initialization (JOB_STATE_RUNNING) can take a long
        # time. We don't block restore on it; the wait is deferred to the first
        # operation that needs a running job via `_ensure_running()`.
        return instance


# Register the factory
factory.register("gemini_msrl", GeminiMsrlTrainerFactory())
