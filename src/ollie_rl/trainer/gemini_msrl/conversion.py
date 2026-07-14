from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast

from gemini_msrl.types import ContentGenerationParameters, GenerationConfig
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    Part,
    Schema,
    Tool,
)
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionTokensDetails

from ollie_rl.trainer.types import Sample
from ollie_rl.types import ChatCompletionRequest


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

# Vertex caps max_output_tokens at 32768 for tuning-scope and endpoint
# generations. OpenAI clients (e.g. cloudcode) often send larger values
# (64000+); clamp to avoid 400s on otherwise valid OpenAI-style requests.
VERTEX_MAX_OUTPUT_TOKENS = 32768


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


def _decode_thought_signature(sig: Any) -> bytes | None:
    if isinstance(sig, str):
        try:
            return base64.b64decode(sig)
        except Exception:
            return sig.encode("utf-8")
    if isinstance(sig, bytes):
        return sig
    return None


def _encode_thought_signature(sig: Any) -> str | None:
    if isinstance(sig, bytes):
        return base64.b64encode(sig).decode("utf-8")
    if isinstance(sig, str):
        return sig
    return None


def _set_extra_content(obj: Any, thought_signature: str) -> None:
    extra_content = {"google": {"thought_signature": thought_signature}}
    if obj.model_extra is None:
        try:
            obj.__pydantic_extra__ = {"extra_content": extra_content}
        except Exception:
            pass
    else:
        obj.model_extra["extra_content"] = extra_content


def build_content_generation_parameters(
    request: ChatCompletionRequest,
) -> ContentGenerationParameters:
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
    # ``ValidatorIterator`` -- safe to consume only once. Materialise it in
    # place on every message we touch so subsequent passes see the same data.
    for msg in other_messages:
        if msg.get("role") == "assistant":
            msg_any = cast(dict[str, Any], msg)
            if msg_any.get("tool_calls") is not None:
                msg_any["tool_calls"] = list(msg_any["tool_calls"])

    # Build a lookup of tool_call_id -> function name from prior assistant
    # messages so that we can rehydrate `tool` role messages into Gemini
    # FunctionResponse parts.
    tool_call_name_by_id: dict[str, str] = {}
    for msg in other_messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                tc_id = (
                    tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
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
            response_obj: dict[str, Any]
            try:
                parsed = json.loads(content_str)
                response_obj = (
                    parsed if isinstance(parsed, dict) else {"result": parsed}
                )
            except ValueError:
                response_obj = {"result": content_str}
            except TypeError:
                response_obj = {"result": content_str}
            contents.append(
                Content(
                    role="user",
                    parts=[
                        Part(
                            function_response=FunctionResponse(
                                # Gemini doesn't support id.
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

        # Extract thought signature from the assistant message if present.
        msg_sig = None
        if role == "assistant":
            extra_content = msg.get("extra_content")
            if isinstance(extra_content, dict):
                msg_sig = extra_content.get("google", {}).get("thought_signature")

        if content_val:
            parts.append(
                Part(
                    text=str(content_val),
                    thought_signature=_decode_thought_signature(msg_sig),
                )
            )

        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
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
                    except ValueError:
                        args_obj = {}
                    except TypeError:
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
                parts.append(
                    Part(
                        function_call=FunctionCall(
                            # Gemini doesn't support id.
                            name=fn_name or "",
                            args=args_obj,
                        ),
                        thought_signature=_decode_thought_signature(actual_sig),
                    )
                )

        if not parts:
            # Preserve original behaviour: emit an empty text part rather than
            # skip the message entirely.
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
                parameters = _sanitize_json_schema(func.parameters)
                decl = FunctionDeclaration(
                    name=func.name,
                    description=func.description,
                    parameters=Schema.model_validate(parameters),
                )
                function_declarations.append(decl)

        if function_declarations:
            gemini_tools = [Tool(function_declarations=function_declarations)]

    max_tokens = request.max_tokens
    if max_tokens is None or max_tokens > VERTEX_MAX_OUTPUT_TOKENS:
        max_tokens = VERTEX_MAX_OUTPUT_TOKENS

    return ContentGenerationParameters(
        contents=contents,
        generation_config=GenerationConfig(max_output_tokens=max_tokens),
        system_instruction=system_instruction,
        tools=gemini_tools,
    )


def _candidate_items(
    candidates: Mapping[str, Candidate] | Sequence[Candidate],
    *,
    id_prefix: str,
) -> list[tuple[str, Candidate]]:
    if isinstance(candidates, Mapping):
        return [
            (str(candidate_id), cast(Candidate, candidate))
            for candidate_id, candidate in candidates.items()
        ]
    return [
        (f"{id_prefix}_{idx}_{uuid.uuid4().hex}", candidate)
        for idx, candidate in enumerate(candidates)
    ]


def sample_from_candidates(
    *,
    candidates: Mapping[str, Candidate] | Sequence[Candidate],
    usage_metadata: Any | None,
    model_name: str,
    policy_generation: int,
    id_prefix: str = "cmpl",
) -> Sample:
    candidate_items = _candidate_items(candidates, id_prefix=id_prefix)
    if not candidate_items:
        raise RuntimeError(
            "Failed to retrieve generated candidates from Gemini response"
        )

    candidate_id, candidate = candidate_items[0]

    text_parts = []
    tool_calls: list[
        ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
    ] = []
    message_thought_sig = None

    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            part_sig_str = _encode_thought_signature(
                getattr(part, "thought_signature", None)
            )

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
                    _set_extra_content(tc, part_sig_str)
                tool_calls.append(tc)

    text_content = "\n".join(text_parts) if text_parts else None

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
    elif candidate.finish_reason:
        candidate_finish_reason = candidate.finish_reason
        if candidate_finish_reason == FinishReason.STOP:
            finish_reason = "stop"
        elif candidate_finish_reason == FinishReason.MAX_TOKENS:
            finish_reason = "length"
        elif candidate_finish_reason == FinishReason.MALFORMED_FUNCTION_CALL:
            finish_reason = "content_filter"
        elif candidate_finish_reason in (
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
        _set_extra_content(message, message_thought_sig)

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
        model=model_name,
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=(usage_metadata.candidates_token_count or 0)
            if usage_metadata
            else 0,
            prompt_tokens=(usage_metadata.prompt_token_count or 0)
            if usage_metadata
            else 0,
            total_tokens=(usage_metadata.total_token_count or 0)
            if usage_metadata
            else 0,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=usage_metadata.thoughts_token_count,
            )
            if usage_metadata
            else None,
        ),
    )

    return Sample(
        completion=completion,
        policy_generation=policy_generation,
    )
