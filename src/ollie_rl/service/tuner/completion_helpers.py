"""Pure helpers for hashing requests and inspecting/rewriting completions."""

import hashlib
import json
from typing import Any, Dict, Optional

from openai.types.chat import ChatCompletion

from ollie_rl.types import ChatCompletionRequest


def hash_request(request: "ChatCompletionRequest") -> str:
    """
    Stable SHA-256 digest of a request's prompt.

    The prompt is everything the model conditions on: the ``messages`` *and*
    the available ``tools`` (different tool schemas can yield different
    responses for the same messages, so they must be part of the key).

    Retries of a stalled request re-send the identical prompt, so this gives a
    per-turn idempotency key: within a linear agent run, a repeat digest is
    always a retry of the same turn. Fields are dumped in JSON mode with sorted
    keys so semantically identical prompts hash the same regardless of dict
    ordering.
    """
    dumped = request.model_dump(mode="json")
    key = {
        "messages": dumped.get("messages", []),
        "tools": dumped.get("tools"),
    }
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _completion_context_tokens(completion: ChatCompletion) -> int:
    """Return prompt + completion + reasoning tokens reported for completion."""
    usage = completion.usage
    if usage is None:
        return 0

    reasoning_tokens = 0
    details = usage.completion_tokens_details
    if details is not None:
        reasoning_tokens = details.reasoning_tokens or 0

    return (
        (usage.prompt_tokens or 0) + (usage.completion_tokens or 0) + reasoning_tokens
    )


def context_tokens_from_response(response: Dict[str, Any]) -> Optional[int]:
    """Return prompt + completion + reasoning tokens from a stored response JSON."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    reasoning_tokens = 0
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = int(details.get("reasoning_tokens") or 0)

    return (
        int(usage.get("prompt_tokens") or 0)
        + int(usage.get("completion_tokens") or 0)
        + reasoning_tokens
    )


def _completion_exceeded_context_window(
    completion: ChatCompletion, max_context_window: Optional[int]
) -> bool:
    if max_context_window is None:
        return False
    return _completion_context_tokens(completion) > max_context_window


def completion_has_length_finish(completion: ChatCompletion) -> bool:
    """Whether any choice in the completion finished due to length."""
    return any(choice.finish_reason == "length" for choice in completion.choices)


def _clear_completion_as_length(completion: ChatCompletion) -> ChatCompletion:
    """Return a copy with every choice converted to an empty length stop."""
    cleared = completion.model_copy(deep=True)
    for choice in cleared.choices:
        choice.finish_reason = "length"
        if choice.message is None:
            continue
        choice.message.content = None
        choice.message.tool_calls = None
        if hasattr(choice.message, "function_call"):
            choice.message.function_call = None
        if hasattr(choice.message, "refusal"):
            choice.message.refusal = None
    return cleared


def apply_max_context_window(
    completion: ChatCompletion, max_context_window: Optional[int]
) -> ChatCompletion:
    """Convert oversized completions to cleared length samples."""
    if not _completion_exceeded_context_window(completion, max_context_window):
        return completion
    return _clear_completion_as_length(completion)
