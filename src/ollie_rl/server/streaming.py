import json
from typing import AsyncIterator
from openai.types.chat import ChatCompletion


def simulate_stream(completion: ChatCompletion) -> AsyncIterator[bytes]:
    """Convert a non-streaming ChatCompletion into a simulated SSE stream of
    ChatCompletionChunk events, matching OpenAI's streaming protocol.

    The full response is emitted as a single content delta chunk followed by a
    terminating chunk with the original finish_reason. This is purely a
    client-compatibility shim; no real token-by-token streaming occurs.
    """

    async def gen() -> AsyncIterator[bytes]:
        base = {
            "id": completion.id,
            "created": completion.created,
            "model": completion.model,
            "object": "chat.completion.chunk",
        }

        # Initial chunk: announce the assistant role for each choice.
        role_chunk = {
            **base,
            "choices": [
                {
                    "index": choice.index,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
                for choice in completion.choices
            ],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8")

        # Content chunk: emit the full content (and any tool_calls) as a single delta.
        content_choices = []
        for choice in completion.choices:
            delta: dict = {}
            if choice.message.content is not None:
                delta["content"] = choice.message.content
            if choice.message.tool_calls:
                tool_calls = []
                for idx, tc in enumerate(choice.message.tool_calls):
                    entry: dict = {"index": idx, "id": tc.id, "type": tc.type}
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        entry["function"] = {
                            "name": fn.name,
                            "arguments": fn.arguments,
                        }
                    tool_calls.append(entry)
                delta["tool_calls"] = tool_calls
            content_choices.append(
                {
                    "index": choice.index,
                    "delta": delta,
                    "finish_reason": None,
                }
            )
        if content_choices:
            content_chunk = {**base, "choices": content_choices}
            yield f"data: {json.dumps(content_chunk)}\n\n".encode("utf-8")

        # Final chunk: empty delta with finish_reason set.
        final_chunk = {
            **base,
            "choices": [
                {
                    "index": choice.index,
                    "delta": {},
                    "finish_reason": choice.finish_reason,
                }
                for choice in completion.choices
            ],
        }
        if completion.usage is not None:
            final_chunk["usage"] = completion.usage.model_dump(exclude_none=True)
        yield f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8")

        # Terminator
        yield b"data: [DONE]\n\n"

    return gen()
