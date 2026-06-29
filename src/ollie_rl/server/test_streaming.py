import json
import unittest
from openai.types.chat import ChatCompletion
from ollie_rl.server.streaming import simulate_stream


class TestStreamingSimulation(unittest.IsolatedAsyncioTestCase):
    async def test_simulate_stream_text_only(self):
        # Construct a simple text ChatCompletion
        completion = ChatCompletion.model_validate(
            {
                "id": "chatcmpl-123",
                "created": 1677652288,
                "model": "test-model",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Hello, world!",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

        # Consume the async generator returned by simulate_stream
        generator = simulate_stream(completion)
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        # We expect exactly 4 chunks:
        # 1. Role chunk
        # 2. Content chunk
        # 3. Final chunk (with finish_reason and usage)
        # 4. DONE terminator
        self.assertEqual(len(chunks), 4)

        # 1. Role chunk assertion
        line_role = chunks[0].decode("utf-8").strip()
        self.assertTrue(line_role.startswith("data: "))
        role_data = json.loads(line_role[6:])
        self.assertEqual(role_data["id"], "chatcmpl-123")
        self.assertEqual(role_data["object"], "chat.completion.chunk")
        self.assertEqual(role_data["choices"][0]["index"], 0)
        self.assertEqual(role_data["choices"][0]["delta"]["role"], "assistant")
        self.assertIsNone(role_data["choices"][0]["finish_reason"])

        # 2. Content chunk assertion
        line_content = chunks[1].decode("utf-8").strip()
        self.assertTrue(line_content.startswith("data: "))
        content_data = json.loads(line_content[6:])
        self.assertEqual(
            content_data["choices"][0]["delta"]["content"], "Hello, world!"
        )
        self.assertIsNone(content_data["choices"][0]["finish_reason"])

        # 3. Final chunk assertion
        line_final = chunks[2].decode("utf-8").strip()
        self.assertTrue(line_final.startswith("data: "))
        final_data = json.loads(line_final[6:])
        self.assertEqual(final_data["choices"][0]["delta"], {})
        self.assertEqual(final_data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(final_data["usage"]["prompt_tokens"], 10)
        self.assertEqual(final_data["usage"]["completion_tokens"], 5)
        self.assertEqual(final_data["usage"]["total_tokens"], 15)

        # 4. Terminator assertion
        self.assertEqual(chunks[3], b"data: [DONE]\n\n")

    async def test_simulate_stream_with_tool_calls(self):
        # Construct a ChatCompletion with tool calls
        completion = ChatCompletion.model_validate(
            {
                "id": "chatcmpl-456",
                "created": 1677652299,
                "model": "test-model-tools",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_weather",
                                        "arguments": '{"location": "San Francisco"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )

        generator = simulate_stream(completion)
        chunks = []
        async for chunk in generator:
            chunks.append(chunk)

        self.assertEqual(len(chunks), 4)

        # Content chunk should contain tool_calls
        line_content = chunks[1].decode("utf-8").strip()
        content_data = json.loads(line_content[6:])
        tool_calls = content_data["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["id"], "call_abc123")
        self.assertEqual(tool_calls[0]["type"], "function")
        self.assertEqual(tool_calls[0]["function"]["name"], "get_current_weather")
        self.assertEqual(
            tool_calls[0]["function"]["arguments"], '{"location": "San Francisco"}'
        )
