import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from ollie_rl.server.app import app, storage, chat_storage
from ollie_rl.server.chat_storage import ChatStorage


class TestChatStorage(unittest.TestCase):
    def setUp(self):
        # Initialize a clean ChatStorage instance (defaults to Fakeredis)
        self.storage = ChatStorage()
        self.client = TestClient(app)

    def tearDown(self):
        asyncio.run(self.storage.close())

    def test_record_and_get_completions(self):
        chat_id = "test-chat-1"
        comp1 = "chatcmpl-001"
        comp2 = "chatcmpl-002"

        # Record completions
        asyncio.run(self.storage.record_completion(chat_id, comp1))
        asyncio.run(self.storage.record_completion(chat_id, comp2))

        # Retrieve completions
        completions = asyncio.run(self.storage.get_completions(chat_id))
        self.assertEqual(completions, [comp1, comp2])

    def test_record_multiple_chats(self):
        chat_a = "chat-a"
        chat_b = "chat-b"
        comp_a = "chatcmpl-a"
        comp_b = "chatcmpl-b"

        asyncio.run(self.storage.record_completion(chat_a, comp_a))
        asyncio.run(self.storage.record_completion(chat_b, comp_b))

        completions_a = asyncio.run(self.storage.get_completions(chat_a))
        completions_b = asyncio.run(self.storage.get_completions(chat_b))

        self.assertEqual(completions_a, [comp_a])
        self.assertEqual(completions_b, [comp_b])

    def test_ttl_duration(self):
        chat_id = "test-ttl-chat"
        comp_id = "chatcmpl-ttl"

        # Record with a specific TTL
        asyncio.run(self.storage.record_completion(chat_id, comp_id, ttl_seconds=100))

        # Check TTL on the underlying client
        key = f"chat:{chat_id}:completions"
        ttl = asyncio.run(self.storage.client.ttl(key))
        self.assertGreater(ttl, 0)
        self.assertLessEqual(ttl, 100)

    def test_api_integration(self):
        # Create a mock tuner
        mock_tuner = MagicMock()
        mock_tuner.sample = AsyncMock()

        mock_completion = ChatCompletion(
            id="chatcmpl-api-integration",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Integration response",
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model="test-model",
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=5,
                prompt_tokens=5,
                total_tokens=10,
            ),
        )
        mock_tuner.sample.return_value = mock_completion

        # Patch storage.get to return our mock_tuner
        with patch.object(storage, "get", return_value=mock_tuner):
            # Clear completions for this chat_id first to ensure test isolation
            chat_id = "chat-integration-123"
            asyncio.run(chat_storage.client.delete(f"chat:{chat_id}:completions"))

            # Make request with x-chat-id header
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "assistant", "content": "Hello"}],
                },
                headers={"x-chat-id": chat_id},
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["id"], "chatcmpl-api-integration")

            # Now retrieve completions using the GET endpoint
            get_response = self.client.get(f"/v1/chats/{chat_id}/completions")
            self.assertEqual(get_response.status_code, 200)
            get_data = get_response.json()
            self.assertEqual(get_data["chat_id"], chat_id)
            self.assertEqual(get_data["completion_ids"], ["chatcmpl-api-integration"])


if __name__ == "__main__":
    unittest.main()
