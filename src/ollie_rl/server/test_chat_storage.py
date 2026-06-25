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

if __name__ == "__main__":
    unittest.main()
