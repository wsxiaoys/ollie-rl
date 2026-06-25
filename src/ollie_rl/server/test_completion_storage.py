import asyncio
import unittest
from fastapi.testclient import TestClient

from ollie_rl.server.app import app
from ollie_rl.server.completion_storage import CompletionStorage


class TestCompletionStorage(unittest.TestCase):
    def setUp(self):
        # Initialize a clean CompletionStorage instance (defaults to Fakeredis)
        self.storage = CompletionStorage()
        self.client = TestClient(app)

    def tearDown(self):
        asyncio.run(self.storage.close())

    def test_record_and_get_chat_completions(self):
        chat_id = "test-chat-1"
        tuner_id = "test-tuner-1"
        comp1 = "chatcmpl-001"
        comp2 = "chatcmpl-002"

        # Record completions
        asyncio.run(
            self.storage.record_completion(
                completion_id=comp1, tuner_id=tuner_id, chat_id=chat_id
            )
        )
        asyncio.run(
            self.storage.record_completion(
                completion_id=comp2, tuner_id=tuner_id, chat_id=chat_id
            )
        )

        # Retrieve completions
        completions = asyncio.run(self.storage.get_chat_completions(chat_id))
        self.assertEqual(completions, [comp1, comp2])

    def test_record_and_get_tuner_completions(self):
        tuner_id = "test-tuner-1"
        comp1 = "chatcmpl-001"
        comp2 = "chatcmpl-002"

        # Record completions
        asyncio.run(
            self.storage.record_completion(completion_id=comp1, tuner_id=tuner_id)
        )
        asyncio.run(
            self.storage.record_completion(completion_id=comp2, tuner_id=tuner_id)
        )

        # Retrieve completions
        completions = asyncio.run(self.storage.get_tuner_completions(tuner_id))
        self.assertEqual(completions, [comp1, comp2])

    def test_record_both_chat_and_tuner(self):
        chat_id = "test-chat-2"
        tuner_id = "test-tuner-2"
        comp_id = "chatcmpl-both"

        # Record completion under both
        asyncio.run(
            self.storage.record_completion(
                completion_id=comp_id, chat_id=chat_id, tuner_id=tuner_id
            )
        )

        # Retrieve from both
        chat_comps = asyncio.run(self.storage.get_chat_completions(chat_id))
        tuner_comps = asyncio.run(self.storage.get_tuner_completions(tuner_id))

        self.assertEqual(chat_comps, [comp_id])
        self.assertEqual(tuner_comps, [comp_id])

    def test_record_multiple_chats_and_tuners(self):
        chat_a = "chat-a"
        chat_b = "chat-b"
        tuner_a = "tuner-a"
        tuner_b = "tuner-b"
        comp_a = "chatcmpl-a"
        comp_b = "chatcmpl-b"

        asyncio.run(
            self.storage.record_completion(
                completion_id=comp_a, chat_id=chat_a, tuner_id=tuner_a
            )
        )
        asyncio.run(
            self.storage.record_completion(
                completion_id=comp_b, chat_id=chat_b, tuner_id=tuner_b
            )
        )

        completions_chat_a = asyncio.run(self.storage.get_chat_completions(chat_a))
        completions_chat_b = asyncio.run(self.storage.get_chat_completions(chat_b))
        completions_tuner_a = asyncio.run(self.storage.get_tuner_completions(tuner_a))
        completions_tuner_b = asyncio.run(self.storage.get_tuner_completions(tuner_b))

        self.assertEqual(completions_chat_a, [comp_a])
        self.assertEqual(completions_chat_b, [comp_b])
        self.assertEqual(completions_tuner_a, [comp_a])
        self.assertEqual(completions_tuner_b, [comp_b])

    def test_ttl_duration(self):
        chat_id = "test-ttl-chat"
        tuner_id = "test-ttl-tuner"
        comp_id = "chatcmpl-ttl"

        # Record with a specific TTL
        asyncio.run(
            self.storage.record_completion(
                completion_id=comp_id,
                chat_id=chat_id,
                tuner_id=tuner_id,
                ttl_seconds=100,
            )
        )

        # Check TTL on the underlying client
        chat_key = f"chat:{chat_id}:completions"
        tuner_key = f"tuner:{tuner_id}:completions"

        chat_ttl = asyncio.run(self.storage.client.ttl(chat_key))
        tuner_ttl = asyncio.run(self.storage.client.ttl(tuner_key))

        self.assertGreater(chat_ttl, 0)
        self.assertLessEqual(chat_ttl, 100)
        self.assertGreater(tuner_ttl, 0)
        self.assertLessEqual(tuner_ttl, 100)


if __name__ == "__main__":
    unittest.main()
