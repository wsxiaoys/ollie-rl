import unittest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from ollie_rl.server.app import app, storage


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_chat_completion_with_trajectory_id(self):
        # Create a mock tuner
        mock_tuner = MagicMock()

        # We need an async sample method
        mock_tuner.sample = AsyncMock()

        # Mock the return value of sample
        mock_completion = ChatCompletion(
            id="chatcmpl-123",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello world",
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

        # Patch storage.get to return our mock_tuner using patch.object
        with patch.object(storage, "get", return_value=mock_tuner) as mock_get:
            # Make request with header
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "assistant", "content": "Hi"}],
                },
                headers={"x-ollie-trajectory-id": "traj-456"},
            )

            # Assertions
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["id"], "chatcmpl-123")
            self.assertEqual(data["choices"][0]["message"]["content"], "Hello world")

            # Verify mock_tuner.sample was called
            mock_tuner.sample.assert_called_once()

            # Verify storage.get was called with correct model
            mock_get.assert_called_once_with("test-model")

    @patch("ollie_rl.server.app.Cookbook.create", new_callable=AsyncMock)
    @patch("ollie_rl.server.app.storage.register_tuner", new_callable=AsyncMock)
    def test_create_tuner(self, mock_register_tuner, mock_cookbook_create):
        mock_tuner = MagicMock()
        mock_cookbook_create.return_value = mock_tuner

        response = self.client.post(
            "/v1/tuners",
            json={
                "tuner_id": "test-tuner",
                "recipe": "gemini_msrl",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["tuner_id"], "test-tuner")
        self.assertEqual(data["recipe"], "gemini_msrl")

        mock_cookbook_create.assert_called_once_with(
            "gemini_msrl", tuner_id="test-tuner"
        )
        mock_register_tuner.assert_called_once_with("test-tuner", mock_tuner)
