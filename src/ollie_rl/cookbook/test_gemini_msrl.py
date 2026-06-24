import unittest
from unittest.mock import AsyncMock, MagicMock
from typing import cast
from ollie_rl.cookbook.gemini_msrl import GeminiMsrlTuner, GeminiMsrlRecipeConfig
from ollie_rl.types import ChatCompletionRequest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from google.genai.types import Candidate, Content, Part, FunctionCall, FinishReason
from gemini_msrl.types import GenerateContentTuningScopeResponse


class TestGeminiMsrlTuner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = GeminiMsrlRecipeConfig(
            auth_token="test-token",
            project_id="test-project",
        )
        self.mock_client = AsyncMock()
        self.job = GeminiMsrlTuner(
            model_id="test-model",
            config=self.config,
            client=self.mock_client,
            tuning_job_name="projects/test-project/locations/us-central1/tuningJobs/test-job-id",
        )

    async def test_sample_text_response(self):
        # Mock LRO response with text
        candidate = Candidate(
            content=Content(
                role="model", parts=[Part(text="Hello, how can I help you today?")]
            ),
            finish_reason=FinishReason.STOP,
        )
        response_payload = GenerateContentTuningScopeResponse(
            candidates={"candidate_1": candidate}, usage_metadata=None
        )

        mock_op = MagicMock()
        mock_op.name = "operation-123"
        self.mock_client.generate_content_tuning_scope.return_value = mock_op

        mock_completed_op = MagicMock()
        mock_completed_op.get_response_as.return_value = response_payload
        self.mock_client.wait_for_operation.return_value = mock_completed_op

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatCompletionMessage(role="assistant", content="Hi")],
            max_tokens=100,
        )

        # Call sample
        completion = await self.job.sample(request)

        # Assertions
        self.assertEqual(completion.id, "candidate_1")
        self.assertEqual(len(completion.choices), 1)
        choice = completion.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.message.content, "Hello, how can I help you today?")
        self.assertEqual(choice.message.role, "assistant")
        self.assertIsNone(choice.message.tool_calls)

    async def test_sample_tool_call_response(self):
        # Mock LRO response with a function call
        candidate = Candidate(
            content=Content(
                role="model",
                parts=[
                    Part(
                        function_call=FunctionCall(
                            id="call-abc",
                            name="get_weather",
                            args={"location": "San Francisco, CA"},
                        )
                    )
                ],
            ),
            finish_reason=FinishReason.STOP,
        )
        response_payload = GenerateContentTuningScopeResponse(
            candidates={"candidate_2": candidate}, usage_metadata=None
        )

        mock_op = MagicMock()
        mock_op.name = "operation-456"
        self.mock_client.generate_content_tuning_scope.return_value = mock_op

        mock_completed_op = MagicMock()
        mock_completed_op.get_response_as.return_value = response_payload
        self.mock_client.wait_for_operation.return_value = mock_completed_op

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionMessage(
                    role="assistant", content="What is the weather in SF?"
                )
            ],
            max_tokens=100,
        )

        # Call sample
        completion = await self.job.sample(request)

        # Assertions
        self.assertEqual(completion.id, "candidate_2")
        self.assertEqual(len(completion.choices), 1)
        choice = completion.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        self.assertEqual(choice.message.role, "assistant")
        self.assertIsNotNone(choice.message.tool_calls)
        assert choice.message.tool_calls is not None
        self.assertEqual(len(choice.message.tool_calls), 1)

        tc = cast(ChatCompletionMessageToolCall, choice.message.tool_calls[0])
        self.assertEqual(tc.id, "call-abc")
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.function.name, "get_weather")
        self.assertEqual(tc.function.arguments, '{"location": "San Francisco, CA"}')

    async def test_sample_mixed_text_and_tool_call_response(self):
        # Mock LRO response with mixed text and function call
        candidate = Candidate(
            content=Content(
                role="model",
                parts=[
                    Part(text="Sure, let me check that for you."),
                    Part(
                        function_call=FunctionCall(
                            id="call-xyz",
                            name="get_weather",
                            args={"location": "Seattle, WA"},
                        )
                    ),
                ],
            ),
            finish_reason=FinishReason.STOP,
        )
        response_payload = GenerateContentTuningScopeResponse(
            candidates={"candidate_3": candidate}, usage_metadata=None
        )

        mock_op = MagicMock()
        mock_op.name = "operation-789"
        self.mock_client.generate_content_tuning_scope.return_value = mock_op

        mock_completed_op = MagicMock()
        mock_completed_op.get_response_as.return_value = response_payload
        self.mock_client.wait_for_operation.return_value = mock_completed_op

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                ChatCompletionMessage(
                    role="assistant", content="Check weather in Seattle"
                )
            ],
            max_tokens=100,
        )

        # Call sample
        completion = await self.job.sample(request)

        # Assertions
        self.assertEqual(completion.id, "candidate_3")
        self.assertEqual(len(completion.choices), 1)
        choice = completion.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertEqual(choice.message.content, "Sure, let me check that for you.")
        self.assertEqual(choice.message.role, "assistant")
        self.assertIsNotNone(choice.message.tool_calls)
        assert choice.message.tool_calls is not None
        self.assertEqual(len(choice.message.tool_calls), 1)

        tc = cast(ChatCompletionMessageToolCall, choice.message.tool_calls[0])
        self.assertEqual(tc.id, "call-xyz")
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.function.name, "get_weather")
        self.assertEqual(tc.function.arguments, '{"location": "Seattle, WA"}')
