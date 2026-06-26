import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
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
            candidates={"candidate_1": candidate},
            usage_metadata=None,
            train_step_id="step-123",
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
        sample_op = await self.job.sample(request)
        sample_res = await sample_op.wait()
        self.assertEqual(sample_res.step_id, "step-123")
        completion = sample_res.completion

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
            candidates={"candidate_2": candidate},
            usage_metadata=None,
            train_step_id="456",
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
        sample_op = await self.job.sample(request)
        sample_res = await sample_op.wait()
        self.assertEqual(sample_res.step_id, "456")
        completion = sample_res.completion

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
            candidates={"candidate_3": candidate},
            usage_metadata=None,
            train_step_id="step-789",
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
        sample_op = await self.job.sample(request)
        sample_res = await sample_op.wait()
        self.assertEqual(sample_res.step_id, "step-789")
        completion = sample_res.completion

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

    async def test_train_step_successful(self):
        from gemini_msrl.types import TrainStepResponse
        from ollie_rl.cookbook.types import Example

        # Mock TrainStepResponse with completed_train_step_id
        response_payload = TrainStepResponse(completed_train_step_id="step-12345")

        mock_op = MagicMock()
        mock_op.name = "operation-train-step"
        self.mock_client.train_step.return_value = mock_op

        mock_completed_op = MagicMock()
        mock_completed_op.get_response_as.return_value = response_payload
        self.mock_client.wait_for_operation.return_value = mock_completed_op

        examples = [Example(chat_completion_id="chatcmpl-1", advantage=1.0)]

        # Call train_step
        train_op = await self.job.train_step(examples)
        await train_op.wait()

        # Assert state persistence works
        state = await self.job.save_state()
        self.assertIn('"tuning_job_name"', state)

    async def test_restore_tuner(self):
        from ollie_rl.cookbook.gemini_msrl import GeminiMsrlRecipe
        import json

        # Create a state string
        state_dict = {
            "tuning_job_name": "projects/test-project/locations/us-central1/tuningJobs/test-job-id",
        }
        state_str = json.dumps(state_dict)

        recipe = GeminiMsrlRecipe()
        # We need to mock wait_for_tuning_job_running since restore calls it
        self.mock_client.wait_for_tuning_job_running = AsyncMock()

        # Let's patch GeminiMsrlClient to return our mock_client
        with patch(
            "ollie_rl.cookbook.gemini_msrl.GeminiMsrlClient"
        ) as mock_client_class:
            mock_client_class.return_value = self.mock_client
            tuner = await recipe.restore(state_str)
            self.assertEqual(
                tuner.tuning_job_name,
                "projects/test-project/locations/us-central1/tuningJobs/test-job-id",
            )

    async def test_sample_op_peek(self):
        from gemini_msrl.types import Operation

        # Create a mock operation that is not done
        mock_op_pending = Operation(name="operation-123", done=False)
        # Create a mock operation that is done
        mock_op_done = Operation(name="operation-123", done=True)

        self.mock_client.get_operation.side_effect = [mock_op_pending, mock_op_done]

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatCompletionMessage(role="assistant", content="Hi")],
            max_tokens=100,
        )

        mock_op = MagicMock()
        mock_op.name = "operation-123"
        self.mock_client.generate_content_tuning_scope.return_value = mock_op

        # Call sample
        sample_op = await self.job.sample(request)

        # First peek should be False
        is_done_1 = await sample_op.peek()
        self.assertFalse(is_done_1)

        # Second peek should be True
        is_done_2 = await sample_op.peek()
        self.assertTrue(is_done_2)

        # Verify get_operation was called with correct name
        self.mock_client.get_operation.assert_has_calls(
            [
                call("operation-123"),
                call("operation-123"),
            ]
        )

    async def test_train_op_peek(self):
        from gemini_msrl.types import Operation
        from ollie_rl.cookbook.types import Example

        mock_op_pending = Operation(name="operation-train-step", done=False)
        mock_op_done = Operation(name="operation-train-step", done=True)

        self.mock_client.get_operation.side_effect = [mock_op_pending, mock_op_done]

        mock_op = MagicMock()
        mock_op.name = "operation-train-step"
        self.mock_client.train_step.return_value = mock_op

        examples = [Example(chat_completion_id="chatcmpl-1", advantage=1.0)]

        # Call train_step
        train_op = await self.job.train_step(examples)

        # First peek should be False
        is_done_1 = await train_op.peek()
        self.assertFalse(is_done_1)

        # Second peek should be True
        is_done_2 = await train_op.peek()
        self.assertTrue(is_done_2)

        # Verify get_operation was called with correct name
        self.mock_client.get_operation.assert_has_calls(
            [
                call("operation-train-step"),
                call("operation-train-step"),
            ]
        )
