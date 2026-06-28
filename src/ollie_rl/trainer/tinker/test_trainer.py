import unittest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from . import (
    TinkerTrainer,
    TinkerTrainerConfig,
)
from .trainer import (
    StaleBatchError,
    TinkerTrainerState,
    TinkerTrainerFactory,
)
from ollie_rl.trainer.types import StateStore, Example
from ollie_rl.types import ChatCompletionRequest


class InMemoryStateStore(StateStore):
    """Trivial in-memory StateStore for tests."""

    def __init__(self, initial: str | None = None):
        self._state = initial
        self.save_count = 0

    async def load(self) -> str | None:
        return self._state

    async def save(self, state: str) -> None:
        self._state = state
        self.save_count += 1


class FakeAPIFuture:
    def __init__(self, result):
        self._result = result

    def __await__(self):
        async def _async_get():
            return self._result

        return _async_get().__await__()


class TestTinkerTrainer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = TinkerTrainerConfig(
            service_url="http://test-tinker",
            api_key="test-api-key",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
        )

        # Setup mocks
        self.mock_service_client = MagicMock()
        self.mock_sampling_client = MagicMock()
        self.mock_training_client = MagicMock()

        # Setup tokenizer mock
        self.mock_tokenizer = MagicMock()

        def encode_side_effect(text, **kwargs):
            if text == "<|eot_id|>":
                return [128009]
            elif text == "<|begin_of_text|>":
                return [128000]
            return [1, 2, 3]

        self.mock_tokenizer.encode.side_effect = encode_side_effect
        self.mock_tokenizer.decode.return_value = "decoded text"
        self.mock_sampling_client.get_tokenizer.return_value = self.mock_tokenizer
        self.mock_training_client.get_tokenizer.return_value = self.mock_tokenizer

        self.state_store = InMemoryStateStore()

        self.state = TinkerTrainerState(
            sampler_path="tinker://test-run/weights/sampler-init",
            optimizer_path=None,
            train_step=0,
            sampler_step=0,
            config=self.config,
        )

        self.trainer = TinkerTrainer(
            config=self.config,
            service_client=self.mock_service_client,
            state=self.state,
            state_store=self.state_store,
            sampling_client=self.mock_sampling_client,
            training_client=self.mock_training_client,
        )

    async def test_sample_stamps_policy_generation(self):
        # Mock sample response
        mock_sequence = MagicMock()
        mock_sequence.tokens = [4, 5, 6, 128009]
        mock_sequence.stop_reason = "stop"

        mock_response = MagicMock()
        mock_response.sequences = [mock_sequence]

        # Make sample_async return the mock response
        self.mock_sampling_client.sample_async = AsyncMock(return_value=mock_response)

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        # Call sample
        sample_op = await self.trainer.sample(request)
        sample_res = await sample_op.wait()

        # Verify policy generation matches sampler_step
        self.assertEqual(sample_res.policy_generation, self.state.sampler_step)
        completion = sample_res.completion

        # Assertions
        self.assertEqual(len(completion.choices), 1)
        choice = completion.choices[0]
        self.assertEqual(choice.finish_reason, "stop")
        self.assertEqual(choice.message.content, "decoded text")
        self.assertEqual(choice.message.role, "assistant")

    async def test_sample_tool_call_response(self):
        from tinker_cookbook.renderers import ToolCall

        mock_sequence = MagicMock()
        mock_sequence.tokens = [4, 5, 6]
        mock_sequence.stop_reason = "stop"

        mock_response = MagicMock()
        mock_response.sequences = [mock_sequence]

        self.mock_sampling_client.sample_async = AsyncMock(return_value=mock_response)

        # Mock parsed message with a tool call
        mock_tool_call = ToolCall(
            id="call_123",
            function=ToolCall.FunctionBody(
                name="get_weather", arguments='{"location": "San Francisco"}'
            ),
        )
        parsed_message = {"content": None, "tool_calls": [mock_tool_call]}

        # Patch self.trainer.renderer.parse_response
        self.trainer.renderer.parse_response = MagicMock(
            return_value=(parsed_message, True)
        )

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What is the weather?"}],
            max_tokens=100,
        )

        sample_op = await self.trainer.sample(request)
        sample_res = await sample_op.wait()

        self.assertEqual(sample_res.policy_generation, self.state.sampler_step)
        completion = sample_res.completion

        self.assertEqual(len(completion.choices), 1)
        choice = completion.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        self.assertIsNone(choice.message.content)
        self.assertIsNotNone(choice.message.tool_calls)
        assert choice.message.tool_calls is not None
        self.assertEqual(len(choice.message.tool_calls), 1)

        from openai.types.chat import ChatCompletionMessageToolCall

        tc = choice.message.tool_calls[0]
        assert isinstance(tc, ChatCompletionMessageToolCall)
        self.assertEqual(tc.function.name, "get_weather")
        self.assertEqual(tc.function.arguments, '{"location": "San Francisco"}')

    async def test_sample_malformed_response_raises_not_implemented_error(self):
        mock_sequence = MagicMock()
        mock_sequence.tokens = [4, 5, 6]
        mock_sequence.stop_reason = "stop"

        mock_response = MagicMock()
        mock_response.sequences = [mock_sequence]

        self.mock_sampling_client.sample_async = AsyncMock(return_value=mock_response)

        # Mock parsed message where parse_success is False
        parsed_message = {
            "content": "some malformed string",
        }
        self.trainer.renderer.parse_response = MagicMock(
            return_value=(parsed_message, False)
        )

        # Create request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )

        sample_op = await self.trainer.sample(request)
        with self.assertRaises(NotImplementedError) as ctx:
            await sample_op.wait()

        self.assertIn(
            "Malformed assistant response or function call", str(ctx.exception)
        )

    def _make_example(
        self,
        chat_completion_id: str = "chatcmpl-1",
        advantage: float = 1.0,
        policy_generation: int = 0,
        prompt_len: int = 3,
        completion_len: int = 4,
        tokens: list[int] | None = None,
        logprobs: list[float] | None = None,
    ) -> Example:
        """Build an Example with cached tokens/logprobs for training tests."""
        if tokens is None:
            tokens = list(range(prompt_len + completion_len))
        if logprobs is None:
            logprobs = [-0.5] * completion_len
        return Example(
            chat_completion_id=chat_completion_id,
            advantage=advantage,
            policy_generation=policy_generation,
            tokens=tokens,
            logprobs=logprobs,
        )

    def _wire_training_mocks(
        self, new_sampler_path: str = "tinker://test-run/weights/sampler-step-1"
    ) -> tuple[AsyncMock, AsyncMock, AsyncMock, AsyncMock]:
        """Stub the training/sampling client async surface used by train_step."""
        fb_future = FakeAPIFuture(MagicMock())
        opt_future = FakeAPIFuture(MagicMock())
        forward_backward = AsyncMock(return_value=fb_future)
        optim_step = AsyncMock(return_value=opt_future)
        self.mock_training_client.forward_backward_async = forward_backward
        self.mock_training_client.optim_step_async = optim_step

        save_response = MagicMock()
        save_response.path = new_sampler_path
        save_future = FakeAPIFuture(save_response)
        save_weights = AsyncMock(return_value=save_future)
        self.mock_training_client.save_weights_for_sampler_async = save_weights

        new_sampling_client = MagicMock()
        new_sampling_client.get_tokenizer.return_value = self.mock_tokenizer
        create_sampling_client = AsyncMock(return_value=new_sampling_client)
        self.mock_service_client.create_sampling_client_async = create_sampling_client

        return forward_backward, optim_step, save_weights, create_sampling_client

    async def test_train_step_invokes_forward_backward_and_optim_step(self):
        forward_backward, optim_step, _, _ = self._wire_training_mocks()

        examples = [self._make_example(chat_completion_id="cmpl-1")]
        train_op = await self.trainer.train_step(examples)
        await train_op.wait()

        forward_backward.assert_awaited_once()
        optim_step.assert_awaited_once()
        # Datum list arg shape check.
        fb_args, fb_kwargs = forward_backward.call_args
        data_arg = fb_args[0]
        loss_fn_arg = fb_args[1]
        self.assertEqual(len(data_arg), 1)
        self.assertEqual(loss_fn_arg, "importance_sampling")
        self.assertEqual(fb_kwargs["loss_fn_config"], {"kl_penalty_coef": 0.0})

        # Step advanced and sampler promoted (sampler_promotion_every=1).
        self.assertEqual(self.trainer.state.train_step, 1)
        self.assertEqual(self.trainer.state.sampler_step, 1)

    async def test_train_step_promotes_sampler_on_cadence(self):
        self.trainer.config.sampler_promotion_every = 2

        # Step 1: no promotion (1 % 2 != 0).
        forward_backward, _, save_weights, _ = self._wire_training_mocks()
        await (await self.trainer.train_step([self._make_example()])).wait()
        self.assertEqual(self.trainer.state.train_step, 1)
        self.assertEqual(self.trainer.state.sampler_step, 0)
        save_weights.assert_not_awaited()

        # Step 2: promotion (2 % 2 == 0).
        forward_backward, _, save_weights, _ = self._wire_training_mocks()
        await (await self.trainer.train_step([self._make_example()])).wait()
        self.assertEqual(self.trainer.state.train_step, 2)
        self.assertEqual(self.trainer.state.sampler_step, 2)
        save_weights.assert_awaited_once()

    async def test_train_step_filters_stale_examples(self):
        forward_backward, _, _, _ = self._wire_training_mocks()

        # Set state.train_step=3, max_steps_off_policy=1.
        # Cutoff: gen < (3 - 1) = 2 → drop generation 0, keep 2 and 3.
        # 1 of 5 stale = 20%, below the default 0.4 threshold.
        self.trainer.state.train_step = 3
        self.trainer.state.sampler_step = 3
        self.trainer.config.max_steps_off_policy = 1

        examples = [
            self._make_example("cmpl-0", policy_generation=0),  # stale
            self._make_example("cmpl-1", policy_generation=2),
            self._make_example("cmpl-2", policy_generation=2),
            self._make_example("cmpl-3", policy_generation=3),
            self._make_example("cmpl-4", policy_generation=3),
        ]
        await (await self.trainer.train_step(examples)).wait()

        forward_backward.assert_awaited_once()
        data_arg = forward_backward.call_args[0][0]
        self.assertEqual(len(data_arg), 4)
        # train_step still advances (we ran forward_backward + optim_step).
        self.assertEqual(self.trainer.state.train_step, 4)

    async def test_train_step_raises_when_too_many_stale(self):
        forward_backward, optim_step, save_weights, _ = self._wire_training_mocks()

        self.trainer.state.train_step = 10
        self.trainer.state.sampler_step = 10
        self.trainer.config.max_steps_off_policy = 0
        self.trainer.config.max_stale_fraction = 0.4

        # All stale → 100% stale, well above the 0.4 threshold.
        examples = [
            self._make_example("cmpl-old", policy_generation=5),
            self._make_example("cmpl-old2", policy_generation=7),
        ]
        with self.assertRaises(StaleBatchError):
            await self.trainer.train_step(examples)

        forward_backward.assert_not_awaited()
        optim_step.assert_not_awaited()
        save_weights.assert_not_awaited()
        # State must NOT advance when train_step rejects the batch.
        self.assertEqual(self.trainer.state.train_step, 10)
        self.assertEqual(self.trainer.state.sampler_step, 10)

    async def test_train_step_tolerates_stale_below_threshold(self):
        forward_backward, _, _, _ = self._wire_training_mocks()

        # 1 of 5 stale → 20% stale, below 0.4 threshold; should proceed.
        self.trainer.state.train_step = 3
        self.trainer.state.sampler_step = 3
        self.trainer.config.max_steps_off_policy = 1
        self.trainer.config.max_stale_fraction = 0.4

        examples = [
            self._make_example("cmpl-0", policy_generation=0),  # stale (cutoff=2)
            self._make_example("cmpl-1", policy_generation=2),
            self._make_example("cmpl-2", policy_generation=3),
            self._make_example("cmpl-3", policy_generation=3),
            self._make_example("cmpl-4", policy_generation=3),
        ]
        await (await self.trainer.train_step(examples)).wait()

        forward_backward.assert_awaited_once()
        data_arg = forward_backward.call_args[0][0]
        self.assertEqual(len(data_arg), 4)

    async def test_train_step_raises_on_empty_batch(self):
        with self.assertRaises(StaleBatchError):
            await self.trainer.train_step([])

    async def test_train_step_skips_examples_without_cached_tokens(self):
        forward_backward, _, _, _ = self._wire_training_mocks()

        # Two examples with cached tokens (fresh), one without (also
        # fresh — staleness fraction stays at 0, well below threshold).
        # The cache-less example is silently skipped during Datum
        # construction; the others train normally.
        good_a = self._make_example("cmpl-good-a")
        good_b = self._make_example("cmpl-good-b")
        bad = Example(
            chat_completion_id="cmpl-bad",
            advantage=1.0,
            policy_generation=0,
            tokens=None,
            logprobs=None,
        )
        await (await self.trainer.train_step([good_a, good_b, bad])).wait()

        forward_backward.assert_awaited_once()
        data_arg = forward_backward.call_args[0][0]
        self.assertEqual(len(data_arg), 2)

    async def test_train_step_raises_when_all_examples_lack_tokens(self):
        forward_backward, _, _, _ = self._wire_training_mocks()

        bad_a = Example(
            chat_completion_id="cmpl-bad-a",
            advantage=1.0,
            policy_generation=0,
            tokens=None,
            logprobs=None,
        )
        bad_b = Example(
            chat_completion_id="cmpl-bad-b",
            advantage=1.0,
            policy_generation=0,
            tokens=None,
            logprobs=None,
        )
        with self.assertRaises(StaleBatchError):
            await self.trainer.train_step([bad_a, bad_b])
        forward_backward.assert_not_awaited()

    async def test_open_bootstrap_path(self):
        fresh_store = InMemoryStateStore()

        # Mock ServiceClient creation and training client creation
        mock_save_response = MagicMock()
        mock_save_response.path = "tinker://new-run/weights/sampler-init"

        mock_future = FakeAPIFuture(mock_save_response)

        self.mock_training_client.save_weights_for_sampler_async = AsyncMock(
            return_value=mock_future
        )
        self.mock_service_client.create_lora_training_client_async = AsyncMock(
            return_value=self.mock_training_client
        )
        self.mock_service_client.create_sampling_client_async = AsyncMock(
            return_value=self.mock_sampling_client
        )

        factory = TinkerTrainerFactory()
        with patch("tinker.ServiceClient", return_value=self.mock_service_client):
            trainer = await factory.create(
                name="test-tuner",
                state_store=fresh_store,
            )

        self.assertEqual(
            trainer.state.sampler_path, "tinker://new-run/weights/sampler-init"
        )
        self.assertEqual(trainer.state.train_step, 0)
        self.assertEqual(fresh_store.save_count, 1)

        # Verify the state is correctly serialized as JSON
        assert fresh_store._state is not None
        state_data = json.loads(fresh_store._state)
        self.assertEqual(
            state_data["sampler_path"], "tinker://new-run/weights/sampler-init"
        )

    async def test_open_restore_path(self):
        # Pre-seed state dict
        state_dict = {
            "sampler_path": "tinker://restored-run/weights/sampler-checkpoint",
            "optimizer_path": "tinker://restored-run/weights/optimizer-checkpoint",
            "train_step": 10,
            "sampler_step": 8,
            "config": {
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "lora_rank": 32,
                "learning_rate": 1e-5,
                "kl_penalty_coef": 0.0,
                "loss_fn": "importance_sampling",
            },
        }
        seeded_store = InMemoryStateStore(initial=json.dumps(state_dict))

        self.mock_service_client.create_training_client_from_state_with_optimizer_async = AsyncMock(
            return_value=self.mock_training_client
        )
        self.mock_service_client.create_sampling_client_async = AsyncMock(
            return_value=self.mock_sampling_client
        )

        factory = TinkerTrainerFactory()
        with patch("tinker.ServiceClient", return_value=self.mock_service_client):
            trainer = await factory.restore(
                name="test-tuner",
                state_store=seeded_store,
            )

        self.assertEqual(
            trainer.state.sampler_path,
            "tinker://restored-run/weights/sampler-checkpoint",
        )
        self.assertEqual(
            trainer.state.optimizer_path,
            "tinker://restored-run/weights/optimizer-checkpoint",
        )
        self.assertEqual(trainer.state.train_step, 10)
        self.assertEqual(trainer.state.sampler_step, 8)

        # Restore path must not overwrite the existing state store
        self.assertEqual(seeded_store.save_count, 0)
