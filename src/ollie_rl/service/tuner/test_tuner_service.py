"""
Unit tests for TunerService public methods.

Uses an in-memory SQLite database (via init_db) so no mocking of the DB layer
is required. The Trainer / TrainerFactory are mocked since they hit external
backends.
"""

import unittest
from datetime import timedelta
from typing import List, Literal, Optional
from unittest.mock import AsyncMock

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from ollie_rl.db.connection import init_db, shutdown_db
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner import (
    EmptyRunError,
    RewardAlreadySetError,
    RunExpiredError,
    RunNotFoundError,
    TunerNotFoundError,
    TunerService,
)
from ollie_rl.trainer import Sample, StateStore, Trainer, TrainerFactory
from ollie_rl.trainer import factory as trainer_factory
from ollie_rl.types import ChatCompletionRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RECIPE = "grpo_16x32"
_TRAINER_KIND = "mock"
FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


def _make_chat_completion(
    completion_id: str = "cmpl-test", finish_reason: FinishReason = "stop"
) -> ChatCompletion:
    return ChatCompletion(
        id=completion_id,
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        created=0,
        model="fake-model",
        object="chat.completion",
    )


def _make_sample_op(
    completion_id: str = "cmpl-test",
    policy_generation: int = 0,
    finish_reason: FinishReason = "stop",
):
    sample = Sample(
        completion=_make_chat_completion(completion_id, finish_reason),
        policy_generation=policy_generation,
    )

    op = AsyncMock()
    op.wait = AsyncMock(return_value=sample)
    op.peek = AsyncMock(return_value=True)
    # `save_state()` is a synchronous accessor; a non-resumable op returns None
    # so the service never persists an in-flight row for it.
    op.save_state = lambda: None
    return op


class FakeTrainer(Trainer):
    """Lightweight in-process Trainer that never calls any remote backend."""

    def __init__(self):
        self._sample_op = _make_sample_op()

    @property
    def policy_generation(self) -> int:
        return 0

    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ):
        return self._sample_op

    async def train_step(self, examples, *, sampler_promotion_every: int = 1):
        op = AsyncMock()
        op.wait = AsyncMock(return_value=None)
        op.peek = AsyncMock(return_value=True)
        return op

    async def create_sampler(self, checkpoint):
        # The test suite only exercises live-policy sampling, so the frozen
        # checkpoint path is never routed here; return self (a Trainer is a
        # Sampler) to satisfy the now-abstract contract.
        return self


class FakeTrainerFactory(TrainerFactory):
    async def create(
        self,
        name: str,
        state_store: StateStore,
        trainer_params: Optional[dict] = None,
    ) -> Trainer:
        return FakeTrainer()

    async def restore(
        self,
        name: str,
        state_store: StateStore,
    ) -> Trainer:
        return FakeTrainer()


# ---------------------------------------------------------------------------
# Test base
# ---------------------------------------------------------------------------


class TunerServiceTestCase(unittest.IsolatedAsyncioTestCase):
    """
    Each test gets a fresh in-memory SQLite database and a fresh TunerService.

    The global DB singletons (_engine / _sessionmaker) are reset between tests
    via shutdown_db / init_db so there is no cross-test state.
    """

    async def asyncSetUp(self):
        # Always use a fresh in-memory database per test.
        await init_db()

        # Register our fake trainer factory (idempotent if already registered).
        if _TRAINER_KIND not in trainer_factory._REGISTRY:
            trainer_factory.register(_TRAINER_KIND, FakeTrainerFactory())

        self.service = TunerService()

    async def asyncTearDown(self):
        await shutdown_db()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def _create_tuner(self, datum_ids: Optional[List[str]] = None) -> str:
        return await self.service.create_tuner(
            recipe=_RECIPE,
            name="test-tuner",
            train_datum_ids=datum_ids or ["datum-1", "datum-2"],
            trainer=_TRAINER_KIND,
        )

    async def _add_run(
        self,
        tuner_id: str,
        datum_id: str = "datum-1",
        reward: Optional[float] = None,
        trained_count: int = 0,
        expired: bool = False,
    ) -> RunModel:

        from ollie_rl.db.connection import get_sessionmaker

        async_session = get_sessionmaker()
        async with async_session() as session:
            async with session.begin():
                expires_at = (
                    utcnow() - timedelta(hours=1)
                    if expired
                    else utcnow() + timedelta(hours=2)
                )
                run = RunModel(
                    tuner_id=tuner_id,
                    datum_id=datum_id,
                    reward=reward,
                    trained_count=trained_count,
                    expires_at=expires_at,
                )
                session.add(run)
                await session.flush()
                run_id = run.id

        # Re-fetch to get a detached copy.
        async with async_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(RunModel).where(RunModel.id == run_id)
            )
            return result.scalar_one()

    async def _record_completion(
        self,
        tuner_id: str,
        run_id: str,
        datum_id: str = "datum-1",
        completion_id: Optional[str] = None,
        policy_generation: int = 0,
    ) -> str:
        """Persist a single chat completion so a run is no longer 'empty'."""
        completion_id = completion_id or f"cmpl-{run_id}"
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        await self.service.record_chat_completion(
            completion_id=completion_id,
            tuner_id=tuner_id,
            run_id=run_id,
            datum_id=datum_id,
            policy_generation=policy_generation,
            request=request,
            response=_make_chat_completion(completion_id=completion_id),
            duration_ms=0,
        )
        return completion_id


# ---------------------------------------------------------------------------
# create_tuner
# ---------------------------------------------------------------------------


class TestCreateTuner(TunerServiceTestCase):
    async def test_returns_tuner_id(self):
        tuner_id = await self._create_tuner()
        self.assertIsNotNone(tuner_id)
        self.assertTrue(tuner_id.startswith("tuner_"))

    async def test_trainer_registered_in_memory(self):
        tuner_id = await self._create_tuner()
        self.assertIn(tuner_id, self.service.active_trainers)

    async def test_unknown_recipe_raises(self):
        with self.assertRaises(ValueError):
            await self.service.create_tuner(
                recipe="nonexistent_recipe",
                name="bad",
                train_datum_ids=["d1"],
                trainer=_TRAINER_KIND,
            )

    async def test_unknown_trainer_raises(self):
        with self.assertRaises(ValueError):
            await self.service.create_tuner(
                recipe=_RECIPE,
                name="bad",
                train_datum_ids=["d1"],
                trainer="nonexistent_trainer",
            )


# ---------------------------------------------------------------------------
# dispense_run
# ---------------------------------------------------------------------------


class TestDispenseRun(TunerServiceTestCase):
    async def test_raises_for_unknown_tuner(self):
        with self.assertRaises(TunerNotFoundError):
            await self.service.dispense_run("tuner_unknown")

    async def test_dispense_run_when_trainer_is_training(self):
        from ollie_rl.cookbook import RECIPES
        from ollie_rl.cookbook.recipes import Recipe

        RECIPES["test_async"] = Recipe(
            group_size=16,
            num_groups_per_batch=32,
        )

        tuner_id = await self.service.create_tuner(
            recipe="test_async",
            name="test-tuner-async",
            train_datum_ids=["datum-1", "datum-2"],
            trainer=_TRAINER_KIND,
        )

        trainer = self.service.active_trainers[tuner_id]
        assert isinstance(trainer, FakeTrainer)

        result = await self.service.dispense_run(tuner_id)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.datum_id, "datum-1")

    async def test_returns_dispense_run_with_valid_fields(self):
        tuner_id = await self._create_tuner(datum_ids=["d1", "d2"])
        result = await self.service.dispense_run(tuner_id)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn(result.datum_id, ["d1", "d2"])
        self.assertTrue(result.run_id.startswith("run_"))
        self.assertGreater(result.expires_at, utcnow())

    async def test_finishes_started_group_before_fresh_datum(self):
        """A started-but-incomplete group is preferred over a fresh datum.

        The scheduler is greedy "most-full-first": it drives an in-progress
        group to completion before starting a new distinct group.
        """
        tuner_id = await self._create_tuner(datum_ids=["d1", "d2"])
        # d1 already has a pending run (started group); d2 has none.
        await self._add_run(tuner_id, datum_id="d1")

        result = await self.service.dispense_run(tuner_id)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.datum_id, "d1")


# ---------------------------------------------------------------------------
# update_reward
# ---------------------------------------------------------------------------


class TestUpdateReward(TunerServiceTestCase):
    async def test_raises_for_unknown_run(self):
        tuner_id = await self._create_tuner()
        with self.assertRaises(RunNotFoundError):
            await self.service.update_reward(tuner_id, "run_unknown", 1.0)

    async def test_raises_for_expired_run(self):
        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id, expired=True)
        with self.assertRaises(RunExpiredError):
            await self.service.update_reward(tuner_id, run.id, 0.5)

    async def test_raises_when_reward_already_set(self):
        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id, reward=0.8)
        with self.assertRaises(RewardAlreadySetError):
            await self.service.update_reward(tuner_id, run.id, 0.5)

    async def test_raises_for_run_without_completions(self):
        # A run with zero recorded chat completions carries no training
        # signal, so rewarding it is rejected outright.
        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)
        with self.assertRaises(EmptyRunError):
            await self.service.update_reward(tuner_id, run.id, 0.75)

    async def test_empty_run_reward_not_persisted(self):
        from sqlalchemy import select

        from ollie_rl.db.connection import get_sessionmaker

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)
        with self.assertRaises(EmptyRunError):
            await self.service.update_reward(tuner_id, run.id, 0.75)

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(RunModel).where(RunModel.id == run.id)
            )
            updated = result.scalar_one()
        # The rejected reward must not be persisted.
        self.assertIsNone(updated.reward)

    async def test_sets_reward_successfully(self):
        from sqlalchemy import select

        from ollie_rl.db.connection import get_sessionmaker

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)
        await self._record_completion(tuner_id, run.id, run.datum_id)
        await self.service.update_reward(tuner_id, run.id, 0.75)

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(RunModel).where(RunModel.id == run.id)
            )
            updated = result.scalar_one()
        self.assertEqual(updated.reward, 0.75)

    async def test_raises_when_run_belongs_to_different_tuner(self):
        tuner_id_a = await self._create_tuner()
        tuner_id_b = await self._create_tuner()
        run = await self._add_run(tuner_id_a)
        with self.assertRaises(RunNotFoundError):
            await self.service.update_reward(tuner_id_b, run.id, 1.0)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


class TestSample(TunerServiceTestCase):
    def _make_request(self) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    async def test_raises_for_unknown_tuner(self):
        with self.assertRaises(TunerNotFoundError):
            await self.service.sample("tuner_unknown", self._make_request())

    async def test_returns_completion_without_run_id(self):
        tuner_id = await self._create_tuner()
        completion = await self.service.sample(tuner_id, self._make_request())
        self.assertIsNotNone(completion)
        self.assertEqual(completion.object, "chat.completion")

    async def test_raises_for_unknown_run_id(self):
        tuner_id = await self._create_tuner()
        with self.assertRaises(RunNotFoundError):
            await self.service.sample(
                tuner_id, self._make_request(), run_id="run_unknown"
            )

    async def test_raises_for_expired_run(self):
        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id, expired=True)
        with self.assertRaises(RunExpiredError):
            await self.service.sample(tuner_id, self._make_request(), run_id=run.id)

    async def test_raises_when_reward_already_set(self):
        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id, reward=1.0)
        with self.assertRaises(RewardAlreadySetError):
            await self.service.sample(tuner_id, self._make_request(), run_id=run.id)

    async def test_records_chat_completion_when_run_id_provided(self):
        from sqlalchemy import select

        from ollie_rl.db.connection import get_sessionmaker
        from ollie_rl.db.models import ChatCompletionModel

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)

        # Make FakeTrainer return a deterministic completion id.
        trainer = self.service.active_trainers[tuner_id]
        assert isinstance(trainer, FakeTrainer)
        trainer._sample_op = _make_sample_op(completion_id="cmpl-recorded")

        req = self._make_request()
        await self.service.sample(tuner_id, req, run_id=run.id)

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(ChatCompletionModel.run_id == run.id)
            )
            record = result.scalar_one_or_none()

        self.assertIsNotNone(record)
        assert record is not None
        self.assertTrue(record.id.startswith("cmpl_"))
        self.assertEqual(record.tuner_id, tuner_id)
        self.assertIsNotNone(record.request)
        self.assertEqual(record.request["model"], "fake-model")
        self.assertIsNotNone(record.response)
        self.assertEqual(record.response["id"], "cmpl-recorded")

    async def test_raises_content_filter_sample_error_and_sets_penalty_reward(self):
        from sqlalchemy import select
        from ollie_rl.db.connection import get_sessionmaker
        from ollie_rl.db.models import RunModel
        from ollie_rl.service.tuner import ContentFilterSampleError

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)

        # Make FakeTrainer return a content-filtered sample.
        trainer = self.service.active_trainers[tuner_id]
        assert isinstance(trainer, FakeTrainer)
        trainer._sample_op = _make_sample_op(
            completion_id="cmpl-content-filter",
            finish_reason="content_filter",
        )

        req = self._make_request()
        with self.assertRaises(ContentFilterSampleError) as ctx:
            await self.service.sample(tuner_id, req, run_id=run.id)

        self.assertIn("Content-filtered sample on run", str(ctx.exception))

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(RunModel).where(RunModel.id == run.id)
            )
            record = result.scalar_one()
            self.assertEqual(record.reward, -1.0)  # default content_filter_penalty


# ---------------------------------------------------------------------------
# record_chat_completion
# ---------------------------------------------------------------------------


class TestRecordChatCompletion(TunerServiceTestCase):
    async def test_persists_record(self):
        from sqlalchemy import select

        from ollie_rl.db.connection import get_sessionmaker
        from ollie_rl.db.models import ChatCompletionModel

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)

        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        response = _make_chat_completion(completion_id="cmpl-direct")

        await self.service.record_chat_completion(
            completion_id="cmpl-direct",
            tuner_id=tuner_id,
            run_id=run.id,
            datum_id=run.datum_id,
            policy_generation=1,
            request=request,
            response=response,
            duration_ms=123,
        )

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(
                    ChatCompletionModel.id == "cmpl-direct"
                )
            )
            record = result.scalar_one_or_none()

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.policy_generation, 1)
        self.assertEqual(record.datum_id, run.datum_id)
        # No tokens/logprobs supplied → columns stay NULL.
        self.assertIsNone(record.tokens)
        self.assertIsNone(record.logprobs)

    async def test_persists_tokens_and_logprobs(self):
        from sqlalchemy import select

        from ollie_rl.db.connection import get_sessionmaker
        from ollie_rl.db.models import ChatCompletionModel

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)

        tokens = [10, 11, 12, 13, 14]
        logprobs = [-0.1, -0.2, -0.3]
        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        response = _make_chat_completion(completion_id="cmpl-with-tokens")

        await self.service.record_chat_completion(
            completion_id="cmpl-with-tokens",
            tuner_id=tuner_id,
            run_id=run.id,
            datum_id=run.datum_id,
            policy_generation=7,
            tokens=tokens,
            logprobs=logprobs,
            request=request,
            response=response,
            duration_ms=456,
        )

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(
                    ChatCompletionModel.id == "cmpl-with-tokens"
                )
            )
            record = result.scalar_one_or_none()

        # The model-layer `_PackedIntList` / `_PackedFloatList` type
        # decorators transparently round-trip the BLOB representation,
        # so the columns surface as plain Python lists.
        assert record is not None
        self.assertEqual(record.tokens, tokens)
        assert record.logprobs is not None
        self.assertEqual(len(record.logprobs), len(logprobs))
        for got, want in zip(record.logprobs, logprobs):
            self.assertAlmostEqual(got, want, places=5)

    async def test_persists_request_and_response(self):
        from sqlalchemy import select

        from ollie_rl.db.connection import get_sessionmaker
        from ollie_rl.db.models import ChatCompletionModel

        tuner_id = await self._create_tuner()
        run = await self._add_run(tuner_id)

        request = ChatCompletionRequest(
            model="fake-model",
            messages=[{"role": "user", "content": "hello world"}],
        )
        response = _make_chat_completion(completion_id="cmpl-with-req-resp")

        await self.service.record_chat_completion(
            completion_id="cmpl-with-req-resp",
            tuner_id=tuner_id,
            run_id=run.id,
            datum_id=run.datum_id,
            policy_generation=3,
            request=request,
            response=response,
            duration_ms=789,
        )

        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(
                    ChatCompletionModel.id == "cmpl-with-req-resp"
                )
            )
            record = result.scalar_one_or_none()

        assert record is not None
        self.assertIsNotNone(record.request)
        self.assertEqual(record.request["model"], "fake-model")
        self.assertEqual(record.request["messages"][0]["content"], "hello world")
        self.assertIsNotNone(record.response)
        self.assertEqual(record.response["id"], "cmpl-with-req-resp")


# ---------------------------------------------------------------------------
# maybe_train
# ---------------------------------------------------------------------------


class TestMaybeTrain(TunerServiceTestCase):
    """
    maybe_train only triggers a training step when enough reward-labeled groups
    are available. We set up the DB rows directly to exercise this logic.
    """

    async def test_raise_for_unknown_tuner(self):
        """Should raise when the trainer cannot be found."""
        with self.assertRaises(TunerNotFoundError):
            await self.service._maybe_train("tuner_unknown")

    async def test_no_op_when_not_enough_groups(self):
        """With too few completed groups, training is not triggered."""
        tuner_id = await self._create_tuner(datum_ids=["d1"])
        trainer = self.service.active_trainers[tuner_id]
        assert isinstance(trainer, FakeTrainer)
        # Spy on train_step.
        original_train_step = trainer.train_step
        called = []

        async def spy_train_step(examples, *, sampler_promotion_every: int = 1):
            called.append(examples)
            return await original_train_step(
                examples, sampler_promotion_every=sampler_promotion_every
            )

        trainer.train_step = spy_train_step  # type: ignore

        # Add only a single rewarded run (group_size=16, target=32 required).
        await self._add_run(tuner_id, datum_id="d1", reward=1.0)
        await self.service._maybe_train(tuner_id)

        self.assertEqual(called, [])

    async def test_train_step_receives_policy_generation(self):
        from ollie_rl.cookbook import RECIPES
        from ollie_rl.cookbook.recipes import Recipe

        # Register a small test recipe
        RECIPES["test_2x2"] = Recipe(
            group_size=2,
            num_groups_per_batch=2,
        )

        tuner_id = await self.service.create_tuner(
            recipe="test_2x2",
            name="test-tuner-small",
            train_datum_ids=["d1", "d2"],
            trainer=_TRAINER_KIND,
        )

        trainer = self.service.active_trainers[tuner_id]
        assert isinstance(trainer, FakeTrainer)

        called_examples = []
        original_train_step = trainer.train_step

        async def spy_train_step(examples, *, sampler_promotion_every: int = 1):
            called_examples.extend(examples)
            return await original_train_step(
                examples, sampler_promotion_every=sampler_promotion_every
            )

        trainer.train_step = spy_train_step  # type: ignore

        # Set up 4 runs (2 groups of size 2)
        runs = []
        for datum_id in ["d1", "d2"]:
            for i in range(2):
                run = await self._add_run(tuner_id, datum_id=datum_id)
                runs.append(run)
                request = ChatCompletionRequest(
                    model="fake-model",
                    messages=[{"role": "user", "content": "hello"}],
                )
                response = _make_chat_completion(completion_id=f"cmpl-{datum_id}-{i}")
                await self.service.record_chat_completion(
                    completion_id=f"cmpl-{datum_id}-{i}",
                    tuner_id=tuner_id,
                    run_id=run.id,
                    datum_id=datum_id,
                    policy_generation=i,
                    request=request,
                    response=response,
                    duration_ms=0,
                )
                await self.service.update_reward(tuner_id, run.id, 1.0)

        # Trigger maybe_train
        await self.service._maybe_train(tuner_id)

        # Verify that train_step was called with the correct policy_generation
        self.assertEqual(len(called_examples), 4)
        for example in called_examples:
            self.assertIsNotNone(example.policy_generation)
            self.assertIn(example.policy_generation, [0, 1])


if __name__ == "__main__":
    unittest.main()
