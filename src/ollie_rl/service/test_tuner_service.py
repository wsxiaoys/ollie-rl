import math
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy import select

from ollie_rl.cookbook.types import Example, Tuner, TrainOp
from ollie_rl.db import TunerModel, ChatCompletionModel, RewardModel
from ollie_rl.db.connection import init_db, shutdown_db
from ollie_rl.service.tuner_service import TunerService


class TestTunerService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reset DB engine and sessionmaker singletons
        await shutdown_db()

        # Use shared cache in-memory SQLite database for testing
        self.test_db_url = "sqlite+aiosqlite:///file:test_tuner_service?mode=memory&cache=shared&uri=true"

        # Create tables
        await init_db(self.test_db_url)

        # Instantiate TunerService
        self.service = TunerService()

    async def asyncTearDown(self):
        # Shutdown and clean up singletons
        await shutdown_db()

    @patch("ollie_rl.cookbook.Cookbook.create")
    async def test_create_tuner(self, mock_cookbook_create):
        # Mock the tuner instance returned by Cookbook
        mock_tuner = MagicMock(spec=Tuner)
        mock_tuner.kind = "gemini_msrl"
        mock_tuner.save_state = AsyncMock(
            return_value='{"tuning_job_name": "test-job"}'
        )
        mock_cookbook_create.return_value = mock_tuner

        tuner_id = await self.service.create_tuner("gemini_msrl", "my-tuner")

        # Verify returned ID format
        self.assertTrue(tuner_id.startswith("tuner_"))

        # Verify active_tuners cache
        self.assertIn(tuner_id, self.service.active_tuners)
        self.assertEqual(self.service.active_tuners[tuner_id], mock_tuner)

        # Verify database record
        async with self.service.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.name, "my-tuner")
            self.assertEqual(record.kind, "gemini_msrl")
            self.assertEqual(record.state, '{"tuning_job_name": "test-job"}')

    @patch("ollie_rl.cookbook.Cookbook.restore")
    async def test_get_active_and_lazy_restore(self, mock_cookbook_restore):
        tuner_id = "tuner_12345"
        mock_tuner = MagicMock(spec=Tuner)
        mock_tuner.kind = "gemini_msrl"

        # 1. Test retrieving from active memory first (cache hit)
        self.service.active_tuners[tuner_id] = mock_tuner
        retrieved = await self.service.get(tuner_id)
        self.assertEqual(retrieved, mock_tuner)
        mock_cookbook_restore.assert_not_called()

        # 2. Test lazy restore from database (cache miss)
        del self.service.active_tuners[tuner_id]

        # Insert record directly to DB
        async with self.service.async_session() as session:
            async with session.begin():
                record = TunerModel(
                    id=tuner_id,
                    name="lazy-tuner",
                    kind="gemini_msrl",
                    state='{"tuning_job_name": "lazy-job"}',
                )
                session.add(record)

        mock_cookbook_restore.return_value = mock_tuner

        # Retrieve again (should hit DB and restore)
        retrieved = await self.service.get(tuner_id)
        self.assertEqual(retrieved, mock_tuner)
        mock_cookbook_restore.assert_called_once_with(
            "gemini_msrl", '{"tuning_job_name": "lazy-job"}'
        )
        self.assertEqual(self.service.active_tuners[tuner_id], mock_tuner)

        # 3. Test retrieving non-existent tuner
        none_tuner = await self.service.get("non-existent-id")
        self.assertIsNone(none_tuner)

    async def test_record_chat_completion(self):
        await self.service.record_chat_completion(
            completion_id="comp_abc",
            tuner_id="tuner_123",
            run_id="run_456",
            datum_id="datum_789",
        )

        async with self.service.async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(ChatCompletionModel.id == "comp_abc")
            )
            record = result.scalar_one_or_none()
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.tuner_id, "tuner_123")
            self.assertEqual(record.run_id, "run_456")
            self.assertEqual(record.datum_id, "datum_789")

    async def test_create_reward_workflow(self):
        tuner_id = "tuner_1"
        run_id = "run_1"
        datum_id = "datum_1"

        # 1. Create a brand new reward
        await self.service.create_reward(tuner_id, datum_id, run_id, 10.5)

        async with self.service.async_session() as session:
            result = await session.execute(
                select(RewardModel).where(
                    RewardModel.tuner_id == tuner_id,
                    RewardModel.run_id == run_id,
                )
            )
            record = result.scalar_one_or_none()
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.datum_id, datum_id)
            self.assertEqual(record.reward, 10.5)
            self.assertEqual(record.train_count, 0)

        # 2. Update existing reward
        await self.service.create_reward(tuner_id, datum_id, run_id, 15.0)

        async with self.service.async_session() as session:
            result = await session.execute(
                select(RewardModel).where(
                    RewardModel.tuner_id == tuner_id,
                    RewardModel.run_id == run_id,
                )
            )
            record = result.scalar_one_or_none()
            assert record is not None
            self.assertEqual(record.reward, 15.0)

        # 3. Mismatching datum_id raises ValueError
        with self.assertRaises(ValueError) as context:
            await self.service.create_reward(tuner_id, "mismatch_datum", run_id, 20.0)
        self.assertIn("datum_id mismatch", str(context.exception))

    async def test_collect_rollout_ready_for_training(self):
        tuner_id = "tuner_collect"
        datum_id = "datum_collect"

        # 1. Insert 15 rewards (less than GROUP_SIZE = 16)
        async with self.service.async_session() as session:
            async with session.begin():
                for i in range(15):
                    session.add(
                        RewardModel(
                            tuner_id=tuner_id,
                            run_id=f"run_{i}",
                            datum_id=datum_id,
                            reward=float(i),
                            train_count=0,
                        )
                    )

        rollouts = await self.service.collect_rollout_ready_for_training(tuner_id)
        self.assertEqual(len(rollouts), 0)  # Not enough items in the group

        # 2. Insert the 16th reward to complete the group
        async with self.service.async_session() as session:
            async with session.begin():
                session.add(
                    RewardModel(
                        tuner_id=tuner_id,
                        run_id="run_15",
                        datum_id=datum_id,
                        reward=15.0,
                        train_count=0,
                    )
                )

        rollouts = await self.service.collect_rollout_ready_for_training(tuner_id)
        self.assertEqual(len(rollouts), 1)
        rollout = rollouts[0]
        self.assertEqual(rollout.datum_id, datum_id)
        self.assertEqual(len(rollout.runs), 16)

        # Check advantage calculation
        # Rewards are [0.0, 1.0, ..., 15.0]
        rewards = list(range(16))
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance)

        for run in rollout.runs:
            expected_adv = (run.reward - mean) / (std + 1e-8)
            self.assertAlmostEqual(run.advantage, expected_adv, places=5)

        # 3. Add rewards with train_count > 0, verify they are not collected
        async with self.service.async_session() as session:
            async with session.begin():
                for i in range(16):
                    session.add(
                        RewardModel(
                            tuner_id=tuner_id,
                            run_id=f"run_trained_{i}",
                            datum_id="datum_trained",
                            reward=1.0,
                            train_count=1,
                        )
                    )

        rollouts = await self.service.collect_rollout_ready_for_training(tuner_id)
        # Should still be 1 (only the first group is ready and has train_count <= 0)
        self.assertEqual(len(rollouts), 1)

    async def test_train_not_enough_groups(self):
        tuner_id = "tuner_train_early"

        # Mock get to return a mock tuner
        mock_tuner = MagicMock(spec=Tuner)
        self.service.active_tuners[tuner_id] = mock_tuner

        # Insert only 1 group of 16 ready rewards (need 32 groups)
        async with self.service.async_session() as session:
            async with session.begin():
                for i in range(16):
                    session.add(
                        RewardModel(
                            tuner_id=tuner_id,
                            run_id=f"run_{i}",
                            datum_id="datum_1",
                            reward=1.0,
                            train_count=0,
                        )
                    )

        await self.service.train(tuner_id)
        mock_tuner.train_step.assert_not_called()

    async def test_train_successful_workflow(self):
        tuner_id = "tuner_train_success"

        # 1. Mock the active tuner and its train_step
        mock_tuner = MagicMock(spec=Tuner)
        mock_train_op = MagicMock(spec=TrainOp)
        mock_train_op.wait = AsyncMock()
        mock_tuner.train_step = AsyncMock(return_value=mock_train_op)
        self.service.active_tuners[tuner_id] = mock_tuner

        # Insert a TunerModel so database checks pass if any
        async with self.service.async_session() as session:
            async with session.begin():
                session.add(
                    TunerModel(
                        id=tuner_id,
                        name="success-tuner",
                        kind="gemini_msrl",
                        state="{}",
                    )
                )

        # 2. Insert 32 groups of 16 ready rewards (512 total)
        # Also insert corresponding ChatCompletionModel records
        async with self.service.async_session() as session:
            async with session.begin():
                for g in range(32):
                    datum_id = f"datum_{g}"
                    for r in range(16):
                        run_id = f"run_g{g}_r{r}"
                        completion_id = f"comp_g{g}_r{r}"

                        session.add(
                            RewardModel(
                                tuner_id=tuner_id,
                                run_id=run_id,
                                datum_id=datum_id,
                                reward=1.0,
                                train_count=0,
                            )
                        )
                        session.add(
                            ChatCompletionModel(
                                id=completion_id,
                                tuner_id=tuner_id,
                                run_id=run_id,
                                datum_id=datum_id,
                            )
                        )

        # 3. Call train
        await self.service.train(tuner_id)

        # 4. Verify train_step was called with 512 examples
        mock_tuner.train_step.assert_called_once()
        examples_arg = mock_tuner.train_step.call_args[0][0]
        self.assertEqual(len(examples_arg), 512)
        self.assertTrue(all(isinstance(ex, Example) for ex in examples_arg))

        # Verify wait was called on training operation
        mock_train_op.wait.assert_called_once()

        # 5. Verify train_count has been incremented to 1 for all the trained rewards
        async with self.service.async_session() as session:
            result = await session.execute(
                select(RewardModel).where(RewardModel.tuner_id == tuner_id)
            )
            rewards = result.scalars().all()
            self.assertEqual(len(rewards), 512)
            self.assertTrue(all(r.train_count == 1 for r in rewards))
