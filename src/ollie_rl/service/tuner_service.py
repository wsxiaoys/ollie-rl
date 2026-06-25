import logging
import math
import uuid
from typing import Dict, List, Optional

from sqlalchemy import select, update
from ollie_rl.cookbook import Tuner, Cookbook
from ollie_rl.cookbook.types import Example
from ollie_rl.db import TunerModel, ChatCompletionModel, RunModel
from ollie_rl.db.connection import get_sessionmaker
from ollie_rl.types import Rollout, RolloutRun

logger = logging.getLogger(__name__)


class TunerService:
    """
    Handles both active in-memory tuners and their persistence to a database.
    Uses SQLAlchemy async engine and sessionmaker from the ollie_rl.db subpackage.
    """

    def __init__(self):
        self.active_tuners: Dict[str, Tuner] = {}
        self.async_session = get_sessionmaker()

    async def get(self, tuner_id: str) -> Optional[Tuner]:
        """
        Retrieve an active tuner instance by tuner_id (UUID).
        If the tuner is not in memory but exists in the database, restore it lazily.
        """
        if tuner_id in self.active_tuners:
            return self.active_tuners[tuner_id]

        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()
            if record and record.state is not None:
                logger.info(
                    f"Lazily restoring tuner for model: {tuner_id} (kind: {record.kind})"
                )
                tuner = await Cookbook.restore(record.kind, record.state)
                self.active_tuners[tuner_id] = tuner
                return tuner

        return None

    async def create_tuner(self, recipe: str, name: str) -> str:
        """
        Create and initialize a tuner using the Cookbook and register it.
        """
        tuner = await Cookbook.create(recipe, name)

        tuner_id = f"tuner_{uuid.uuid4()}"
        self.active_tuners[tuner_id] = tuner
        state_str = await tuner.save_state()
        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(TunerModel).where(TunerModel.id == tuner_id)
                )
                record = result.scalar_one_or_none()
                if record:
                    record.state = state_str
                else:
                    record = TunerModel(
                        id=tuner_id,
                        name=name,
                        kind=tuner.kind,
                        state=state_str,
                        train_step_id=0,
                    )
                    session.add(record)
        logger.info(f"Successfully persisted tuner {tuner_id} to database")
        return tuner_id

    async def record_chat_completion(
        self, completion_id: str, tuner_id: str, run_id: str, datum_id: str
    ) -> None:
        """
        Record a chat completion event in the database.
        """
        async with self.async_session() as session:
            async with session.begin():
                db_completion = ChatCompletionModel(
                    id=completion_id,
                    tuner_id=tuner_id,
                    run_id=run_id,
                    datum_id=datum_id,
                )
                session.add(db_completion)
        logger.info(f"Recorded chat completion {completion_id} in database")

    async def set_run_reward(
        self, tuner_id: str, datum_id: str, run_id: str, reward: float
    ) -> None:
        """
        Set or update the reward for a specific run.
        """
        async with self.async_session() as session:
            async with session.begin():
                run = RunModel(
                    id=run_id,
                    tuner_id=tuner_id,
                    datum_id=datum_id,
                    reward=reward,
                )
                await session.merge(run)
        logger.info(f"Successfully set reward for run {run_id} to {reward}")

    async def collect_rollout_ready_for_training(self, tuner_id: str) -> List[Rollout]:
        """
        Collect all rollouts ready for training under a specific tuner_id.
        1. Collect all runs that have train_count equal to 0.
        2. Group runs by datum_id, and whenever a group size is >= 8, then this group is considered ready.
        """
        TARGET_MAX_TRAIN_COUNT = 0
        GROUP_SIZE = 16

        async with self.async_session() as session:
            result = await session.execute(
                select(RunModel).where(
                    RunModel.tuner_id == tuner_id,
                    RunModel.train_count <= TARGET_MAX_TRAIN_COUNT,
                )
            )
            runs = result.scalars().all()

            # Group runs by datum_id
            grouped_runs: Dict[str, List[RunModel]] = {}
            for run_model in runs:
                if run_model.datum_id not in grouped_runs:
                    grouped_runs[run_model.datum_id] = []
                if len(grouped_runs[run_model.datum_id]) < GROUP_SIZE:
                    grouped_runs[run_model.datum_id].append(run_model)

            # Process only completed groups (size == GROUP_SIZE)
            rollouts: List[Rollout] = []
            for datum_id, group in grouped_runs.items():
                if len(group) != GROUP_SIZE:
                    continue

                # Calculate mean and std of rewards for this group
                rewards = [
                    run.reward if run.reward is not None else 0.0 for run in group
                ]
                mean = sum(rewards) / len(rewards)
                variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
                std = math.sqrt(variance)

                rollout_runs = []
                for run_model, reward in zip(group, rewards):
                    advantage = (reward - mean) / (std + 1e-8) if std > 1e-8 else 0.0
                    rollout_runs.append(
                        RolloutRun(
                            id=run_model.id,
                            datum_id=run_model.datum_id,
                            reward=reward,
                            advantage=advantage,
                        )
                    )
                rollouts.append(Rollout(datum_id=datum_id, runs=rollout_runs))

            return rollouts

    async def train(self, tuner_id: str) -> None:
        """
        Run a single RL training step (e.g., PPO/GRPO) for a tuner.
        1. Retrieve the active tuner instance.
        2. Collect rollouts ready for training. Only train if we have at least 32 groups (rollouts).
           If there are more than 32, only pick the first 32.
        3. Convert rollouts into Examples (by mapping RunModel IDs to ChatCompletionModel IDs).
        4. Update the train_count of the trained runs in the database using an update query.
        5. Call tuner.train_step(examples).
        """
        TARGET_GROUP_COUNT = 32

        tuner = await self.get(tuner_id)
        if not tuner:
            logger.error(f"Tuner {tuner_id} not found.")
            return

        # 1. Collect rollouts ready for training
        rollouts = await self.collect_rollout_ready_for_training(tuner_id)
        if len(rollouts) < TARGET_GROUP_COUNT:
            logger.info(
                f"Not enough groups ready for training under tuner {tuner_id} "
                f"(got {len(rollouts)}, need at least {TARGET_GROUP_COUNT})"
            )
            return

        # If there are more than 32 groups, only pick the first 32
        rollouts = rollouts[:TARGET_GROUP_COUNT]

        # 2. Map run advantages
        run_advantages: Dict[str, float] = {}
        for rollout in rollouts:
            for run in rollout.runs:
                run_advantages[run.id] = run.advantage

        run_ids = list(run_advantages.keys())

        # 3. Retrieve ChatCompletionModel records for these run_ids
        async with self.async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(
                    ChatCompletionModel.tuner_id == tuner_id,
                    ChatCompletionModel.run_id.in_(run_ids),
                )
            )
            completions = result.scalars().all()

        if not completions:
            logger.warning(
                f"No chat completions found for the ready runs under tuner {tuner_id}"
            )
            return

        # 4. Create Examples for Tuner.train_step
        examples = [
            Example(chat_completion_id=c.id, advantage=run_advantages[c.run_id])
            for c in completions
            if c.run_id in run_advantages
        ]

        # 5. Call tuner.train_step(examples) to trigger the training step operation
        train_op = await tuner.train_step(examples)

        # 6. Update train_count for the runs using an update query (safe now that backend accepted the request)
        async with self.async_session() as session:
            async with session.begin():
                await session.execute(
                    update(RunModel)
                    .where(RunModel.id.in_(run_ids))
                    .values(train_count=RunModel.train_count + 1)
                )

        # 7. Wait for the training step operation to complete
        await train_op.wait()

        logger.info(
            f"Successfully completed train step for tuner {tuner_id} using {len(examples)} examples"
        )
