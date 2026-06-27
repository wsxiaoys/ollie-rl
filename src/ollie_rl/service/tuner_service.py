import logging
import math
import uuid
from typing import Dict, List, Optional

from sqlalchemy import select, update
from ollie_rl.cookbook import Tuner, Cookbook
from ollie_rl.cookbook.types import Example, StateStore
from ollie_rl.db import TunerModel, ChatCompletionModel, DatumRowModel
from ollie_rl.db.connection import get_sessionmaker
from ollie_rl.db.models import RunModel
from ollie_rl.types import Rollout, RolloutRun

logger = logging.getLogger(__name__)


class RunNotFoundError(Exception):
    pass


class RunExpiredError(Exception):
    pass


class RewardAlreadySetError(Exception):
    pass


class _DbStateStore(StateStore):
    """
    StateStore implementation backed by the `tuners` table.

    Read-your-writes is provided by the underlying transactional UPDATE +
    SELECT against a single row keyed by `tuner_id`.
    """

    def __init__(self, tuner_id: str):
        self._tuner_id = tuner_id

    async def load(self) -> Optional[str]:
        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(TunerModel.state).where(TunerModel.id == self._tuner_id)
            )
            return result.scalar_one_or_none()

    async def save(self, state: str) -> None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            async with session.begin():
                await session.execute(
                    update(TunerModel)
                    .where(TunerModel.id == self._tuner_id)
                    .values(state=state)
                )
        logger.debug(f"Persisted state for tuner {self._tuner_id}")


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
        Retrieve an active tuner instance by tuner_id.
        If the tuner is not in memory but exists in the database, restore it lazily
        by opening it against its DB-backed StateStore (which will hand the persisted
        blob back to the recipe on load).
        """
        if tuner_id in self.active_tuners:
            return self.active_tuners[tuner_id]

        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()

        if record is None or record.state is None:
            # Either no such row, or row exists but the Tuner never persisted
            # initial state (e.g. crashed mid-bootstrap). Treat as not found.
            return None

        logger.info(
            f"Lazily restoring tuner for model: {tuner_id} (kind: {record.kind})"
        )
        state_store = _DbStateStore(tuner_id)
        tuner = await Cookbook.open(record.kind, record.name, state_store)
        self.active_tuners[tuner_id] = tuner
        return tuner

    async def create_tuner(self, recipe: str, name: str, datum_ids: List[str]) -> str:
        """
        Create and initialize a tuner using the Cookbook and register it.

        The service inserts a row with `state=NULL`; the Tuner is then
        responsible for filling it in via its StateStore.
        """
        tuner_id = f"tuner_{uuid.uuid4()}"

        # 1. Reserve the row so the StateStore has something to UPDATE against.
        async with self.async_session() as session:
            async with session.begin():
                session.add(
                    TunerModel(
                        id=tuner_id,
                        name=name,
                        kind=recipe,
                        state=None,
                    )
                )
                for datum_id in datum_ids:
                    session.add(
                        DatumRowModel(
                            tuner_id=tuner_id,
                            datum_id=datum_id,
                        )
                    )

        # 2. Open the tuner against its DB-backed store. The recipe will
        # call `state_store.save(...)` once it has a persistable snapshot.
        state_store = _DbStateStore(tuner_id)
        tuner = await Cookbook.open(recipe, name, state_store)
        self.active_tuners[tuner_id] = tuner

        logger.info(f"Successfully created tuner {tuner_id}")
        return tuner_id

    async def record_chat_completion(
        self, completion_id: str, tuner_id: str, run_id: str, datum_id: str, policy_generation: str,
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
                    policy_generation=policy_generation,
                )
                session.add(db_completion)
        logger.info(f"Recorded chat completion {completion_id} in database")

    async def update_reward(
        self, tuner_id: str, run_id: str, reward: float
    ) -> None:
        """
        Record or update the reward for a specific run.
        """
        from datetime import datetime
        async with self.async_session() as session:
            async with session.begin():
                result = await session.execute(
                    select(RunModel).where(
                        RunModel.id == run_id,
                        RunModel.tuner_id == tuner_id,
                    )
                )
                record = result.scalar_one_or_none()
                if not record:
                    raise RunNotFoundError(f"Run '{run_id}' not found under tuner '{tuner_id}'")

                if record.reward is not None:
                    raise RewardAlreadySetError(f"Reward already set for run '{run_id}'")

                now = datetime.now()
                if record.expires_at.tzinfo is not None:
                    from datetime import timezone
                    now = datetime.now(timezone.utc)

                if record.expires_at <= now:
                    raise RunExpiredError(f"Run '{run_id}' has expired")

                record.reward = reward
                record.updated_at = now
        logger.info(f"Successfully recorded reward {reward} for run {run_id}")

    async def collect_rollout_ready_for_training(self, tuner_id: str) -> List[Rollout]:
        """
        Collect all rollouts ready for training under a specific tuner_id.
        1. Collect all rewards that have train_count equal to 0.
        2. Group rewards by datum_id, and whenever a group size is >= 16, then this group is considered ready.
        """
        TARGET_MAX_TRAIN_COUNT = 0
        GROUP_SIZE = 16

        async with self.async_session() as session:
            result = await session.execute(
                select(RunModel).where(
                    RunModel.tuner_id == tuner_id,
                    RunModel.train_count <= TARGET_MAX_TRAIN_COUNT,
                    RunModel.reward != None,
                )
            )
            run_records = result.scalars().all()

            # Group rewards by datum_id
            grouped_runs: Dict[str, List[RunModel]] = {}
            for reward in run_records:
                if reward.datum_id not in grouped_runs:
                    grouped_runs[reward.datum_id] = []
                if len(grouped_runs[reward.datum_id]) < GROUP_SIZE:
                    grouped_runs[reward.datum_id].append(reward)

            # Process only completed groups (size == GROUP_SIZE)
            rollouts: List[Rollout] = []
            for group in grouped_runs.values():
                if len(group) != GROUP_SIZE:
                    continue

                # Calculate mean and std of rewards for this group
                rewards = [
                    reward_model.reward if reward_model.reward is not None else 0.0
                    for reward_model in group
                ]
                mean = sum(rewards) / len(rewards)
                variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
                std = math.sqrt(variance)

                rollout_runs = []
                for record, reward in zip(group, rewards):
                    advantage = (reward - mean) / (std + 1e-8) if std > 1e-8 else 0.0
                    rollout_runs.append(
                        RolloutRun(
                            id=record.id,
                            reward=reward,
                            advantage=advantage,
                        )
                    )
                rollouts.append(Rollout(runs=rollout_runs))

            return rollouts

    async def train(self, tuner_id: str) -> None:
        """
        Run a single RL training step (e.g., PPO/GRPO) for a tuner.
        1. Retrieve the active tuner instance.
        2. Collect rollouts ready for training. Only train if we have at least 32 groups (rollouts).
           If there are more than 32, only pick the first 32.
        3. Convert rollouts into Examples (by mapping RewardModel IDs to ChatCompletionModel IDs).
        4. Update the train_count of the trained rewards in the database using an update query.
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

        # 6. Update train_count for the rewards using an update query (safe now that backend accepted the request)
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
