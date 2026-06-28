import asyncio
import logging
import math
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, update

from ollie_rl.cookbook import Cookbook, Recipe
from ollie_rl.trainer import Trainer, StateStore, Example
from ollie_rl.trainer import factory as trainer_factory
from ollie_rl.db import TunerModel, ChatCompletionModel, DatumRowModel
from ollie_rl.db.connection import get_sessionmaker
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from openai.types.chat import ChatCompletion
from ollie_rl.types import Rollout, RolloutRun, DispenseRun, ChatCompletionRequest

logger = logging.getLogger(__name__)


class TunerNotFoundError(Exception):
    pass


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
    Handles both active in-memory trainers and their persistence to a database.
    Uses SQLAlchemy async engine and sessionmaker from the ollie_rl.db subpackage.
    """

    def __init__(self):
        self.active_trainers: Dict[str, Trainer] = {}
        self.async_session = get_sessionmaker()
        # Global lock ensuring `maybe_train` runs serially across all tuners in this process.
        self._train_lock = asyncio.Lock()

    async def get_trainer(self, tuner_id: str) -> Optional[Trainer]:
        """
        Retrieve an active trainer instance by tuner_id.
        If the trainer is not in memory but exists in the database, restore it lazily
        by opening it against its DB-backed StateStore.
        """
        if tuner_id in self.active_trainers:
            return self.active_trainers[tuner_id]

        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()

        if record is None or record.state is None:
            return None

        return await self._materialize(tuner_id, record)

    async def _materialize(self, tuner_id: str, record: TunerModel) -> Trainer:
        if tuner_id in self.active_trainers:
            return self.active_trainers[tuner_id]

        trainer = record.trainer

        logger.info(f"Lazily restoring trainer for tuner: {tuner_id} (kind: {trainer})")
        state_store = _DbStateStore(tuner_id)
        factory = trainer_factory.get(trainer)
        trainer_instance = await factory.open(record.name, state_store)
        self.active_trainers[tuner_id] = trainer_instance
        return trainer_instance

    async def create_tuner(
        self,
        recipe: str,
        name: str,
        datum_ids: List[str],
        trainer: str,
    ) -> str:
        """
        Create and initialize a tuner using the Cookbook and register it.
        """
        assert Cookbook.has(recipe)
        factory = trainer_factory.get(trainer)  # validate now, fail fast

        async with self.async_session() as session:
            async with session.begin():
                tuner_record = TunerModel(
                    name=name,
                    recipe=recipe,
                    trainer=trainer,
                    state=None,
                )
                session.add(tuner_record)
                await session.flush()
                for datum_id in datum_ids:
                    session.add(
                        DatumRowModel(
                            tuner_id=tuner_record.id,
                            datum_id=datum_id,
                        )
                    )

        tuner_id = tuner_record.id
        state_store = _DbStateStore(tuner_id)
        trainer_instance = await factory.open(name, state_store)
        self.active_trainers[tuner_id] = trainer_instance

        logger.info(f"Successfully created tuner {tuner_id}")
        return tuner_id

    async def sample(
        self,
        tuner_id: str,
        request: ChatCompletionRequest,
        run_id: Optional[str] = None,
    ) -> ChatCompletion:
        """
        Generate a chat completion from the active policy of the requested model,
        and optionally record metadata if run_id is provided.
        """
        trainer = await self.get_trainer(tuner_id)
        if not trainer:
            raise TunerNotFoundError(
                f"Tuner '{tuner_id}' not found or not initialized."
            )

        datum_id = None
        if run_id is not None:
            async with self.async_session() as session:
                result = await session.execute(
                    select(RunModel).where(
                        RunModel.tuner_id == tuner_id,
                        RunModel.id == run_id,
                    )
                )
                run_record = result.scalar_one_or_none()
                if not run_record:
                    raise RunNotFoundError(f"Unknown run_id {run_id}")

                if run_record.reward is not None:
                    raise RewardAlreadySetError(
                        f"Reward already set for run '{run_id}'"
                    )

                now = utcnow()
                if run_record.expires_at <= now:
                    raise RunExpiredError(f"Run '{run_id}' has expired")

                # Override datum_id from database record to prevent client lying
                datum_id = run_record.datum_id

        # Generate completion
        sample_op = await trainer.sample(request)
        sample = await sample_op.wait()
        policy_generation = sample.policy_generation

        # Record completion metadata
        if run_id is not None:
            assert datum_id is not None
            await self.record_chat_completion(
                completion_id=sample.completion.id,
                tuner_id=tuner_id,
                run_id=run_id,
                datum_id=datum_id,
                policy_generation=policy_generation,
            )

        return sample.completion

    async def record_chat_completion(
        self,
        completion_id: str,
        tuner_id: str,
        run_id: str,
        datum_id: str,
        policy_generation: str,
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

    async def update_reward(self, tuner_id: str, run_id: str, reward: float) -> None:
        """
        Record or update the reward for a specific run.
        """
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
                    raise RunNotFoundError(
                        f"Run '{run_id}' not found under tuner '{tuner_id}'"
                    )

                if record.reward is not None:
                    raise RewardAlreadySetError(
                        f"Reward already set for run '{run_id}'"
                    )

                now = utcnow()
                if record.expires_at <= now:
                    raise RunExpiredError(f"Run '{run_id}' has expired")

                record.reward = reward
                record.updated_at = now
        logger.info(f"Successfully recorded reward {reward} for run {run_id}")

    async def maybe_train(self, tuner_id: str) -> None:
        """
        Attempt to start (and wait for) a train step for `tuner_id`.

        Serialized globally via `self._train_lock` to ensure only one train step
        runs at a time in this process.
        """
        async with self._train_lock:
            trainer = await self.get_trainer(tuner_id)
            if trainer is None:
                return

            if await trainer.is_training():
                return

            train_op = None
            async with self.async_session() as session:
                async with session.begin():
                    batch, run_ids = await self._collect_consumable_batch(
                        tuner_id, session
                    )
                    if not batch:
                        return

                    train_op = await trainer.train_step(
                        batch,
                    )  # submits LRO + state_store.save
                    await session.execute(  # bump trained_count
                        update(RunModel)
                        .where(RunModel.tuner_id == tuner_id)
                        .where(RunModel.id.in_(run_ids))
                        .values(trained_count=RunModel.trained_count + 1)
                    )

            if train_op is not None:
                await train_op.wait()
                logger.info(f"Successfully completed train step for tuner {tuner_id}")

    async def _collect_consumable_batch(
        self, tuner_id: str, session
    ) -> Tuple[List[Example], List[str]]:
        recipe = await self._recipe_for(tuner_id)
        if recipe is None:
            return [], []

        result = await session.execute(
            select(RunModel).where(
                RunModel.tuner_id == tuner_id,
                RunModel.trained_count <= 0,
                RunModel.reward != None,  # noqa: E711
            )
        )
        run_records = result.scalars().all()

        # Group rewards by datum_id
        grouped_runs: Dict[str, List[RunModel]] = {}
        for reward in run_records:
            if reward.datum_id not in grouped_runs:
                grouped_runs[reward.datum_id] = []
            if len(grouped_runs[reward.datum_id]) < recipe.group_size:
                grouped_runs[reward.datum_id].append(reward)

        # Process only completed groups (size == group_size)
        rollouts: List[Rollout] = []
        for group in grouped_runs.values():
            if len(group) != recipe.group_size:
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
            for record_item, reward in zip(group, rewards):
                advantage = (reward - mean) / (std + 1e-8) if std > 1e-8 else 0.0
                rollout_runs.append(
                    RolloutRun(
                        id=record_item.id,
                        reward=reward,
                        advantage=advantage,
                    )
                )
            rollouts.append(Rollout(runs=rollout_runs))

        if len(rollouts) < recipe.num_groups_per_batch:
            logger.info(
                f"Not enough groups ready for training under tuner {tuner_id} "
                f"(got {len(rollouts)}, need at least {recipe.num_groups_per_batch})"
            )
            return [], []

        # If there are more than num_groups_per_batch groups, only pick the first num_groups_per_batch
        rollouts = rollouts[: recipe.num_groups_per_batch]

        # Map run advantages
        run_advantages: Dict[str, float] = {}
        for rollout in rollouts:
            for run in rollout.runs:
                run_advantages[run.id] = run.advantage

        run_ids = list(run_advantages.keys())

        # Retrieve ChatCompletionModel records for these run_ids
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
            return [], []

        # Create Examples for Trainer.train_step
        examples = [
            Example(
                chat_completion_id=c.id,
                advantage=run_advantages[c.run_id],
                policy_generation=c.policy_generation,
            )
            for c in completions
            if c.run_id in run_advantages
        ]

        return examples, run_ids

    async def dispense_run(self, tuner_id: str) -> Optional[DispenseRun]:
        """
        Dispense a run for a tuner.
        """
        trainer = await self.get_trainer(tuner_id)
        if not trainer:
            raise TunerNotFoundError(
                f"Tuner '{tuner_id}' not found or not initialized."
            )

        recipe = await self._recipe_for(tuner_id)
        if recipe and not recipe.allow_dispense_during_training:
            if await trainer.is_training():
                return None

        async with self.async_session() as session:
            datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)

        datum_id = self._pick_datum(datum_pool, runs)
        if datum_id is None:
            return None

        run_record = RunModel(
            tuner_id=tuner_id,
            datum_id=datum_id,
            reward=None,
            trained_count=0,
            expires_at=utcnow() + timedelta(seconds=7200),
        )
        async with self.async_session() as session:
            async with session.begin():
                session.add(run_record)

        return DispenseRun(
            run_id=run_record.id,
            datum_id=run_record.datum_id,
            expires_at=run_record.expires_at,
        )

    async def _recipe_for(self, tuner_id: str) -> Optional[Recipe]:
        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()
            if record:
                return Cookbook.get(record.recipe)

    async def _load_pool_and_runs(
        self, tuner_id: str, session
    ) -> Tuple[List[str], List[RunModel]]:
        result = await session.execute(
            select(DatumRowModel.datum_id).where(DatumRowModel.tuner_id == tuner_id)
        )
        datum_pool = list(result.scalars().all())

        runs_result = await session.execute(
            select(RunModel).where(RunModel.tuner_id == tuner_id)
        )
        runs = list(runs_result.scalars().all())
        return datum_pool, runs

    def _pick_datum(
        self,
        datum_pool: List[str],
        runs: List[RunModel],
    ) -> Optional[str]:
        now = utcnow()
        score = {d: 0 for d in datum_pool}
        for r in runs:
            if r.datum_id not in score:
                continue
            has_reward = r.reward is not None
            is_pending = r.reward is None and r.expires_at > now
            if has_reward or is_pending:
                score[r.datum_id] += 1
        return min(score, key=lambda d: score[d])
