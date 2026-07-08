"""Background train loop and batch collection for the tuner service."""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, update

from ollie_rl.db import ChatCompletionModel, TunerModel
from ollie_rl.db.models import CheckpointModel, DatumRowModel, RunModel
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.trainer import Checkpoint, Example, Trainer
from ollie_rl.types import Rollout, RolloutRun

logger = logging.getLogger(__name__)


class TrainingMixin(TunerServiceBase):
    """Train-loop lifecycle plus the consumable-batch collection logic."""

    def start_train_loop(self, interval: float = 10.0) -> None:
        """Start the background train loop (idempotent).

        The loop periodically attempts a train step for every tuner, skipping
        any tuner whose train lock is currently held (i.e. already training).
        """
        if self._train_loop_task is not None and not self._train_loop_task.done():
            return
        self._train_loop_task = asyncio.create_task(self._train_loop(interval))

    async def stop_train_loop(self) -> None:
        """Stop the background train loop and wait for it to unwind."""
        task = self._train_loop_task
        if task is None:
            return
        self._train_loop_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _train_loop(self, interval: float) -> None:
        """Periodically trigger `maybe_train` for every tuner.

        Runs forever until cancelled. Each iteration sleeps for `interval`
        seconds, then attempts a train step for each tuner that is not already
        training. Failures for a single tuner never abort the loop.
        """
        logger.info(f"Starting train loop (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)
                await self._train_all_pending()
            except asyncio.CancelledError:
                logger.info("Train loop cancelled")
                raise
            except Exception:
                logger.exception("Unexpected error in train loop")

    async def _train_all_pending(self) -> None:
        """Trigger `maybe_train` for every tuner not currently training.

        Tuners whose train lock is already held are skipped (don't block on a
        long in-progress train step); the rest are fired off as independent
        background tasks so a slow train step never holds up the loop.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel.id).where(TunerModel.trainer_state.is_not(None))
            )
            tuner_ids = [row[0] for row in result.all()]

        async def _train_one(tuner_id: str) -> None:
            try:
                await self._maybe_train(tuner_id)
            except Exception:
                logger.exception(f"Scheduled train step failed for tuner {tuner_id}")

        for tuner_id in tuner_ids:
            lock = self._train_locks.get(tuner_id)
            if lock is not None and lock.locked():
                # Already training; don't queue behind the in-progress step.
                continue
            # Fire-and-forget: let the train step run independently of the loop.
            self._background_jobs.spawn(_train_one(tuner_id))

    async def _maybe_train(self, tuner_id: str) -> None:
        """
        Attempt to start (and wait for) a train step for `tuner_id`.

        Serialized per-tuner via `self._train_lock_for` so only one train step
        runs at a time for a given tuner, while distinct tuners train
        concurrently.
        """
        async with self._train_lock_for(tuner_id):
            trainer = await self._get_trainer(tuner_id)

            train_op = await trainer.pending_train_op()
            if train_op is not None:
                # A train op is in flight. In steady state the coroutine that
                # submitted it still holds this lock and we never reach here. If
                # we DID acquire the lock, no one is awaiting it (e.g. it was
                # submitted before a restart): fall through to the shared
                # wait/persist below to drive it to completion so trainer state
                # advances and `pending_train_op` clears.
                logger.info(
                    f"Reconciling in-flight train op for tuner {tuner_id}"
                )
            else:
                async with self.async_session() as session:
                    async with session.begin():
                        batch, run_ids = await self._collect_consumable_batch(
                            tuner_id, session, trainer
                        )
                        if not batch:
                            return

                        # Accepted limitation (dual-write, not fully atomic):
                        # `train_step` submits the backend LRO and persists
                        # `pending_train_op` in its *own* transaction, before
                        # the `trained_count` bump below commits. If the process
                        # crashes in between, on restart the LRO still completes
                        # (advancing policy_generation) but these runs keep
                        # `trained_count = 0` and get collected into a later
                        # batch -- i.e. the batch may be trained twice. Bounded
                        # (and dampened by the off-policy staleness filter);
                        # tolerated for now rather than adding a cross-backend
                        # 2-phase commit.
                        train_op = await trainer.train_step(
                            batch,
                        )  # submits LRO + state_store.save
                        await session.execute(  # bump trained_count
                            update(RunModel)
                            .where(RunModel.tuner_id == tuner_id)
                            .where(RunModel.id.in_(run_ids))
                            .values(trained_count=RunModel.trained_count + 1)
                        )

            # Single completion barrier for both paths (fresh submit + restart
            # reconcile): await the op once and persist the checkpoint it
            # yielded. A checkpoint can complete across a restart, so the insert
            # must run on the reconcile path too.
            if train_op is not None:
                checkpoint = await train_op.wait()
                await self._persist_checkpoint(tuner_id, checkpoint)
                logger.info(f"Successfully completed train step for tuner {tuner_id}")

    async def _persist_checkpoint(
        self, tuner_id: str, checkpoint: Optional[Checkpoint]
    ) -> None:
        """Persist the checkpoint a completed train step yielded.

        Shared by both ``_maybe_train`` ``wait()`` sites (fresh submit +
        post-restart reconcile) since a checkpoint can complete across a
        restart. A ``None`` checkpoint (the backend emitted no generation for
        the step) is a no-op. The service, not the trainer, owns this DB write.
        This ``checkpoints`` table is the eval scheduler's source of truth for
        "what checkpoints exist to be evaluated".
        """
        if checkpoint is None:
            return
        async with self.async_session() as session:
            async with session.begin():
                session.add(
                    CheckpointModel(
                        tuner_id=tuner_id,
                        ref=checkpoint.ref,
                        policy_generation=checkpoint.policy_generation,
                    )
                )

    async def _collect_consumable_batch(
        self, tuner_id: str, session, trainer: Trainer
    ) -> Tuple[List[Example], List[str]]:
        recipe = await self._recipe_for(tuner_id)

        # Exclude eval runs: eval-ness is derived from the datum's kind, so
        # anti-join against the tuner's eval datum set. A rewarded eval run
        # must never enter a GRPO group (it scores a held-out datum only).
        eval_datums = select(DatumRowModel.datum_id).where(
            DatumRowModel.tuner_id == tuner_id,
            DatumRowModel.kind == "eval",
        )
        result = await session.execute(
            select(RunModel).where(
                RunModel.tuner_id == tuner_id,
                RunModel.trained_count <= 0,
                RunModel.rejected_count <= 0,
                RunModel.reward != None,  # noqa: E711
                RunModel.datum_id.not_in(eval_datums),
            )
        )
        run_records = list(result.scalars().all())

        if not run_records:
            return [], []

        # 1. Retrieve ChatCompletions for all candidate runs to check for staleness.
        # Only project the columns actually consumed below; skip the large,
        # blob-heavy `request`/`response` payloads and instead extract the
        # single needed field (`response['id']`, the backend candidate id) via
        # a JSON query.
        candidate_run_ids = [r.id for r in run_records]
        result = await session.execute(
            select(
                ChatCompletionModel.id,
                ChatCompletionModel.run_id,
                ChatCompletionModel.policy_generation,
                ChatCompletionModel.tokens,
                ChatCompletionModel.logprobs,
                ChatCompletionModel.response["id"].as_string().label("candidate_id"),
            ).where(
                ChatCompletionModel.tuner_id == tuner_id,
                ChatCompletionModel.run_id.in_(candidate_run_ids),
            )
        )
        completions = result.all()
        completion_by_run_id = {c.run_id: c for c in completions if c.run_id}

        # 2. Filter out stale runs and requeue them (mark them as rejected)
        trainer_generation = trainer.policy_generation
        max_off_policy_generation = recipe.max_off_policy_generation

        stale_run_ids = []
        fresh_run_records = []
        for run in run_records:
            completion = completion_by_run_id.get(run.id)
            if completion is not None:
                if (
                    trainer_generation - completion.policy_generation
                    > max_off_policy_generation
                ):
                    stale_run_ids.append(run.id)
                    continue
            fresh_run_records.append(run)

        if stale_run_ids:
            logger.info(
                f"Requeuing {len(stale_run_ids)} stale runs for tuner {tuner_id} "
                f"(trainer_generation={trainer_generation}, max_off_policy_generation={max_off_policy_generation})"
            )
            await session.execute(
                update(RunModel)
                .where(RunModel.tuner_id == tuner_id)
                .where(RunModel.id.in_(stale_run_ids))
                .values(rejected_count=RunModel.rejected_count + 1)
            )
            run_records = fresh_run_records

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
            logger.debug(
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

        # Filter completions to only include those in run_ids
        completions = [c for c in completions if c.run_id in run_advantages]

        if not completions:
            logger.warning(
                f"No chat completions found for the ready runs under tuner {tuner_id} "
                f"(run_ids={run_ids})"
            )
            return [], []

        # Create Examples for Trainer.train_step. `tokens` / `logprobs`
        # are decoded transparently by the model-layer TypeDecorators.
        #
        # `chat_completion_id` must be the *backend-issued* candidate id (what
        # gemini_msrl replays via `candidate_id`), which is the completion's
        # own id captured at sample time and persisted in the `response`
        # payload -- extracted above via the `response['id']` JSON query as
        # `candidate_id`. The row primary key (`c.id`) is a synthetic internal
        # id and must NOT leak to a training backend; fall back to it only if
        # the response somehow lacks an id.
        examples = []
        for c in completions:
            if c.run_id not in run_advantages:
                continue
            examples.append(
                Example(
                    chat_completion_id=c.candidate_id or c.id,
                    advantage=run_advantages[c.run_id],
                    policy_generation=c.policy_generation,
                    tokens=c.tokens,
                    logprobs=c.logprobs,
                )
            )

        return examples, run_ids
