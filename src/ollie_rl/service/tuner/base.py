"""Shared state and data-access core for the tuner service mixins.

:class:`TunerServiceBase` holds the in-memory trainer registry, the various
concurrency locks, and the DB-access helpers that more than one mixin depends
on (trainer materialization, recipe lookup, and the pool/runs/quarantine
queries). The concrete :class:`~ollie_rl.service.tuner.service.TunerService`
composes the feature mixins on top of this base.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openai.types.chat import ChatCompletion
from sqlalchemy import func, select

from ollie_rl.background import BackgroundJob
from ollie_rl.cookbook import Cookbook, Recipe
from ollie_rl.db import (
    ChatCompletionModel,
    DatumRowModel,
    InFlightChatCompletionModel,
    TunerModel,
)
from ollie_rl.db.connection import get_sessionmaker
from ollie_rl.db.models import RunModel
from ollie_rl.service.tuner.dispense import RewardedRun
from ollie_rl.service.tuner.completion_helpers import completion_has_length_finish
from ollie_rl.service.tuner.constants import RUN_EXPIRE_GENERATION_BUDGET_MS
from ollie_rl.service.tuner.errors import TunerNotFoundError
from ollie_rl.service.tuner.locks import KeyedLocks
from ollie_rl.service.tuner.state_store import DbStateStore
from ollie_rl.trainer import Trainer
from ollie_rl.trainer import factory as trainer_factory

logger = logging.getLogger(__name__)


class TunerServiceBase:
    """
    Handles both active in-memory trainers and their persistence to a database.
    Uses SQLAlchemy async engine and sessionmaker from the ollie_rl.db subpackage.
    """

    def __init__(self):
        self.active_trainers: Dict[str, Trainer] = {}
        # Per-tuner train locks: a tuner serializes its own train steps (the
        # `is_training()` check + batch collection + `trained_count` bump must
        # be atomic), but distinct tuners are independent trainers with no
        # shared state, so they train concurrently instead of being head-of-line
        # blocked behind one another's (potentially long) train step.
        self._train_locks: Dict[str, asyncio.Lock] = {}
        # Per-tuner materialization locks: serialize the lazy
        # restore-into-`active_trainers` of a single tuner (so concurrent first
        # requests don't each build a trainer), while distinct tuners
        # materialize concurrently.
        self._materialize_locks = KeyedLocks()
        # Lock serializing the read-pick-insert critical section of `dispense_run`
        # so concurrent dispenses observe each other's in-flight runs and don't
        # all pick the same datum (which would over-dispense past `group_size`).
        self._dispense_lock = asyncio.Lock()
        # Per-(tuner_id, run_id, request_hash) locks that make sampling
        # idempotent: a retry of a stalled request must replay the stored
        # completion instead of generating a duplicate sibling. Serializing on
        # this key closes the check-then-record race between concurrent
        # retries of the same turn.
        self._sample_locks = KeyedLocks()
        # Background task that periodically polls every tuner and triggers a
        # train step when a batch is ready, replacing the per-reward
        # fire-and-forget trigger.
        self._train_loop_task: Optional[asyncio.Task] = None
        # Holds strong references to fire-and-forget train-step tasks so they
        # aren't garbage-collected mid-flight (see BackgroundJob).
        self._background_jobs = BackgroundJob()

    def _train_lock_for(self, tuner_id: str) -> asyncio.Lock:
        """Return the per-tuner train lock, creating it on first use.

        Race-free under asyncio: there is no ``await`` between the ``get`` and
        the insert, so the event loop cannot interleave another coroutine and
        create a competing lock for the same ``tuner_id``.
        """
        lock = self._train_locks.get(tuner_id)
        if lock is None:
            lock = asyncio.Lock()
            self._train_locks[tuner_id] = lock
        return lock

    @property
    def async_session(self):
        return get_sessionmaker()

    async def _get_trainer(self, tuner_id: str) -> Trainer:
        """
        Retrieve an active trainer instance by tuner_id.
        If the trainer is not in memory but exists in the database, restore it lazily
        by opening it against its DB-backed StateStore.
        """
        # Fast path: Return immediately if the trainer is already loaded in memory,
        # avoiding any lock acquisition overhead for subsequent requests.
        if tuner_id in self.active_trainers:
            return self.active_trainers[tuner_id]

        async with self._materialize_locks.acquire(tuner_id):
            # Double-checked locking pattern: A concurrent request might have
            # finished materializing the trainer while we were waiting for the lock.
            if tuner_id in self.active_trainers:
                return self.active_trainers[tuner_id]

            return await self._materialize(tuner_id)

    async def _materialize(self, tuner_id: str) -> Trainer:
        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()

        if record is None or record.trainer_state is None:
            raise TunerNotFoundError(
                f"Tuner '{tuner_id}' not found or not initialized."
            )

        trainer = record.trainer

        logger.info(f"Lazily restoring trainer for tuner: {tuner_id} (kind: {trainer})")
        state_store = DbStateStore(tuner_id)
        factory = trainer_factory.get(trainer)

        trainer_instance = await factory.restore(record.name, state_store)
        self.active_trainers[tuner_id] = trainer_instance
        return trainer_instance

    async def _recipe_for(self, tuner_id: str) -> Recipe:
        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()
            if not record:
                raise TunerNotFoundError(
                    f"Tuner '{tuner_id}' not found or not initialized."
                )
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

    async def _rewarded_datums(self, tuner_id: str, session) -> Dict[str, RewardedRun]:
        """Per *rewarded* run, a :class:`RewardedRun` (datum id + reward).

        Returns a ``run_id -> RewardedRun`` map used by the dispenser's
        quarantine logic for the rewarded denominator. A rewarded run has no
        lingering in-flight row -- it is deleted on success -- so it is located
        via its completions. The join to ``RunModel`` keeps the map to exactly
        the rewarded runs the quarantine algorithm consults (and carries each
        run's ``datum_id`` + ``reward``, so ``terminal_stats`` can tally both the
        rewarded denominator and the ``reward == 1.0`` success numerator without
        the full run list or a second query). Every rewarded run for the tuner is
        counted (no recency window).
        """
        result = await session.execute(
            select(ChatCompletionModel.run_id, RunModel.datum_id, RunModel.reward)
            .join(RunModel, RunModel.id == ChatCompletionModel.run_id)
            .where(
                ChatCompletionModel.tuner_id == tuner_id,
                RunModel.reward.is_not(None),
            )
            .distinct()
        )
        return {
            run_id: RewardedRun(datum_id=datum_id, reward=reward)
            for run_id, datum_id, reward in result.all()
            if run_id is not None
        }

    async def _length_datums(
        self,
        tuner_id: str,
        session,
        run_ids: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Per *length-limited* run, its ``datum_id``.

        Returns a ``run_id -> datum_id`` map for runs with at least one recorded
        completion whose finish reason is ``"length"``. This backs the
        ``RunStatus == "length"`` split for ``list_runs``/``get_run`` and the
        dispenser's length-rate quarantine numerator. The source of truth is the
        persisted response: either the backend returned a length-limited sample,
        or the context-window guard rewrote an oversized completion to a cleared
        ``finish_reason == "length"`` response before recording it. Length runs
        remain ordinary rewarded runs for terminal-stat denominator accounting.
        """
        if run_ids is not None and not run_ids:
            return {}

        stmt = (
            select(
                ChatCompletionModel.run_id,
                RunModel.datum_id,
                ChatCompletionModel.response,
            )
            .join(RunModel, RunModel.id == ChatCompletionModel.run_id)
            .where(ChatCompletionModel.tuner_id == tuner_id)
        )
        if run_ids is not None:
            stmt = stmt.where(ChatCompletionModel.run_id.in_(run_ids))

        result = await session.execute(stmt)
        length_datum_by_run: Dict[str, str] = {}
        for run_id, datum_id, response in result.all():
            if run_id is None:
                continue
            completion = ChatCompletion.model_validate(response)
            if completion_has_length_finish(completion):
                length_datum_by_run[run_id] = datum_id
        return length_datum_by_run

    async def _expired_datums(
        self,
        tuner_id: str,
        now: datetime,
        session,
        run_ids: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Per *expired, unrewarded* run, its ``datum_id``.

        Returns a ``run_id -> datum_id`` map of `expired` runs (as opposed to
        `lost`). Among unrewarded, past-lease runs
        (``reward IS NULL AND expires_at <= now``, enforced here in SQL via the
        join to ``RunModel``), a run is `expired` when *either* of two
        compute-waste signals holds:

        1. **Lingering in-flight op.** The run still has an
           ``InFlightChatCompletionModel`` row (an op that timed out or was
           cancelled and is still progressing on the backend) -- the signal that
           the generation itself stalled past the lease.
        2. **Duration past the expiration threshold.** The summed
           ``duration_ms`` across the run's recorded completions is
           ``>= RUN_EXPIRE_GENERATION_BUDGET_MS`` -- the run burned real generation time
           yet never earned a reward.

        A run matching either signal is `expired`; the rest are `lost` (a
        crashed/abandoned worker, or ops that all finished but no reward was ever
        posted). This is the single source of truth for the `expired` (vs
        `lost`) definition used by ``list_runs``/``get_run``'s run-status split
        and by progress observability. Expiration is no longer a dispenser
        quarantine metric.

        Every expired, unrewarded run for the tuner is counted (no recency
        window).

        When ``run_ids`` is given, both scans are restricted to those runs (used
        by the paginated run listing so it doesn't scan every row).
        """
        # Signal 1: lingering in-flight op.
        in_flight_stmt = (
            select(InFlightChatCompletionModel.run_id, RunModel.datum_id)
            .join(RunModel, RunModel.id == InFlightChatCompletionModel.run_id)
            .where(
                InFlightChatCompletionModel.tuner_id == tuner_id,
                RunModel.reward.is_(None),
                RunModel.expires_at <= now,
            )
            .distinct()
        )
        if run_ids is not None:
            in_flight_stmt = in_flight_stmt.where(
                InFlightChatCompletionModel.run_id.in_(run_ids)
            )

        # Signal 2: total recorded generation time past the expiration
        # threshold.
        duration_stmt = (
            select(ChatCompletionModel.run_id, RunModel.datum_id)
            .join(RunModel, RunModel.id == ChatCompletionModel.run_id)
            .where(
                ChatCompletionModel.tuner_id == tuner_id,
                RunModel.reward.is_(None),
                RunModel.expires_at <= now,
            )
            .group_by(ChatCompletionModel.run_id, RunModel.datum_id)
            .having(
                func.coalesce(func.sum(ChatCompletionModel.duration_ms), 0)
                >= RUN_EXPIRE_GENERATION_BUDGET_MS
            )
        )
        if run_ids is not None:
            duration_stmt = duration_stmt.where(ChatCompletionModel.run_id.in_(run_ids))

        expired: Dict[str, str] = {}
        for stmt in (in_flight_stmt, duration_stmt):
            result = await session.execute(stmt)
            for run_id, datum_id in result.all():
                if run_id is not None:
                    expired[run_id] = datum_id
        return expired
