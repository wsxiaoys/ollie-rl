import asyncio
import base64
import binascii
import hashlib
import json
import logging
import math
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Tuple

from sqlalchemy import func, select, update

from ollie_rl.cookbook import Cookbook, Recipe
from ollie_rl.trainer import Trainer, StateStore, Example
from ollie_rl.trainer import factory as trainer_factory
from ollie_rl.db import TunerModel, ChatCompletionModel, DatumRowModel
from ollie_rl.db.connection import get_sessionmaker
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from openai.types.chat import ChatCompletion
from ollie_rl.types import (
    Rollout,
    RolloutRun,
    DispenseRun,
    ChatCompletionRequest,
    ChatCompletionItem,
    GetTunerResponse,
    TunerItem,
    TrainingProgress,
    BatchProgress,
    DatumProgress,
    DatumCoverage,
    DatumPool,
    RunProgress,
    RunItem,
    RunStatus,
    ListRunsResponse,
    RunDetailResponse,
    ChatCompletionDetailResponse,
    NextPick,
)

logger = logging.getLogger(__name__)


class TunerNotFoundError(Exception):
    pass


class InvalidRunCursorError(Exception):
    """Raised when a runs pagination cursor cannot be decoded."""

    pass


class RunNotFoundError(Exception):
    pass


class ChatCompletionNotFoundError(Exception):
    pass


class RunExpiredError(Exception):
    pass


class RewardAlreadySetError(Exception):
    pass


class EmptyRunError(Exception):
    """Raised when a reward is submitted for a run with no chat completions.

    A run that produced zero completions carries no training signal, so
    rewarding it is meaningless. We reject the reward outright; the run's
    lease simply expires and the datum is re-dispensed for a fresh attempt.
    """

    pass


class MalformedSampleError(Exception):
    def __init__(self, message: str, raw_content: Optional[str] = None):
        super().__init__(message)
        self.raw_content = raw_content


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
                select(TunerModel.trainer_state).where(TunerModel.id == self._tuner_id)
            )
            return result.scalar_one_or_none()

    async def save(self, trainer_state: str) -> None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            async with session.begin():
                await session.execute(
                    update(TunerModel)
                    .where(TunerModel.id == self._tuner_id)
                    .values(trainer_state=trainer_state)
                )
        logger.debug(f"Persisted state for tuner {self._tuner_id}")


def _scheduler_scores(
    datum_pool: List[str],
    runs: List[RunModel],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Scheduler-view consumable score per datum (no staleness filter).

    Returns ``(score, trained)`` where ``score[datum]`` counts runs that
    are still *consumable* by a future train step from the scheduler's
    point of view (not yet trained, not requeued, and either rewarded or
    still pending/not expired) and ``trained[datum]`` accumulates prior
    training exposure for the fresh-tier tie-break.

    Shared by ``_pick_datum`` (dispense decision) and the progress builder
    (``next_pick`` labeling) so the two never drift.
    """
    now = utcnow()
    score = {d: 0 for d in datum_pool}
    trained = {d: 0 for d in datum_pool}
    for r in runs:
        if r.datum_id not in score:
            continue
        if r.trained_count > 0:
            # Track prior training exposure for the fresh-tier tie-break.
            trained[r.datum_id] += r.trained_count
            continue
        if r.rejected_count > 0:
            continue
        has_reward = r.reward is not None
        is_pending = r.reward is None and r.expires_at > now
        if has_reward or is_pending:
            score[r.datum_id] += 1
    return score, trained


def _pick_tier(
    datum: str, score: Dict[str, int], recipe: Recipe
) -> Tuple[Literal["incomplete", "fresh", "saturated", "none"], str]:
    """Label the scheduler tier (+ human reason) for a candidate datum.

    Mirrors the tiers in ``_pick_datum.priority`` so a dispense preview can
    explain *why* a datum would be chosen next.
    """
    count = score.get(datum, 0)
    group_size = recipe.group_size
    if 0 < count < group_size:
        return "incomplete", f"closest-to-complete group ({count}/{group_size})"
    if count == 0:
        return "fresh", "starting a new group from a fresh (least-trained) datum"
    if recipe.max_off_policy_generation > 0:
        return (
            "saturated",
            f"group full ({count}/{group_size}); dispensing off-policy surplus",
        )
    return "none", "all groups saturated; on-policy surplus would be requeued"


def _pick_datum(
    datum_pool: List[str],
    runs: List[RunModel],
    recipe: Recipe,
) -> Optional[str]:
    """Pick the next datum to dispense a run for.

    Pure scheduling helper (no service/DB state) so it can be reasoned about
    and unit-tested in isolation.

    Uses a greedy "most-full-first" strategy via tiered priority. Only runs
    that are still *consumable* by a future train step are counted, i.e.
    not yet trained (``trained_count <= 0``), not requeued
    (``rejected_count <= 0``), and either rewarded or pending (not expired).
    This mirrors ``TunerService._collect_consumable_batch`` so a datum whose
    group was already trained resets to "fresh" for the next generation.

    1. Started-but-incomplete groups (0 < count < group_size) come first,
       ordered by highest count, so the closest-to-complete group finishes
       ASAP. This minimizes the number of in-flight partial groups and gets
       complete groups ready for training as soon as possible.
    2. Fresh datums (count == 0) come next, so we start new distinct groups
       before over-producing existing ones. Among fresh datums the
       least-trained one wins, so never-trained datums are sampled before
       re-sampling datums that already contributed a trained group (better
       dataset coverage).
    3. Saturated datums (count >= group_size) come last, and only when
       off-policy samples are allowed (``max_off_policy_generation > 0``):
       the surplus runs can be consumed by a later train step within the
       off-policy window. Among saturated datums we prefer the
       least-saturated to spread the surplus. When off-policy is disabled
       the surplus would just be requeued, so saturated datums are excluded
       and ``None`` is returned if nothing else is available.
    """
    if not datum_pool:
        return None

    group_size = recipe.group_size
    allow_surplus = recipe.max_off_policy_generation > 0

    score, trained = _scheduler_scores(datum_pool, runs)

    def priority(datum: str) -> Tuple[int, int]:
        count = score[datum]
        if 0 < count < group_size:
            # Started but incomplete: finish closest-to-complete first.
            return (2, count)
        if count == 0:
            # Fresh: start new distinct groups before over-producing, and
            # prefer the least-trained datum so never-trained ones go first.
            return (1, -trained[datum])
        # Saturated (count >= group_size): the group is already complete, so
        # any further runs are surplus. Only dispatchable as off-policy
        # samples for a later train step; spread across the least-saturated.
        if allow_surplus:
            return (0, -count)
        # Strictly on-policy: surplus would be requeued, so don't dispatch.
        return (-1, 0)

    best = max(datum_pool, key=priority)
    if priority(best)[0] < 0:
        # All datums saturated and off-policy surplus is not allowed.
        return None
    return best


def _last_train_op_duration_seconds(state_data: object) -> Optional[float]:
    """Derive the most recent completed train op's execution time (seconds).

    Reads the LRO timing captured under
    ``last_train_op.metadata.{create_time, update_time}`` and returns
    ``update_time - create_time``. Robust to camelCase serialization and
    tolerant of missing/partial timing (returns ``None``). Trainers that don't
    persist a ``last_train_op`` (e.g. inline backends) yield ``None``.
    """
    if not isinstance(state_data, dict):
        return None
    op = state_data.get("last_train_op")
    if not isinstance(op, dict):
        return None
    meta = op.get("metadata")
    if not isinstance(meta, dict):
        return None
    create = meta.get("create_time") or meta.get("createTime")
    update = meta.get("update_time") or meta.get("updateTime")
    if not isinstance(create, str) or not isinstance(update, str):
        return None
    try:
        start = datetime.fromisoformat(create.replace("Z", "+00:00"))
        end = datetime.fromisoformat(update.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (end - start).total_seconds()


def _run_status(run: RunModel, now: datetime) -> RunStatus:
    """Derive a single mutually-exclusive lifecycle label for a run.

    Priority mirrors how the bookkeeping columns accumulate: a run that
    has been trained or requeued (rejected) takes precedence over its
    reward/lease state.
    """
    if run.trained_count > 0:
        return "trained"
    if run.rejected_count > 0:
        return "rejected"
    if run.reward is not None:
        return "rewarded"
    if run.expires_at > now:
        return "in_flight"
    return "expired"


def _encode_run_cursor(created_at: datetime, run_id: str) -> str:
    """Encode a ``(created_at, id)`` run position into an opaque cursor.

    The two fields form the stable sort key used by ``list_runs``; base64 keeps
    the token opaque so clients treat it as a handle rather than parsing it.
    """
    raw = f"{created_at.isoformat()}|{run_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def _decode_run_cursor(cursor: str) -> Tuple[datetime, str]:
    """Decode a cursor produced by :func:`_encode_run_cursor`.

    Raises ``InvalidRunCursorError`` for malformed tokens so the API layer can
    surface a 400 rather than a 500.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        created_at_str, run_id = raw.rsplit("|", 1)
        return datetime.fromisoformat(created_at_str), run_id
    except (ValueError, UnicodeDecodeError, binascii.Error) as e:
        raise InvalidRunCursorError(f"Invalid runs cursor: {cursor!r}") from e


def _build_run_item(
    run: RunModel,
    completion_count: int,
    now: datetime,
    policy_generation: Optional[int] = None,
) -> RunItem:
    return RunItem(
        run_id=run.id,
        datum_id=run.datum_id,
        status=_run_status(run, now),
        reward=run.reward,
        policy_generation=policy_generation,
        trained_count=run.trained_count,
        rejected_count=run.rejected_count,
        completion_count=completion_count,
        created_at=run.created_at,
        expires_at=run.expires_at,
    )


def _request_hash(request: "ChatCompletionRequest") -> str:
    """
    Stable SHA-256 digest of a request's prompt.

    The prompt is everything the model conditions on: the ``messages`` *and*
    the available ``tools`` (different tool schemas can yield different
    responses for the same messages, so they must be part of the key).

    Retries of a stalled request re-send the identical prompt, so this gives a
    per-turn idempotency key: within a linear agent run, a repeat digest is
    always a retry of the same turn. Fields are dumped in JSON mode with sorted
    keys so semantically identical prompts hash the same regardless of dict
    ordering.
    """
    dumped = request.model_dump(mode="json")
    key = {
        "messages": dumped.get("messages", []),
        "tools": dumped.get("tools"),
    }
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _KeyedLocks:
    """
    A manager of per-key ``asyncio.Lock``s with reference-counted cleanup.

    Used to serialize sampling per ``(tuner_id, run_id, request_hash)`` so
    concurrent retries of the same turn don't both generate + record a
    duplicate sibling completion. Locks are created lazily and dropped once
    no coroutine holds or waits on them, so the table doesn't grow unbounded
    across the many distinct turns of a training run.
    """

    def __init__(self) -> None:
        self._locks: Dict[Any, asyncio.Lock] = {}
        self._refcounts: Dict[Any, int] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: Any) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            self._refcounts[key] = self._refcounts.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                self._refcounts[key] -= 1
                if self._refcounts[key] <= 0:
                    self._refcounts.pop(key, None)
                    self._locks.pop(key, None)


class TunerService:
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
        # Lock to prevent race conditions during lazy restoration/materialization of trainers.
        self._materialize_lock = asyncio.Lock()
        # Lock serializing the read-pick-insert critical section of `dispense_run`
        # so concurrent dispenses observe each other's in-flight runs and don't
        # all pick the same datum (which would over-dispense past `group_size`).
        self._dispense_lock = asyncio.Lock()
        # Per-(tuner_id, run_id, request_hash) locks that make sampling
        # idempotent: a retry of a stalled request must replay the stored
        # completion instead of generating a duplicate sibling. Serializing on
        # this key closes the check-then-record race between concurrent
        # retries of the same turn.
        self._sample_locks = _KeyedLocks()
        # Background task that periodically polls every tuner and triggers a
        # train step when a batch is ready, replacing the per-reward
        # fire-and-forget trigger.
        self._train_loop_task: Optional[asyncio.Task] = None

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
            asyncio.create_task(_train_one(tuner_id))

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

        async with self._materialize_lock:
            # Double-checked locking pattern: A concurrent request might have
            # finished materializing the trainer while we were waiting for the lock.
            if tuner_id in self.active_trainers:
                return self.active_trainers[tuner_id]

            return await self._materialize(tuner_id)

    async def get_tuner_details(
        self, tuner_id: str, include_progress: bool = False
    ) -> GetTunerResponse:
        """
        Retrieve tuner details, including current policy_generation and stored trainer state.

        When `include_progress` is set, a recipe-aware `TrainingProgress`
        snapshot is computed and attached (extra DB reads).
        """
        import json

        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            record = result.scalar_one_or_none()

        if record is None:
            raise TunerNotFoundError(f"Tuner '{tuner_id}' not found.")

        trainer = await self._get_trainer(tuner_id)
        policy_generation = trainer.policy_generation

        state_data = None
        if record.trainer_state:
            try:
                state_data = json.loads(record.trainer_state)
            except json.JSONDecodeError:
                state_data = record.trainer_state

        progress = None
        if include_progress:
            progress = await self.get_progress(tuner_id)

        return GetTunerResponse(
            tuner_id=record.id,
            name=record.name,
            recipe=Cookbook.get(record.recipe),
            trainer=record.trainer,
            policy_generation=policy_generation,
            trainer_state=state_data,
            progress=progress,
            is_training=await trainer.is_training(),
            last_train_op_duration_seconds=_last_train_op_duration_seconds(state_data),
        )

    async def get_progress(self, tuner_id: str) -> TrainingProgress:
        """
        Build a recipe-aware training-progress snapshot for `tuner_id`.

        Batch readiness and per-datum group coverage use the *trainer view*
        (mirrors `_collect_consumable_batch`, including the off-policy
        staleness filter) so it accurately reflects how close the next
        train step is. `next_pick` uses the *scheduler view* (mirrors
        `_pick_datum`) since that is what actually drives dispensing.
        """
        trainer = await self._get_trainer(tuner_id)
        recipe = await self._recipe_for(tuner_id)
        generation = trainer.policy_generation

        async with self.async_session() as session:
            datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)

            # Progress only needs each run's policy generation to apply the
            # off-policy staleness filter below, so aggregate it in SQL instead
            # of hydrating full `ChatCompletionModel` rows. The full rows carry
            # the `request`/`response` JSON and `tokens`/`logprobs` blobs
            # (tens of KB each); loading them for every completion just to read
            # one integer dominated this (frequently polled) endpoint. We take
            # the max generation per run, mirroring how `list_runs` labels a
            # run's generation.
            generation_by_run_id: Dict[str, int] = {}
            if runs:
                result = await session.execute(
                    select(
                        ChatCompletionModel.run_id,
                        func.max(ChatCompletionModel.policy_generation),
                    )
                    .where(ChatCompletionModel.tuner_id == tuner_id)
                    .group_by(ChatCompletionModel.run_id)
                )
                generation_by_run_id = {
                    run_id: max_generation
                    for run_id, max_generation in result.all()
                    if run_id is not None
                }

        now = utcnow()
        max_off = recipe.max_off_policy_generation
        group_size = recipe.group_size

        in_flight = expired = rewarded = consumable = trained = rejected = 0
        consumable_by_datum: Dict[str, int] = {d: 0 for d in datum_pool}
        in_flight_by_datum: Dict[str, int] = {d: 0 for d in datum_pool}
        trained_by_datum: Dict[str, int] = {d: 0 for d in datum_pool}

        for r in runs:
            rewarded_flag = r.reward is not None
            if rewarded_flag:
                rewarded += 1
            elif r.expires_at > now:
                in_flight += 1
                if r.datum_id in in_flight_by_datum:
                    in_flight_by_datum[r.datum_id] += 1
            else:
                expired += 1

            if r.trained_count > 0:
                trained += 1
                if r.datum_id in trained_by_datum:
                    trained_by_datum[r.datum_id] += r.trained_count
            if r.rejected_count > 0:
                rejected += 1

            # Trainer-view consumable: rewarded, not trained, not rejected,
            # and within the off-policy window.
            if rewarded_flag and r.trained_count <= 0 and r.rejected_count <= 0:
                run_generation = generation_by_run_id.get(r.id)
                if run_generation is None or (generation - run_generation <= max_off):
                    consumable += 1
                    if r.datum_id in consumable_by_datum:
                        consumable_by_datum[r.datum_id] += 1

        items: List[DatumProgress] = []
        groups_ready = 0
        groups_in_progress = 0
        datums_in_progress = 0
        for datum_id in consumable_by_datum:
            count = consumable_by_datum[datum_id]
            pending = in_flight_by_datum.get(datum_id, 0)
            trained_here = trained_by_datum.get(datum_id, 0)
            # Surface any datum that has activity worth showing: a group
            # forming (rewarded runs counting toward the batch, or runs still
            # awaiting a reward) or one that has already contributed a trained
            # group. Without the trained check a datum whose group was fully
            # trained (consumable/in-flight back to 0) would silently vanish
            # from the pool even though it carries training history.
            if count <= 0 and pending <= 0 and trained_here <= 0:
                continue
            # "in progress" and batch readiness only reflect datums with an
            # active (consumable or in-flight) group. A purely trained datum is
            # listed for visibility but isn't forming a new group, so it must
            # not inflate these counters.
            if count > 0 or pending > 0:
                datums_in_progress += 1
                ready = count >= group_size
                if ready:
                    groups_ready += 1
                else:
                    # Not-yet-ready group with >=1 consumable or in-flight run.
                    groups_in_progress += 1
            items.append(
                DatumProgress(
                    datum_id=datum_id,
                    consumable=count,
                    in_flight=pending,
                    trained=trained_here,
                )
            )
        items.sort(key=lambda g: (g.consumable, g.in_flight), reverse=True)

        never_trained = sum(1 for d in datum_pool if trained_by_datum.get(d, 0) == 0)

        score, _ = _scheduler_scores(datum_pool, runs)
        picked = _pick_datum(datum_pool, runs, recipe)
        if picked is None:
            next_pick = NextPick(
                datum_id=None,
                tier="none",
                reason="no datum dispensable (pool empty or all groups saturated on-policy)",
            )
        else:
            tier, reason = _pick_tier(picked, score, recipe)
            next_pick = NextPick(datum_id=picked, tier=tier, reason=reason)

        return TrainingProgress(
            batch=BatchProgress(
                groups_ready=groups_ready,
                groups_in_progress=groups_in_progress,
            ),
            runs=RunProgress(
                total=len(runs),
                in_flight=in_flight,
                expired=expired,
                rewarded=rewarded,
                consumable=consumable,
                trained=trained,
                rejected=rejected,
            ),
            data=DatumPool(
                coverage=DatumCoverage(
                    in_progress=datums_in_progress,
                    never_trained=never_trained,
                ),
                items=items,
            ),
            next_pick=next_pick,
        )

    async def list_tuners(self) -> List[TunerItem]:
        """
        Retrieve all tuners, including their current policy_generation.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(TunerModel).where(TunerModel.trainer_state.is_not(None))
            )
            records = result.scalars().all()

        tuners_list = []
        for record in records:
            try:
                trainer = await self._get_trainer(record.id)
                tuners_list.append(
                    TunerItem(
                        tuner_id=record.id,
                        name=record.name,
                        trainer=record.trainer,
                        policy_generation=trainer.policy_generation,
                    )
                )
            except Exception:
                logger.exception(f"Failed to get trainer for tuner '{record.id}'")

        return tuners_list

    async def list_datums(self, tuner_id: str) -> List[str]:
        """Return the full datum-id pool registered for ``tuner_id``.

        Used to populate the runs filter dropdown. The pool is the static set
        of datums the tuner was created with, returned in a stable
        (alphabetical) order so the dropdown listing is deterministic.
        """
        async with self.async_session() as session:
            exists = await session.execute(
                select(TunerModel.id).where(TunerModel.id == tuner_id)
            )
            if exists.scalar_one_or_none() is None:
                raise TunerNotFoundError(f"Tuner '{tuner_id}' not found.")

            result = await session.execute(
                select(DatumRowModel.datum_id)
                .where(DatumRowModel.tuner_id == tuner_id)
                .order_by(DatumRowModel.datum_id.asc())
            )
            return list(result.scalars().all())

    async def list_runs(
        self,
        tuner_id: str,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        datum_id: Optional[str] = None,
    ) -> ListRunsResponse:
        """
        List runs for a tuner (newest first), each with its derived lifecycle
        status and the number of recorded chat completions.

        Pagination is cursor-based over the stable ``(created_at, id)`` ordering
        (newest first). Pass ``limit`` to bound the page and ``cursor`` (from a
        previous response's ``next_cursor``) to fetch the runs immediately after
        the last item of that page. Leave ``limit`` as ``None`` to return every
        run in one shot (``next_cursor`` is then always ``None``).

        Pass ``datum_id`` to restrict the listing to runs dispensed for that
        datum; the filter composes with cursor-based pagination.
        """
        if limit is not None and limit < 0:
            limit = 0

        cursor_key = _decode_run_cursor(cursor) if cursor else None

        async with self.async_session() as session:
            exists = await session.execute(
                select(TunerModel.id).where(TunerModel.id == tuner_id)
            )
            if exists.scalar_one_or_none() is None:
                raise TunerNotFoundError(f"Tuner '{tuner_id}' not found.")

            runs_stmt = (
                select(RunModel)
                .where(RunModel.tuner_id == tuner_id)
                .order_by(RunModel.created_at.desc(), RunModel.id.desc())
            )
            if datum_id is not None:
                runs_stmt = runs_stmt.where(RunModel.datum_id == datum_id)
            if cursor_key is not None:
                cursor_created_at, cursor_id = cursor_key
                # Rows strictly "after" the cursor in (created_at DESC, id DESC)
                # order: an older timestamp, or the same timestamp with a
                # smaller id (the tie-breaker keeps paging deterministic when
                # multiple runs share a created_at).
                runs_stmt = runs_stmt.where(
                    (RunModel.created_at < cursor_created_at)
                    | (
                        (RunModel.created_at == cursor_created_at)
                        & (RunModel.id < cursor_id)
                    )
                )
            if limit is not None:
                # Fetch one extra row to detect whether another page exists
                # without a separate count query.
                runs_stmt = runs_stmt.limit(limit + 1)

            runs_result = await session.execute(runs_stmt)
            runs = list(runs_result.scalars().all())

            has_more = False
            if limit is not None and len(runs) > limit:
                has_more = True
                runs = runs[:limit]

            counts: Dict[str, int] = {}
            generations: Dict[str, int] = {}
            if runs:
                # One grouped pass yields both the completion count and the
                # run's policy generation (max across its completions), so the
                # runs list can bucket rewards by generation without an extra
                # per-run fetch. Scope the aggregate to the page's run ids so a
                # paginated request doesn't scan every completion.
                run_ids = [r.id for r in runs]
                agg_result = await session.execute(
                    select(
                        ChatCompletionModel.run_id,
                        func.count(),
                        func.max(ChatCompletionModel.policy_generation),
                    )
                    .where(
                        ChatCompletionModel.tuner_id == tuner_id,
                        ChatCompletionModel.run_id.in_(run_ids),
                    )
                    .group_by(ChatCompletionModel.run_id)
                )
                for run_id, count, max_generation in agg_result.all():
                    if run_id is None:
                        continue
                    counts[run_id] = count
                    if max_generation is not None:
                        generations[run_id] = max_generation

        now = utcnow()
        items = [
            _build_run_item(r, counts.get(r.id, 0), now, generations.get(r.id))
            for r in runs
        ]
        next_cursor = (
            _encode_run_cursor(runs[-1].created_at, runs[-1].id)
            if has_more and runs
            else None
        )
        return ListRunsResponse(runs=items, next_cursor=next_cursor)

    async def get_run_details(self, tuner_id: str, run_id: str) -> RunDetailResponse:
        """
        Return a single run plus its chat completions (oldest first) so the
        full request/response transcript can be visualized.
        """
        async with self.async_session() as session:
            run_result = await session.execute(
                select(RunModel).where(
                    RunModel.tuner_id == tuner_id,
                    RunModel.id == run_id,
                )
            )
            run = run_result.scalar_one_or_none()
            if run is None:
                raise RunNotFoundError(
                    f"Run '{run_id}' not found under tuner '{tuner_id}'"
                )

            comp_result = await session.execute(
                select(ChatCompletionModel)
                .where(
                    ChatCompletionModel.tuner_id == tuner_id,
                    ChatCompletionModel.run_id == run_id,
                )
                .order_by(ChatCompletionModel.created_at.asc())
            )
            completions = list(comp_result.scalars().all())

        now = utcnow()
        policy_generation = (
            max(c.policy_generation for c in completions) if completions else None
        )
        run_item = _build_run_item(run, len(completions), now, policy_generation)
        completion_items = [
            ChatCompletionItem(
                id=c.id,
                policy_generation=c.policy_generation,
                created_at=c.created_at,
                request=ChatCompletionRequest.model_validate(c.request),
                response=ChatCompletion.model_validate(c.response),
            )
            for c in completions
        ]
        return RunDetailResponse(run=run_item, completions=completion_items)

    async def get_completion_details(
        self, tuner_id: str, run_id: str, completion_id: str
    ) -> ChatCompletionDetailResponse:
        """
        Return a single recorded chat completion (request, response, and the
        optional sample-time tensors) so it can be inspected in isolation.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel).where(
                    ChatCompletionModel.tuner_id == tuner_id,
                    ChatCompletionModel.run_id == run_id,
                    ChatCompletionModel.id == completion_id,
                )
            )
            record = result.scalar_one_or_none()

        if record is None:
            raise ChatCompletionNotFoundError(
                f"Chat completion '{completion_id}' not found under run "
                f"'{run_id}' of tuner '{tuner_id}'"
            )

        return ChatCompletionDetailResponse(
            id=record.id,
            tuner_id=record.tuner_id,
            run_id=record.run_id or run_id,
            datum_id=record.datum_id,
            policy_generation=record.policy_generation,
            created_at=record.created_at,
            request=ChatCompletionRequest.model_validate(record.request),
            response=ChatCompletion.model_validate(record.response),
            tokens=record.tokens,
            logprobs=record.logprobs,
        )

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
        state_store = _DbStateStore(tuner_id)
        factory = trainer_factory.get(trainer)

        trainer_instance = await factory.restore(record.name, state_store)
        self.active_trainers[tuner_id] = trainer_instance
        return trainer_instance

    async def create_tuner(
        self,
        recipe: str,
        name: str,
        datum_ids: List[str],
        trainer: str,
        trainer_params: Optional[dict] = None,
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
                    trainer_state=None,
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
        trainer_instance = await factory.create(
            name, state_store, trainer_params=trainer_params
        )
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
        trainer = await self._get_trainer(tuner_id)
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

        # Ad-hoc sampling (no run to record against) never duplicates, so skip
        # the idempotency machinery entirely.
        if run_id is None:
            sample_op = await trainer.sample(request)
            sample = await sample_op.wait()
            return sample.completion

        assert datum_id is not None

        # Make sampling idempotent per turn. A slow/cancelled request is
        # retried by the client with the identical prompt; because an agent
        # run is linear, a repeat `(tuner_id, run_id, request_hash)` is always
        # such a retry. Serialize on that key so concurrent retries can't race
        # the check-then-record window, and replay any already-recorded
        # completion instead of generating a duplicate sibling (which would
        # fork the trajectory and double-count in training).
        request_hash = _request_hash(request)
        async with self._sample_locks.acquire((tuner_id, run_id, request_hash)):
            existing = await self._find_recorded_completion(
                tuner_id, run_id, request_hash
            )
            if existing is not None:
                logger.info(
                    f"Replaying recorded completion for run {run_id} "
                    f"(request_hash={request_hash[:12]}...); skipping resample."
                )
                return existing

            # Generate completion
            sample_op = await trainer.sample(request)
            sample = await sample_op.wait()
            policy_generation = sample.policy_generation

            # Record completion metadata
            completion_id = f"cmpl_{uuid.uuid4().hex}"
            await self.record_chat_completion(
                completion_id=completion_id,
                tuner_id=tuner_id,
                run_id=run_id,
                datum_id=datum_id,
                policy_generation=policy_generation,
                tokens=sample.tokens,
                logprobs=sample.logprobs,
                request=request,
                response=sample.completion,
                request_hash=request_hash,
            )
            if sample.malformed:
                recipe = await self._recipe_for(tuner_id)
                malformed_penalty = recipe.malformed_penalty
                await self.update_reward(tuner_id, run_id, reward=malformed_penalty)
                raw_content = None
                if sample.completion.choices and sample.completion.choices[0].message:
                    raw_content = sample.completion.choices[0].message.content
                raise MalformedSampleError(
                    f"Malformed sample on run {run_id}; reward set to {malformed_penalty}",
                    raw_content=raw_content,
                )

            return sample.completion

    async def _find_recorded_completion(
        self, tuner_id: str, run_id: str, request_hash: str
    ) -> Optional[ChatCompletion]:
        """
        Return the completion already recorded for this exact turn, if any.

        Backs `sample()`'s idempotent replay: when a retry re-sends the
        identical prompt, the stored completion for
        `(tuner_id, run_id, request_hash)` is returned verbatim instead of
        resampling. The earliest row wins so replays stay stable.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(ChatCompletionModel)
                .where(
                    ChatCompletionModel.tuner_id == tuner_id,
                    ChatCompletionModel.run_id == run_id,
                    ChatCompletionModel.request_hash == request_hash,
                )
                .order_by(ChatCompletionModel.created_at.asc())
                .limit(1)
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            return ChatCompletion.model_validate(record.response)

    async def record_chat_completion(
        self,
        completion_id: str,
        tuner_id: str,
        run_id: str,
        datum_id: str,
        policy_generation: int,
        request: ChatCompletionRequest,
        response: ChatCompletion,
        tokens: Optional[List[int]] = None,
        logprobs: Optional[List[float]] = None,
        request_hash: Optional[str] = None,
    ) -> None:
        """
        Record a chat completion event in the database.

        `tokens` and `logprobs` are optional sample-time tensors required
        by trainers (e.g. Tinker) that train on raw rollouts. They are
        stored as JSON-encoded text and may be NULL for trainers that do
        not provide them.

        `request_hash` is the per-turn idempotency key (SHA-256 of the
        request messages) used by `sample()` to replay this completion for a
        retried request. It may be NULL for direct callers that don't dedup.
        """
        async with self.async_session() as session:
            async with session.begin():
                db_completion = ChatCompletionModel(
                    id=completion_id,
                    tuner_id=tuner_id,
                    run_id=run_id,
                    datum_id=datum_id,
                    policy_generation=policy_generation,
                    tokens=tokens,
                    logprobs=logprobs,
                    request=request.model_dump(mode="json"),
                    response=response.model_dump(mode="json"),
                    request_hash=request_hash,
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

                # A run with zero recorded completions carries no training
                # signal, so a reward for it is useless. Reject it instead of
                # persisting a rewardless-but-scored run that would otherwise
                # count toward a training group.
                completion_count = await session.scalar(
                    select(func.count())
                    .select_from(ChatCompletionModel)
                    .where(
                        ChatCompletionModel.tuner_id == tuner_id,
                        ChatCompletionModel.run_id == run_id,
                    )
                )
                if not completion_count:
                    raise EmptyRunError(
                        f"Run '{run_id}' has no chat completions; refusing to "
                        f"record a reward for an empty run"
                    )

                record.reward = reward
                record.updated_at = now
        logger.info(f"Successfully recorded reward {reward} for run {run_id}")

    async def _maybe_train(self, tuner_id: str) -> None:
        """
        Attempt to start (and wait for) a train step for `tuner_id`.

        Serialized per-tuner via `self._train_lock_for` so only one train step
        runs at a time for a given tuner, while distinct tuners train
        concurrently.
        """
        async with self._train_lock_for(tuner_id):
            trainer = await self._get_trainer(tuner_id)

            # Skip if a train step is already in progress for this trainer.
            if await trainer.is_training():
                logger.info(
                    f"Skipping train step for tuner {tuner_id}: already training"
                )
                return

            train_op = None
            async with self.async_session() as session:
                async with session.begin():
                    batch, run_ids = await self._collect_consumable_batch(
                        tuner_id, session, trainer
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
        self, tuner_id: str, session, trainer: Trainer
    ) -> Tuple[List[Example], List[str]]:
        recipe = await self._recipe_for(tuner_id)

        result = await session.execute(
            select(RunModel).where(
                RunModel.tuner_id == tuner_id,
                RunModel.trained_count <= 0,
                RunModel.rejected_count <= 0,
                RunModel.reward != None,  # noqa: E711
            )
        )
        run_records = list(result.scalars().all())

        if not run_records:
            return [], []

        # 1. Retrieve ChatCompletions for all candidate runs to check for staleness
        candidate_run_ids = [r.id for r in run_records]
        result = await session.execute(
            select(ChatCompletionModel).where(
                ChatCompletionModel.tuner_id == tuner_id,
                ChatCompletionModel.run_id.in_(candidate_run_ids),
            )
        )
        completions = result.scalars().all()
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
        # payload. The row primary key (`c.id`) is a synthetic internal id and
        # must NOT leak to a training backend; fall back to it only if the
        # response somehow lacks an id.
        examples = []
        for c in completions:
            if c.run_id not in run_advantages:
                continue
            candidate_id = (
                c.response.get("id") if isinstance(c.response, dict) else None
            )
            examples.append(
                Example(
                    chat_completion_id=candidate_id or c.id,
                    advantage=run_advantages[c.run_id],
                    policy_generation=c.policy_generation,
                    tokens=c.tokens,
                    logprobs=c.logprobs,
                )
            )

        return examples, run_ids

    async def dispense_run(self, tuner_id: str) -> Optional[DispenseRun]:
        """
        Dispense a run for a tuner.
        """
        # Ensure trainer is initialized.
        _trainer = await self._get_trainer(tuner_id)
        recipe = await self._recipe_for(tuner_id)

        # Serialize the read-pick-insert sequence: the scheduler decision in
        # `_pick_datum` depends on the current set of in-flight runs, so two
        # concurrent dispenses that both read the pre-insert snapshot would
        # otherwise pick the same datum and over-dispense it past
        # `group_size` (the source of the inflated in_flight counts).
        async with self._dispense_lock:
            async with self.async_session() as session:
                datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)

            datum_id = _pick_datum(datum_pool, runs, recipe)
            if datum_id is None:
                return None

            run_record = RunModel(
                tuner_id=tuner_id,
                datum_id=datum_id,
                reward=None,
                trained_count=0,
                expires_at=utcnow() + timedelta(seconds=1200),
            )
            async with self.async_session() as session:
                async with session.begin():
                    session.add(run_record)

        return DispenseRun(
            run_id=run_record.id,
            datum_id=run_record.datum_id,
            expires_at=run_record.expires_at,
        )

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
