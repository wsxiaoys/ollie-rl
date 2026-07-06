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
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from sqlalchemy import delete, func, select, update

from ollie_rl.background import BackgroundJob
from ollie_rl.cookbook import Cookbook, Recipe
from ollie_rl.service.dispense import (
    RewardedRun,
    TerminalStats,
    pick_datum,
    pick_tier,
    quarantined_datums,
    scheduler_scores,
    terminal_stats,
)
from ollie_rl.trainer import Trainer, StateStore, Example
from ollie_rl.trainer import factory as trainer_factory
from ollie_rl.db import (
    TunerModel,
    ChatCompletionModel,
    InFlightChatCompletionModel,
    DatumRowModel,
)
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

# Compute-based signal for the `expired` (vs `lost`) classification: a
# run that has accumulated at least this much total generation time (the summed
# `duration_ms` across its recorded completions) without ever earning a reward
# is treated as `expired` even if no in-flight op lingers. It burned real
# compute yet never finished -- the same waste the `expired` label flags -- so it
# should not be dismissed as merely `lost`. Measured in milliseconds. Like the
# in-flight-op signal, it honors the dispenser's recency (policy-generation)
# window when one is supplied.
RUN_EXPIRE_GENERATION_BUDGET_MS = 15 * 60 * 1000

# Fixed time budget (seconds) granted to a run at creation. The whole run (all
# turns combined) must finish within this window before it is considered expired;
# the deadline never moves once set.
RUN_LEASE_SECONDS = 3600


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


class ContentFilterSampleError(Exception):
    def __init__(self, message: str, raw_content: Optional[str] = None):
        super().__init__(message)
        self.raw_content = raw_content


class LengthSampleError(Exception):
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


def _run_status(run: RunModel, now: datetime, is_expired: bool) -> RunStatus:
    """Derive a single mutually-exclusive lifecycle label for a run.

    Priority mirrors how the bookkeeping columns accumulate: a run that
    has been trained or requeued (rejected) takes precedence over its
    reward/lease state.

    Once a run is past its lease with no reward, ``is_expired`` splits it into
    ``expired`` vs ``lost``. ``is_expired`` is true when the run either still
    has a lingering ``InFlightChatCompletionModel`` row (the generation itself
    stalled past the lease) or has burned at least ``RUN_EXPIRE_GENERATION_BUDGET_MS`` of
    total generation time without a reward -- both signal wasted compute on a run
    that never finished. Otherwise the run is ``lost`` (crashed/abandoned
    worker, or ops finished but no reward was ever posted).
    """
    if run.trained_count > 0:
        return "trained"
    if run.rejected_count > 0:
        return "rejected"
    if run.reward is not None:
        return "rewarded"
    if run.expires_at > now:
        return "in_flight"
    return "expired" if is_expired else "lost"


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
    duration_ms_total: Optional[int] = None,
    is_expired: bool = False,
) -> RunItem:
    return RunItem(
        run_id=run.id,
        datum_id=run.datum_id,
        status=_run_status(run, now, is_expired),
        reward=run.reward,
        policy_generation=policy_generation,
        trained_count=run.trained_count,
        rejected_count=run.rejected_count,
        completion_count=completion_count,
        duration_ms_total=duration_ms_total,
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


def _completion_context_tokens(completion: ChatCompletion) -> int:
    """Return prompt + completion + reasoning tokens reported for completion."""
    usage = completion.usage
    if usage is None:
        return 0

    reasoning_tokens = 0
    details = usage.completion_tokens_details
    if details is not None:
        reasoning_tokens = details.reasoning_tokens or 0

    return (
        (usage.prompt_tokens or 0) + (usage.completion_tokens or 0) + reasoning_tokens
    )


def _clear_completion_as_length(completion: ChatCompletion) -> ChatCompletion:
    """Return a copy with every choice converted to an empty length stop."""
    cleared = completion.model_copy(deep=True)
    for choice in cleared.choices:
        choice.finish_reason = "length"
        if choice.message is None:
            continue
        choice.message.content = None
        choice.message.tool_calls = None
        if hasattr(choice.message, "function_call"):
            choice.message.function_call = None
        if hasattr(choice.message, "refusal"):
            choice.message.refusal = None
    return cleared


def _apply_max_context_window(
    completion: ChatCompletion, max_context_window: Optional[int]
) -> ChatCompletion:
    """Convert oversized completions to cleared length samples."""
    if max_context_window is None:
        return completion
    if _completion_context_tokens(completion) <= max_context_window:
        return completion
    return _clear_completion_as_length(completion)


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
        # Holds strong references to fire-and-forget train-step tasks so they
        # aren't garbage-collected mid-flight (see BackgroundJob).
        self._background_jobs = BackgroundJob()

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
        `pick_datum`) since that is what actually drives dispensing.
        """
        trainer = await self._get_trainer(tuner_id)
        recipe = await self._recipe_for(tuner_id)
        generation = trainer.policy_generation
        now = utcnow()
        max_off = recipe.max_off_policy_generation

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

            # Quarantine inputs. Reuse the dispenser's own helpers so the
            # dashboard's expired/rewarded counts match exactly what the
            # dispenser would quarantine on: rewarded runs (denominator's
            # rewarded side, each carrying its reward) and expired, unrewarded
            # runs (the expiration numerator -- a run with a lingering in-flight
            # op or total duration past the expiration threshold).
            # Kept separate from `generation_by_run_id` so the trainer-view
            # consumable calc (which only cares about recorded completions) is
            # unaffected.
            rewarded_by_run: Dict[str, RewardedRun] = {}
            expired_datum_by_run: Dict[str, str] = {}
            if runs:
                rewarded_by_run = await self._rewarded_datums(tuner_id, session)
                expired_datum_by_run = await self._expired_datums(
                    tuner_id, now, session
                )

        group_size = recipe.group_size

        in_flight = expired = lost = rewarded = consumable = trained = rejected = 0
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
            elif r.id in expired_datum_by_run:
                # Expired: generation stalled (lingering in-flight op) or the
                # run's total duration crossed the expiration threshold.
                expired += 1
            else:
                # Neither expiration signal fired: lost/abandoned.
                lost += 1

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

        # Terminal stats per datum, computed from the same two maps the
        # dispenser feeds `terminal_stats`, so the dashboard's expired/rewarded
        # counts match what the quarantine filters act on.
        stats_by_datum = terminal_stats(
            datum_pool,
            rewarded_by_run,
            expired_datum_by_run,
        )

        items: List[DatumProgress] = []
        groups_ready = 0
        groups_in_progress = 0
        datums_in_progress = 0
        for datum_id in consumable_by_datum:
            count = consumable_by_datum[datum_id]
            pending = in_flight_by_datum.get(datum_id, 0)
            trained_here = trained_by_datum.get(datum_id, 0)
            # `terminal_stats` gives per-datum (expired, rewarded, succeeded),
            # the same all-time tallies the dispenser quarantines on. `expired`
            # is the headline "how flaky is this datum" count; rewarded and
            # succeeded let a client derive the expire/success ratios.
            stats = stats_by_datum.get(
                datum_id, TerminalStats(expired=0, rewarded=0, succeeded=0)
            )
            # Surface any datum that has activity worth showing: a group
            # forming (rewarded runs counting toward the batch, or runs still
            # awaiting a reward) or one that has already contributed a trained
            # group. Without the trained check a datum whose group was fully
            # trained (consumable/in-flight back to 0) would silently vanish
            # from the pool even though it carries training history. Expired
            # runs also count as activity worth surfacing: a datum that keeps
            # expiring signals it is hard to finish in time.
            if count <= 0 and pending <= 0 and trained_here <= 0 and stats.expired <= 0:
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
                    expired=stats.expired,
                    rewarded=stats.rewarded,
                    succeeded=stats.succeeded,
                    trained=trained_here,
                )
            )
        items.sort(key=lambda g: (g.consumable, g.in_flight), reverse=True)

        never_trained = sum(1 for d in datum_pool if trained_by_datum.get(d, 0) == 0)

        score, _ = scheduler_scores(datum_pool, runs)
        picked = pick_datum(datum_pool, runs, recipe)
        if picked is None:
            next_pick = NextPick(
                datum_id=None,
                tier="none",
                reason="no datum dispensable (pool empty or all groups saturated on-policy)",
            )
        else:
            tier, reason = pick_tier(picked, score, recipe)
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
                lost=lost,
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
            durations: Dict[str, int] = {}
            if runs:
                # One grouped pass yields the completion count, the run's
                # policy generation (max across its completions), and the total
                # generation latency (sum of durations), so the runs list can
                # bucket rewards by generation and show timing without an extra
                # per-run fetch. Scope the aggregate to the page's run ids so a
                # paginated request doesn't scan every completion.
                run_ids = [r.id for r in runs]
                agg_result = await session.execute(
                    select(
                        ChatCompletionModel.run_id,
                        func.count(),
                        func.max(ChatCompletionModel.policy_generation),
                        func.sum(ChatCompletionModel.duration_ms),
                    )
                    .where(
                        ChatCompletionModel.tuner_id == tuner_id,
                        ChatCompletionModel.run_id.in_(run_ids),
                    )
                    .group_by(ChatCompletionModel.run_id)
                )
                for run_id, count, max_generation, total_duration in agg_result.all():
                    if run_id is None:
                        continue
                    counts[run_id] = count
                    if max_generation is not None:
                        generations[run_id] = max_generation
                    if total_duration is not None:
                        durations[run_id] = int(total_duration)

            # Runs on this page that count as `expired` (unrewarded, past lease,
            # with either a lingering in-flight op or total duration past the
            # expiration threshold) -- the same definition the dispenser uses --
            # so a past-lease
            # unrewarded run can be split into `expired` vs `lost`. Scoped to the
            # page's run ids.
            now = utcnow()
            expired_run_ids = set(
                await self._expired_datums(
                    tuner_id, now, session, run_ids=[r.id for r in runs]
                )
            )

        items = [
            _build_run_item(
                r,
                counts.get(r.id, 0),
                now,
                generations.get(r.id),
                durations.get(r.id),
                r.id in expired_run_ids,
            )
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
            expired_run_ids = set(
                await self._expired_datums(tuner_id, now, session, run_ids=[run_id])
            )

        policy_generation = (
            max(c.policy_generation for c in completions) if completions else None
        )
        durations = [c.duration_ms for c in completions if c.duration_ms is not None]
        duration_ms_total = sum(durations) if durations else None
        run_item = _build_run_item(
            run,
            len(completions),
            now,
            policy_generation,
            duration_ms_total,
            run_id in expired_run_ids,
        )
        completion_items = [
            ChatCompletionItem(
                id=c.id,
                policy_generation=c.policy_generation,
                created_at=c.created_at,
                duration_ms=c.duration_ms,
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
            duration_ms=record.duration_ms,
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

        # Accepted limitation (non-atomic creation): the tuner row is committed
        # with `trainer_state=None` here, then `factory.create(...)` below
        # provisions the backend and persists the real state. A crash/reboot in
        # that window leaves a `trainer_state IS NULL` zombie row that is
        # filtered out of listings but can never materialize. It's harmless
        # (the client just re-creates); tolerated rather than adding a startup
        # sweep or a lifecycle status column.
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
            recipe = await self._recipe_for(tuner_id)
            return _apply_max_context_window(
                sample.completion, recipe.max_context_window
            )

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

            # Middle state: an op may already be in flight for this exact turn
            # from a previous (cancelled/timed-out) attempt. A Gemini LRO keeps
            # progressing on the backend regardless of whether we are polling,
            # so re-attach to it via `restore_state` instead of submitting a
            # fresh op (which would orphan the old one and burn the lease).
            in_flight = await self._find_in_flight_completion(
                tuner_id, run_id, request_hash
            )
            if in_flight is not None:
                logger.info(
                    f"Re-attaching to in-flight op for run {run_id} "
                    f"(request_hash={request_hash[:12]}...); "
                    "continuing to wait rather than resubmitting."
                )
                sample_op = await trainer.sample(request, restore_state=in_flight.state)
                # End-to-end start is the original first-submit time so the
                # recorded latency spans all re-attach cycles.
                first_submit = in_flight.created_at
            else:
                sample_op = await trainer.sample(request)
                first_submit = utcnow()
                state = sample_op.save_state()
                if state is not None:
                    # Resumable backend: persist resume state the moment the op
                    # is submitted so the next retry can re-attach. Stamp the
                    # trainer's current generation so a still-churning run can be
                    # placed on the policy-generation timeline before it records
                    # any completion.
                    await self._record_in_flight_completion(
                        tuner_id,
                        run_id,
                        request_hash,
                        state,
                        first_submit,
                        trainer.policy_generation,
                    )

            try:
                sample = await sample_op.wait()
            except asyncio.CancelledError, TimeoutError:
                # The op is still running on the backend. KEEP the in-flight row
                # so the next retry re-attaches, and re-raise so the client sees
                # the timeout/cancel.
                raise
            except Exception:
                # Terminal op failure (op done but failed / no candidates). The
                # saved state is now useless -- delete it so the next retry
                # submits fresh.
                await self._clear_in_flight_completion(tuner_id, run_id, request_hash)
                raise

            recipe = await self._recipe_for(tuner_id)
            sample.completion = _apply_max_context_window(
                sample.completion, recipe.max_context_window
            )

            # End-to-end generation latency from first submit, so re-attach
            # cycles are counted. For a turn that never re-attaches (the common
            # fast path) `first_submit ~= now` at submit, matching prior timing.
            duration_ms = int((utcnow() - first_submit).total_seconds() * 1000)
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
                duration_ms=duration_ms,
            )
            # The recorded 200 supersedes the in-flight row; drop it.
            await self._clear_in_flight_completion(tuner_id, run_id, request_hash)
            raw_content = None
            finish_reason = None
            if sample.completion.choices:
                choice = sample.completion.choices[0]
                finish_reason = choice.finish_reason
                if choice.message:
                    raw_content = choice.message.content

            if finish_reason == "content_filter":
                content_filter_penalty = recipe.content_filter_penalty
                await self.update_reward(
                    tuner_id, run_id, reward=content_filter_penalty
                )
                raise ContentFilterSampleError(
                    f"Content-filtered sample on run {run_id}; reward set to {content_filter_penalty}",
                    raw_content=raw_content,
                )

            if finish_reason == "length":
                length_penalty = recipe.length_penalty
                await self.update_reward(tuner_id, run_id, reward=length_penalty)
                raise LengthSampleError(
                    f"Length-limited sample on run {run_id}; reward set to {length_penalty}",
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

    async def _find_in_flight_completion(
        self, tuner_id: str, run_id: str, request_hash: str
    ) -> Optional[InFlightChatCompletionModel]:
        """Return the durable resume state for an in-flight op for this turn.

        Consulted only when there is no recorded 200 yet; the returned row's
        `state` re-attaches to the same backend op via
        `trainer.sample(request, restore_state=...)`.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(InFlightChatCompletionModel).where(
                    InFlightChatCompletionModel.tuner_id == tuner_id,
                    InFlightChatCompletionModel.run_id == run_id,
                    InFlightChatCompletionModel.request_hash == request_hash,
                )
            )
            return result.scalar_one_or_none()

    async def _record_in_flight_completion(
        self,
        tuner_id: str,
        run_id: str,
        request_hash: str,
        state: str,
        created_at: datetime,
        policy_generation: int,
    ) -> None:
        """Persist an op's resume state the moment it is submitted.

        Keyed by the turn identity so at most one in-flight op exists per turn;
        the composite PK plus the per-turn sample lock guarantee no duplicate
        rows race in. ``policy_generation`` is the trainer's current generation
        at submit time, stamped so the expiration-quarantine window can locate a
        still-churning run on the policy-generation timeline before it records
        any completion.
        """
        async with self.async_session() as session:
            async with session.begin():
                session.add(
                    InFlightChatCompletionModel(
                        tuner_id=tuner_id,
                        run_id=run_id,
                        request_hash=request_hash,
                        state=state,
                        created_at=created_at,
                        policy_generation=policy_generation,
                    )
                )

    async def _clear_in_flight_completion(
        self, tuner_id: str, run_id: str, request_hash: str
    ) -> None:
        """Delete the in-flight resume state for a turn.

        Called only on (a) recorded success or (b) terminal op failure -- never
        on cancel/timeout, since those mean the backend op is still progressing
        and the next retry must re-attach.
        """
        async with self.async_session() as session:
            async with session.begin():
                await session.execute(
                    delete(InFlightChatCompletionModel).where(
                        InFlightChatCompletionModel.tuner_id == tuner_id,
                        InFlightChatCompletionModel.run_id == run_id,
                        InFlightChatCompletionModel.request_hash == request_hash,
                    )
                )

    async def record_chat_completion(
        self,
        completion_id: str,
        tuner_id: str,
        run_id: str,
        datum_id: str,
        policy_generation: int,
        request: ChatCompletionRequest,
        response: ChatCompletion,
        duration_ms: int,
        tokens: Optional[List[int]] = None,
        logprobs: Optional[List[float]] = None,
        request_hash: Optional[str] = None,
    ) -> None:
        """
        Record a chat completion event in the database.

        `duration_ms` is the wall-clock generation latency in milliseconds,
        required so every recorded completion carries timing.

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
                    duration_ms=duration_ms,
                )
                session.add(db_completion)

                # The run's `expires_at` is a fixed lease set at creation
                # (`now + RUN_LEASE_SECONDS`) and is intentionally left untouched
                # here: the whole run shares one time slot rather than each
                # completion granting a fresh budget.

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
                logger.debug(
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

                    # Accepted limitation (dual-write, not fully atomic):
                    # `train_step` submits the backend LRO and persists
                    # `pending_train_op` in its *own* transaction, before the
                    # `trained_count` bump below commits. If the process crashes
                    # in between, on restart the LRO still completes (advancing
                    # policy_generation) but these runs keep `trained_count = 0`
                    # and get collected into a later batch -- i.e. the batch may
                    # be trained twice. Bounded (and dampened by the off-policy
                    # staleness filter); tolerated for now rather than adding a
                    # cross-backend 2-phase commit.
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

    async def dispense_run(
        self,
        tuner_id: str,
        *,
        max_expire_rate: Optional[float] = None,
        max_succeed_ratio: Optional[float] = None,
    ) -> Optional[DispenseRun]:
        """
        Dispense a run for a tuner.

        When ``max_expire_rate`` is provided, datums that genuinely keep
        expiring are quarantined and excluded from the candidate pool. The rate
        is measured over *all* of the datum's terminal attempts (no recency
        window) and a datum is skipped once it has accumulated at least
        ``0.5 * recipe.group_size`` attempts (half a group's worth) with an
        expiration rate ``>= max_expire_rate``. Only `expired` runs count --
        those that still have a lingering in-flight op (the generation itself
        stalled past the lease) or ran past the total-duration budget; `lost`
        runs (crashed/abandoned worker, or runs abandoned after their ops
        completed) are ignored, so neither poisons a datum (see
        ``expiring_datums``). When ``max_expire_rate`` is ``None`` the feature
        is disabled.

        When ``max_succeed_ratio`` is provided, datums that are solved too
        reliably are quarantined too: a datum is skipped once it has at least
        ``0.5 * recipe.group_size`` terminal attempts and a success ratio
        (runs with reward ``== 1.0`` over all terminal attempts, i.e. the same
        ``expired + rewarded`` denominator the expire rate uses) that is
        ``> max_succeed_ratio``. Such datums are considered too easy and no
        longer produce a useful learning signal (see ``quarantined_datums``).
        When ``max_succeed_ratio`` is ``None`` this feature is disabled. Both
        filters may be combined; a datum caught by either is excluded.
        """
        # Ensure trainer is initialized.
        _trainer = await self._get_trainer(tuner_id)
        recipe = await self._recipe_for(tuner_id)

        # Serialize the read-pick-insert sequence: the scheduler decision in
        # `pick_datum` depends on the current set of in-flight runs, so two
        # concurrent dispenses that both read the pre-insert snapshot would
        # otherwise pick the same datum and over-dispense it past
        # `group_size` (the source of the inflated in_flight counts).
        async with self._dispense_lock:
            async with self.async_session() as session:
                datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)

                quarantine_enabled = (
                    max_expire_rate is not None or max_succeed_ratio is not None
                )
                if quarantine_enabled and datum_pool:
                    # Both filters share the rewarded denominator: a rewarded
                    # run has no in-flight row (deleted on success). Each
                    # `RewardedRun` carries its datum_id + reward, so the success
                    # numerator (reward == 1.0) is derived from this same map --
                    # no separate query. Counted over the datum's entire history
                    # (no recency window), so the pure helper just tallies.
                    rewarded_by_run = await self._rewarded_datums(tuner_id, session)
                    # The expired numerator is only needed for the expire filter.
                    # An `expired` run is an expired, unrewarded run that either
                    # kept a lingering in-flight op (the generation stalled past
                    # the lease) or ran past the total-duration expiration
                    # threshold.
                    expired_datum_by_run: Dict[str, str] = {}
                    if max_expire_rate is not None:
                        expired_datum_by_run = await self._expired_datums(
                            tuner_id, utcnow(), session
                        )
                    excluded = quarantined_datums(
                        datum_pool,
                        rewarded_by_run,
                        expired_datum_by_run,
                        min_samples=0.5 * recipe.group_size,
                        max_expire_rate=max_expire_rate,
                        max_succeed_ratio=max_succeed_ratio,
                    )
                    if excluded:
                        datum_pool = [d for d in datum_pool if d not in excluded]

            datum_id = pick_datum(datum_pool, runs, recipe)
            if datum_id is None:
                return None

            run_record = RunModel(
                tuner_id=tuner_id,
                datum_id=datum_id,
                reward=None,
                trained_count=0,
                expires_at=utcnow() + timedelta(seconds=RUN_LEASE_SECONDS),
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

    async def _rewarded_datums(self, tuner_id: str, session) -> Dict[str, RewardedRun]:
        """Per *rewarded* run, a :class:`RewardedRun` (datum id + reward).

        Returns a ``run_id -> RewardedRun`` map used by the dispenser's
        quarantine logic for the denominator's rewarded side. A rewarded run has
        no lingering in-flight row -- it is deleted on success -- so it is
        located via its completions. The join to ``RunModel`` keeps the map to
        exactly the rewarded runs the quarantine algorithm consults (and carries
        each run's ``datum_id`` + ``reward``, so ``terminal_stats`` can tally
        both the rewarded denominator and the ``reward == 1.0`` success
        numerator without the full run list or a second query), mirroring how
        :meth:`_expired_datums` covers the expiration numerator. Every rewarded
        run for the tuner is counted (no recency window).
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
        `lost`) definition shared by two consumers:

        * the dispenser's quarantine numerator (aggregated per datum via
          ``.values()``), and
        * ``list_runs``/``get_run``'s ``expired`` vs ``lost`` run-status split
          (scoped to a page via ``run_ids``, then read as ``set(...)`` for the
          run-id keys).

        Every expired, unrewarded run for the tuner is counted (no recency
        window). Complements :meth:`_rewarded_datums`, which covers the rewarded
        side of the denominator.

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
