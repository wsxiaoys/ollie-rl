"""Read-only query endpoints: tuner details, progress, run/completion listings."""

import json
import logging
from typing import Dict, List, Optional

from openai.types.chat import ChatCompletion
from sqlalchemy import func, select

from ollie_rl.cookbook import Cookbook
from ollie_rl.db import ChatCompletionModel, DatumRowModel, TunerModel
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner.dispense import (
    RewardedRun,
    TerminalStats,
    pick_datum,
    pick_tier,
    scheduler_scores,
    terminal_stats,
)
from ollie_rl.service.tuner.completion_helpers import context_tokens_from_response
from ollie_rl.service.tuner.run_helpers import (
    build_run_item,
    decode_run_cursor,
    encode_run_cursor,
    last_train_op_duration_seconds,
)
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.errors import (
    ChatCompletionNotFoundError,
    RunNotFoundError,
    TunerNotFoundError,
)
from ollie_rl.types import (
    BatchProgress,
    ChatCompletionDetailResponse,
    ChatCompletionItem,
    ChatCompletionRequest,
    DatumCoverage,
    DatumPool,
    DatumProgress,
    GetTunerResponse,
    ListRunsResponse,
    NextPick,
    RunDetailResponse,
    RunProgress,
    TrainingProgress,
    TunerItem,
)

logger = logging.getLogger(__name__)


class QueryMixin(TunerServiceBase):
    """Dashboard/observability reads over tuners, runs, and completions."""

    async def get_tuner_details(
        self, tuner_id: str, include_progress: bool = False
    ) -> GetTunerResponse:
        """
        Retrieve tuner details, including current policy_generation and stored trainer state.

        When `include_progress` is set, a recipe-aware `TrainingProgress`
        snapshot is computed and attached (extra DB reads).
        """
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
            last_train_op_duration_seconds=last_train_op_duration_seconds(state_data),
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

            # Quarantine/progress inputs. Reuse the dispenser's own helpers so
            # the dashboard's length/rewarded counts match exactly what the
            # dispenser quarantines on. Expired, unrewarded runs are still loaded
            # for progress/status observability, but expiration is no longer a
            # quarantine metric.
            # Kept separate from `generation_by_run_id` so the trainer-view
            # consumable calc (which only cares about recorded completions) is
            # unaffected.
            rewarded_by_run: Dict[str, RewardedRun] = {}
            expired_datum_by_run: Dict[str, str] = {}
            length_datum_by_run: Dict[str, str] = {}
            if runs:
                rewarded_by_run = await self._rewarded_datums(tuner_id, session)
                expired_datum_by_run = await self._expired_datums(
                    tuner_id, now, session
                )
                length_datum_by_run = await self._length_datums(tuner_id, session)

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

        # Terminal stats per datum, computed from the same rewarded/length maps
        # the dispenser feeds `terminal_stats`, so the dashboard's length and
        # rewarded counts match what the quarantine filters act on.
        stats_by_datum = terminal_stats(
            datum_pool,
            rewarded_by_run,
            length_datum_by_run,
        )
        expired_by_datum: Dict[str, int] = {d: 0 for d in datum_pool}
        for datum_id in expired_datum_by_run.values():
            if datum_id in expired_by_datum:
                expired_by_datum[datum_id] += 1

        items: List[DatumProgress] = []
        groups_ready = 0
        groups_in_progress = 0
        datums_in_progress = 0
        for datum_id in consumable_by_datum:
            count = consumable_by_datum[datum_id]
            pending = in_flight_by_datum.get(datum_id, 0)
            trained_here = trained_by_datum.get(datum_id, 0)
            # `terminal_stats` gives per-datum (rewarded, length, succeeded).
            # `length`, `rewarded`, and `succeeded` let a client derive the
            # active length/success quarantine ratios. Expired is counted
            # separately for run-status observability only.
            stats = stats_by_datum.get(datum_id, TerminalStats())
            expired_here = expired_by_datum.get(datum_id, 0)
            # Surface any datum that has activity worth showing: a group
            # forming (rewarded runs counting toward the batch, or runs still
            # awaiting a reward) or one that has already contributed a trained
            # group. Without the trained check a datum whose group was fully
            # trained (consumable/in-flight back to 0) would silently vanish
            # from the pool even though it carries training history. Expired and
            # length-limited runs also count as activity worth surfacing.
            if (
                count <= 0
                and pending <= 0
                and trained_here <= 0
                and expired_here <= 0
                and stats.length <= 0
            ):
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
                    expired=expired_here,
                    length=stats.length,
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

        cursor_key = decode_run_cursor(cursor) if cursor else None

        async with self.async_session() as session:
            tuner_result = await session.execute(
                select(TunerModel).where(TunerModel.id == tuner_id)
            )
            tuner = tuner_result.scalar_one_or_none()
            if tuner is None:
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
            context_windows: Dict[str, int] = {}
            length_datum_by_run: Dict[str, str] = {}
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

                context_result = await session.execute(
                    select(
                        ChatCompletionModel.run_id,
                        ChatCompletionModel.response,
                    ).where(
                        ChatCompletionModel.tuner_id == tuner_id,
                        ChatCompletionModel.run_id.in_(run_ids),
                    )
                )
                for run_id, response in context_result.all():
                    if run_id is None:
                        continue
                    context_tokens = context_tokens_from_response(response)
                    if context_tokens is None:
                        continue
                    context_windows[run_id] = max(
                        context_windows.get(run_id, 0), context_tokens
                    )

                length_datum_by_run = await self._length_datums(
                    tuner_id, session, run_ids=run_ids
                )

            # Runs on this page that count as `expired` (unrewarded, past lease,
            # with either a lingering in-flight op or total duration past the
            # expiration threshold), so a past-lease unrewarded run can be split
            # into `expired` vs `lost`. Scoped to the page's run ids. Expiration
            # is observability-only and no longer drives dispenser quarantine.
            now = utcnow()
            expired_run_ids = set(
                await self._expired_datums(
                    tuner_id, now, session, run_ids=[r.id for r in runs]
                )
            )

        items = [
            build_run_item(
                r,
                counts.get(r.id, 0),
                now,
                generations.get(r.id),
                durations.get(r.id),
                context_windows.get(r.id),
                r.id in expired_run_ids,
                r.id in length_datum_by_run,
            )
            for r in runs
        ]
        next_cursor = (
            encode_run_cursor(runs[-1].created_at, runs[-1].id)
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
            length_datum_by_run = await self._length_datums(
                tuner_id, session, run_ids=[run_id]
            )

        policy_generation = (
            max(c.policy_generation for c in completions) if completions else None
        )
        durations = [c.duration_ms for c in completions if c.duration_ms is not None]
        duration_ms_total = sum(durations) if durations else None
        context_windows = [
            context_tokens
            for c in completions
            if (context_tokens := context_tokens_from_response(c.response)) is not None
        ]
        context_window_tokens_max = max(context_windows) if context_windows else None
        run_item = build_run_item(
            run,
            len(completions),
            now,
            policy_generation,
            duration_ms_total,
            context_window_tokens_max,
            run_id in expired_run_ids,
            run_id in length_datum_by_run,
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
