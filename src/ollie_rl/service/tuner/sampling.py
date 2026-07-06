"""Sampling, in-flight op bookkeeping, completion recording, and rewards."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from openai.types.chat import ChatCompletion
from sqlalchemy import delete, func, select, update

from ollie_rl.db import ChatCompletionModel, InFlightChatCompletionModel
from ollie_rl.db.models import RunModel
from ollie_rl.db.types import utcnow
from ollie_rl.service.tuner.completion_helpers import (
    apply_max_context_window,
    hash_request,
)
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.constants import RUN_LEASE_SECONDS
from ollie_rl.service.tuner.errors import (
    ContentFilterSampleError,
    EmptyRunError,
    LengthSampleError,
    RewardAlreadySetError,
    RunExpiredError,
    RunNotFoundError,
    TunerNotFoundError,
)
from ollie_rl.types import ChatCompletionRequest

logger = logging.getLogger(__name__)


class SamplingMixin(TunerServiceBase):
    """Generation, idempotent replay, completion persistence, and rewards."""

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
            return apply_max_context_window(
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
        request_hash = hash_request(request)
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
            sample.completion = apply_max_context_window(
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

                # Extend the run's lease: each recorded completion pushes the
                # deadline out to `RUN_LEASE_SECONDS` from the completion time,
                # so an actively-progressing multi-turn run isn't expired
                # mid-flight. A genuinely stalled/abandoned run records no new
                # completion and so still lapses at its last deadline.
                await session.execute(
                    update(RunModel)
                    .where(
                        RunModel.tuner_id == tuner_id,
                        RunModel.id == run_id,
                    )
                    .values(
                        expires_at=utcnow() + timedelta(seconds=RUN_LEASE_SECONDS)
                    )
                )

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
