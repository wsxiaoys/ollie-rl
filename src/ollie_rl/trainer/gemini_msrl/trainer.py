from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from gemini_msrl import GeminiMsrlClient
from gemini_msrl.types import (
    GenerateContentTuningScopeRequest,
    ReinforcementTuningTrainingData,
    ReinforcementTuningTrainingDataBatch,
    TrainStepRequest,
)

from ollie_rl.trainer.types import (
    LIVE_POLICY_CHECKPOINT,
    Checkpoint,
    Example,
    Sampler,
    StateStore,
    Trainer,
)
from ollie_rl.types import ChatCompletionRequest

from .conversion import build_content_generation_parameters
from .ops import GeminiMsrlSamplingOp, GeminiMsrlTrainOp
from .sampler import GeminiMsrlSampler
from .state import GeminiMsrlTrainerConfig, GeminiMsrlTrainerState, PendingTrainOp

logger = logging.getLogger(__name__)


class GeminiMsrlTrainer(Trainer):
    """
    Trainer wrapping the Gemini MSRL tuning client.

    The Trainer's persistable state lives directly on `self.state`
    (a `GeminiMsrlTrainerState`). Mutate that object in place and then
    call `_persist_state()` to push it to the backing store.
    """

    config: GeminiMsrlTrainerConfig
    client: GeminiMsrlClient
    state: GeminiMsrlTrainerState
    state_store: StateStore

    def __init__(
        self,
        config: GeminiMsrlTrainerConfig,
        client: GeminiMsrlClient,
        state: GeminiMsrlTrainerState,
        state_store: StateStore,
    ):
        self.config = config
        self.client = client
        self.state = state
        self.state_store = state_store
        # Lazily-populated one-shot guard that resolves once the tuning job has
        # entered the RUNNING state. TPU allocation/warm-up can take a long
        # time, so we don't block create()/restore() on it; instead the first
        # operation that needs a running job awaits `_ensure_running()`.
        self._running_ready: Optional[asyncio.Task[None]] = None

    async def _ensure_running(self) -> None:
        """Wait (at most once) for the tuning job to enter RUNNING state.

        The wait is deferred out of create()/restore() so those return quickly.
        Concurrent callers share the same underlying wait; on failure the guard
        is reset so a later call can retry.
        """
        if self._running_ready is None:
            self._running_ready = asyncio.create_task(self._wait_for_running())
        try:
            await self._running_ready
        except Exception:
            self._running_ready = None
            raise

    async def _wait_for_running(self) -> None:
        logger.info(
            f"Waiting for tuning job '{self.tuning_job_name}' to enter RUNNING state..."
        )
        await self.client.wait_for_tuning_job_running(
            self.tuning_job_name,
            timeout_seconds=5.0,
            poll_interval=1.0,
        )
        logger.info("Gemini MSRL Tuning Job is successfully running.")

    @property
    def policy_generation(self) -> int:
        return self.state.train_step

    async def create_sampler(self, checkpoint: Checkpoint) -> Sampler:
        if checkpoint.ref == LIVE_POLICY_CHECKPOINT:
            return self
        return GeminiMsrlSampler(
            client=self.client,
            endpoint_name=checkpoint.ref,
            policy_generation=checkpoint.policy_generation,
        )

    @property
    def tuning_job_name(self) -> str:
        return self.state.tuning_job_name

    async def pending_train_op(self) -> Optional[GeminiMsrlTrainOp]:
        """The in-flight train-step LRO, if one is running, else None.

        `train_step` records the in-flight op name in `pending_train_op` the
        moment it submits the LRO, and `GeminiMsrlTrainOp.wait()` clears it
        once the op completes. A non-None `pending_train_op` therefore reliably
        tracks an in-flight op without an extra backend round-trip; we just
        wrap the op name in a fresh (authoritative) `GeminiMsrlTrainOp` handle.
        """
        if self.state.pending_train_op is not None:
            return GeminiMsrlTrainOp(self, self.state.pending_train_op.name)
        return None

    async def _persist_state(self) -> None:
        await self.state_store.save(self.state.model_dump_json())

    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ) -> GeminiMsrlSamplingOp:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        if restore_state is not None:
            # Re-attach to the already-submitted op instead of submitting a new
            # one. This is the exact inline reconstruction that `train_step` /
            # `restore` already do for the train op. `model_name` only shapes
            # the returned ChatCompletion envelope and comes from the request.
            return GeminiMsrlSamplingOp(self.client, restore_state, request.model)

        # Ensure the tuning job is running before submitting work to it.
        await self._ensure_running()

        tuning_job_id = self.tuning_job_name.split("/")[-1]
        scope_req = GenerateContentTuningScopeRequest(
            content_generation_parameters=build_content_generation_parameters(request)
        )

        # 2. Trigger Generation LRO
        op = await self.client.generate_content_tuning_scope(tuning_job_id, scope_req)

        return GeminiMsrlSamplingOp(self.client, op.name, request.model)

    async def train_step(
        self,
        examples: List[Example],
        *,
        sampler_promotion_every: int = 1,
    ) -> GeminiMsrlTrainOp:
        assert self.client and self.tuning_job_name, "Tuning job not initialized"

        # Ensure the tuning job is running before submitting work to it.
        await self._ensure_running()

        if self.state.pending_train_op:
            last_op = GeminiMsrlTrainOp(
                self,
                self.state.pending_train_op.name,
            )
            if not await last_op.peek():
                raise RuntimeError(
                    "Last training step "
                    f"{self.state.pending_train_op.name} is still active"
                )

        # 1. Translate examples to TrainStepRequest
        rt_examples = []
        for item in examples:
            rt_examples.append(
                ReinforcementTuningTrainingData(
                    candidate_id=item.chat_completion_id,
                    advantage=item.advantage,
                )
            )

        tuning_job_id = self.tuning_job_name.split("/")[-1]

        # Promote a sampler only every `sampler_promotion_every` steps: skip the
        # weight sync on the steps in between. The step this op completes is the
        # next one after the last completed step, so promote when that lands on
        # the cadence boundary.
        next_step = self.state.train_step + 1
        skip_weight_sync = next_step % sampler_promotion_every != 0

        train_req = TrainStepRequest(
            reinforcement_tuning_training_data_batch=ReinforcementTuningTrainingDataBatch(
                examples=rt_examples
            ),
            skip_weight_sync=skip_weight_sync,
        )

        # 2. Trigger TrainStep LRO
        op = await self.client.train_step(tuning_job_id, train_req)

        self.state.pending_train_op = PendingTrainOp(
            name=op.name,
            skip_weight_sync=skip_weight_sync,
        )
        await self._persist_state()

        # The returned op is the single authoritative waiter: the service
        # awaits `wait()`, which drives the op to completion, records
        # `last_train_op`, clears `pending_train_op` and persists. No
        # background poll is spawned here (nor re-spawned on restart) --
        # reconcile flows through `pending_train_op()` instead.
        return GeminiMsrlTrainOp(self, op.name)
