from __future__ import annotations

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from gemini_msrl import GeminiMsrlClient
from gemini_msrl.types import (
    GenerateContentTuningScopeResponse,
    GenericMetadata,
    TrainStepResponse,
)

from ollie_rl.trainer.types import (
    LIVE_POLICY_CHECKPOINT,
    Checkpoint,
    Sample,
    SampleOp,
    TrainOp,
)

from .conversion import sample_from_candidates
from .state import CompletedTrainOp

if TYPE_CHECKING:
    from .trainer import GeminiMsrlTrainer

logger = logging.getLogger(__name__)


class GeminiMsrlOp:
    def __init__(
        self,
        client: GeminiMsrlClient,
        op_name: str,
    ):
        self.client = client
        self.op_name = op_name

    async def peek(self) -> bool:
        """Return True iff the op has reached a terminal state."""
        operation = await self.client.get_operation(self.op_name)
        return bool(operation.done)

    def save_state(self) -> Optional[str]:
        """Serialize this op's resume state: the LRO operation resource name.

        Persisting this the moment the op is submitted lets a later retry
        re-attach to the *same* in-flight Gemini operation (via
        ``sample(request, restore_state=...)``) instead of spawning a fresh op.
        """
        return self.op_name


class GeminiMsrlSamplingOp(GeminiMsrlOp, SampleOp):
    def __init__(
        self,
        client: GeminiMsrlClient,
        op_name: str,
        model_name: str,
    ):
        super().__init__(client, op_name)
        self.model_name = model_name

    async def wait(self) -> Sample:
        completed_op = await self.client.wait_for_operation(
            self.op_name,
        )

        response = completed_op.get_response_as(GenerateContentTuningScopeResponse)
        if not response or not response.candidates:
            raise RuntimeError(
                "Failed to retrieve generated candidates from tuning scope response"
            )

        return sample_from_candidates(
            candidates=response.candidates,
            usage_metadata=response.usage_metadata,
            model_name=self.model_name,
            policy_generation=int(response.train_step_id),
        )


class GeminiMsrlEndpointSampleOp(SampleOp):
    def __init__(self, task: asyncio.Task[Sample]):
        self.task = task

    async def wait(self) -> Sample:
        return await self.task

    async def peek(self) -> bool:
        return self.task.done()


class GeminiMsrlTrainOp(GeminiMsrlOp, TrainOp):
    """The single, restart-surviving completion path for a train-step LRO.

    Holds a reference to its :class:`GeminiMsrlTrainer` so ``wait()`` can
    mutate and persist trainer state as it drives the op to completion. This
    is *the* authoritative waiter: the service awaits it (either the fresh op
    returned by ``train_step`` or the one handed back by ``pending_train_op``
    on reconcile) and it records ``last_train_op``, clears ``pending_train_op``
    and persists.
    """

    def __init__(self, trainer: GeminiMsrlTrainer, op_name: str):
        super().__init__(trainer.client, op_name)
        self.trainer = trainer

    async def wait(self) -> Optional[Checkpoint]:
        """Wait for the train-step LRO to terminate, record the completed
        ``TrainStepResponse`` in ``last_train_op`` and clear ``pending_train_op``.

        Returns the :class:`Checkpoint` the step produced (using the deployed
        ``TunedModelCheckpoint.endpoint`` when present, or the live-policy
        sentinel for promotion steps that omit a checkpoint), or ``None`` when
        the completed op carried no ``completed_train_step_id``.

        A train step can legitimately run much longer than a single
        ``wait_for_operation`` budget, so we keep retrying on timeouts /
        transient errors rather than giving up. Bailing out early is what
        previously left ``pending_train_op`` permanently stuck even though the
        underlying Vertex op had finished. Terminal errors are logged and
        swallowed (never raised out of ``wait()``) so the service reconcile
        loop is not killed.
        """
        # A short backoff between retries after a transient (non-timeout)
        # failure so we don't spin tightly on a persistent error.
        _RETRY_BACKOFF_SECONDS = 5.0

        trainer = self.trainer
        op_name = self.op_name

        completed_op = None
        while True:
            try:
                completed_op = await self.client.wait_for_operation(op_name)
                break
            except TimeoutError:
                # Op is still running past the poll budget; keep waiting so the
                # in-flight pointer eventually gets cleared once it completes.
                logger.warning(
                    "Train op %s still running; continuing to poll",
                    op_name,
                )
                continue
            except Exception as e:  # noqa: BLE001
                # Transient failure (e.g. token refresh / network blip). Back
                # off and retry rather than abandoning the poll, which would
                # leave `pending_train_op` stuck.
                logger.warning(
                    "Error polling train op %s: %s; retrying in %.0fs",
                    op_name,
                    e,
                    _RETRY_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                continue

        checkpoint: Optional[Checkpoint] = None
        try:
            response = completed_op.get_response_as(TrainStepResponse)
            if asyncio.iscoroutine(response):
                response = await response
            if (
                isinstance(response, TrainStepResponse)
                and response.completed_train_step_id
            ):
                # Record the completed response monotonically so
                # `policy_generation` never regresses (guards against an older
                # op's poll landing after a newer one).
                completed = int(response.completed_train_step_id)
                if completed >= trainer.state.train_step:
                    # Persist the response together with the op name + its
                    # timing metadata so the exact operation is traceable and
                    # the train-step execution time (updateTime - createTime)
                    # is recoverable without re-querying Vertex.
                    generic_metadata = (completed_op.metadata or {}).get(
                        "genericMetadata"
                    )
                    trainer.state.last_train_op = CompletedTrainOp(
                        response=response,
                        name=completed_op.name,
                        metadata=(
                            GenericMetadata.model_validate(generic_metadata)
                            if generic_metadata
                            else None
                        ),
                    )
                # The completed step yields a checkpoint. When Vertex returns a
                # `TunedModelCheckpoint`, persist its endpoint resource name as
                # the checkpoint ref. If this was a promotion step but Gemini
                # somehow omits the checkpoint entirely, fall back to the
                # live-policy sentinel so eval still attributes to the promoted
                # generation instead of losing the checkpoint.
                tuned_checkpoint = response.tuned_model_checkpoint
                if tuned_checkpoint is not None:
                    # Vertex returned an addressable frozen checkpoint: persist
                    # the deployed endpoint as the ref and attribute the sample
                    # to the checkpoint's own required `step`.
                    checkpoint = Checkpoint(
                        ref=tuned_checkpoint.endpoint,
                        policy_generation=tuned_checkpoint.step,
                    )
                elif not (
                    trainer.state.pending_train_op is not None
                    and trainer.state.pending_train_op.skip_weight_sync
                ):
                    # No checkpoint returned, but this was a promotion step
                    # (`skip_weight_sync=False`): the live policy advanced, so
                    # attribute the sample to the completed train step id via
                    # the live-policy sentinel. (An unknown/cleared pending op,
                    # e.g. a legacy reconcile, is treated as a promotion too.)
                    checkpoint = Checkpoint(
                        ref=LIVE_POLICY_CHECKPOINT,
                        policy_generation=completed,
                    )
                # Otherwise `skip_weight_sync=True` produced no checkpoint at
                # all, so leave `checkpoint` as None: no new generation.

            # Clear the in-flight pointer if it still refers to this op; a newer
            # train_step may have already replaced it.
            if (
                trainer.state.pending_train_op is not None
                and trainer.state.pending_train_op.name == op_name
            ):
                trainer.state.pending_train_op = None

            await trainer._persist_state()
        except Exception as e:
            logger.error(f"Error polling train step or updating state: {e}")

        return checkpoint
