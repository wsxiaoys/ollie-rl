from __future__ import annotations

from typing import Any, Optional

from gemini_msrl.types import GenericMetadata, TrainStepResponse
from pydantic import BaseModel, model_validator


class GeminiMsrlTrainerConfig(BaseModel):
    base_model: str = "gemini-3.5-flash"
    adapter_size: str = "ADAPTER_SIZE_SIXTEEN"
    checkpoint_interval: int = 10
    tuning_job_name: Optional[str] = None


class CompletedTrainOp(BaseModel):
    """The most recent *completed* train-step LRO and its bookkeeping.

    Bundles the train-step response (source of truth for `policy_generation`)
    together with the operation's resource name and timing metadata so the
    exact Vertex operation is traceable and its execution time
    (`metadata.update_time - metadata.create_time`) is recoverable.
    """

    # The completed train-step response payload.
    response: TrainStepResponse
    # Operation resource name of the completed train-step LRO.
    name: Optional[str] = None
    # `genericMetadata` (createTime / updateTime) of the completed LRO.
    # `update_time - create_time` gives the train op's execution time.
    metadata: Optional[GenericMetadata] = None


class PendingTrainOp(BaseModel):
    """An in-flight train-step LRO and the metadata needed to reconcile it.

    Set the moment a train step is submitted and cleared once
    `GeminiMsrlTrainOp.wait()` observes its completion. Used to detect
    in-flight training and, via `pending_train_op()`, to reconcile an op that
    was submitted before a restart.
    """

    # Operation resource name of the in-flight train-step LRO.
    name: str
    # `skip_weight_sync` flag the op was submitted with. Needed so `wait()` can
    # tell a promotion step that produced no `TunedModelCheckpoint` (attribute
    # to the completed step id) apart from a non-promotion step (no new
    # checkpoint at all).
    skip_weight_sync: bool = False


class GeminiMsrlTrainerState(BaseModel):
    tuning_job_name: str
    # The most recent *completed* train-step op. This is the source of truth
    # for `policy_generation` and is NEVER overwritten by an in-flight op name,
    # so the generation can't regress to 0 while a new train step is running
    # (that is tracked separately by `pending_train_op`).
    last_train_op: Optional[CompletedTrainOp] = None
    # The in-flight train-step LRO, if any.
    pending_train_op: Optional[PendingTrainOp] = None
    config: GeminiMsrlTrainerConfig

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_last_train_op(cls, data: Any) -> Any:
        """Backwards-compat for `last_train_op` shape changes:

        1. Oldest format stored the in-flight op name as a `str`; migrate it
           into `pending_train_op`.
        2. Previous format stored the bare `TrainStepResponse` dict; wrap it
           into the new `CompletedTrainOp` container under `response`.
        3. `pending_train_op` was previously a bare op-name `str`; wrap it
           into the `PendingTrainOp` container (defaulting `skip_weight_sync`
           to False).
        """
        if not isinstance(data, dict):
            return data
        legacy = data.get("last_train_op")
        if isinstance(legacy, str):
            data = dict(data)
            data.setdefault("pending_train_op", legacy)
            data["last_train_op"] = None
        elif isinstance(legacy, dict) and "response" not in legacy:
            data = dict(data)
            data["last_train_op"] = {"response": legacy}
        legacy_pending = data.get("pending_train_op")
        if isinstance(legacy_pending, str):
            data = dict(data)
            data["pending_train_op"] = {"name": legacy_pending}
        return data

    @property
    def train_step(self) -> int:
        if self.last_train_op and self.last_train_op.response.completed_train_step_id:
            return int(self.last_train_op.response.completed_train_step_id)
        return 0
