from __future__ import annotations

import logging
from typing import Any, Optional

from gemini_msrl.types import (
    CreateTuningJobRequest,
    MultiStepReinforcementTuningHyperParameters,
    MultiStepReinforcementTuningSpec,
)

from ollie_rl.trainer import factory
from ollie_rl.trainer.types import StateStore, TrainerFactory

from .state import GeminiMsrlTrainerConfig, GeminiMsrlTrainerState
from .trainer import GeminiMsrlTrainer

logger = logging.getLogger(__name__)


def _create_client():
    """Instantiate via the package export to keep legacy patch paths working.

    Tests and downstream code historically patched
    ``ollie_rl.trainer.gemini_msrl.GeminiMsrlClient`` when this implementation
    lived in a single module. Looking the symbol up on the package at runtime
    preserves that extension point after splitting the code into submodules.
    """
    import ollie_rl.trainer.gemini_msrl as gemini_msrl_pkg

    return gemini_msrl_pkg.GeminiMsrlClient()


class GeminiMsrlTrainerFactory(TrainerFactory):
    """
    Trainer factory for Gemini MSRL trainers.
    """

    async def create(
        self,
        name: str,
        state_store: StateStore,
        trainer_params: Optional[dict] = None,
    ) -> GeminiMsrlTrainer:
        config_kwargs: dict[str, Any] = {}

        if trainer_params:
            config_kwargs.update(trainer_params)

        config = GeminiMsrlTrainerConfig(**config_kwargs)
        # If GEMINI_MSRL_ENV_FILE is set, the client will re-read the auth
        # token from that file (by mtime) on every outgoing request. This lets
        # us refresh tokens externally (e.g. `gcloud auth application-default
        # print-access-token > .env`) without restarting the server.
        client = _create_client()

        # Bootstrap path: create a fresh tuning job and persist its name.
        if config.tuning_job_name:
            tuning_job_name = config.tuning_job_name
            if "/" not in tuning_job_name:
                tuning_job_name = f"projects/{client.project_id}/locations/{client.location}/tuningJobs/{tuning_job_name}"
            logger.info(f"Using pre-created Gemini MSRL tuning job: {tuning_job_name}")
        else:
            req = CreateTuningJobRequest(
                tuned_model_display_name=name,
                base_model=config.base_model,
                multi_step_reinforcement_tuning_spec=MultiStepReinforcementTuningSpec(
                    hyper_parameters=MultiStepReinforcementTuningHyperParameters(
                        adapter_size=config.adapter_size,
                        checkpoint_interval=config.checkpoint_interval,
                    )
                ),
            )

            logger.info(
                f"Creating Gemini MSRL tuning job for model display name: {name}"
            )
            job = await client.create_tuning_job(req)
            tuning_job_name = job.name

        instance = GeminiMsrlTrainer(
            config=config,
            client=client,
            state=GeminiMsrlTrainerState(
                tuning_job_name=tuning_job_name,
                config=config,
            ),
            state_store=state_store,
        )

        # Persist initial state as soon as we have a tuning_job_name, so
        # we can recover even if TPU warm-up is interrupted below.
        await instance._persist_state()

        # TPU allocation and initialization (JOB_STATE_RUNNING) can take a long
        # time. We don't block creation on it; the wait is deferred to the
        # first operation that needs a running job via `_ensure_running()`.
        return instance

    async def restore(
        self,
        name: str,
        state_store: StateStore,
    ) -> GeminiMsrlTrainer:
        raw_state = await state_store.load()
        if raw_state is None:
            raise ValueError(
                f"Cannot restore Gemini MSRL trainer for {name}: no persisted state found."
            )

        state = GeminiMsrlTrainerState.model_validate_json(raw_state)
        config = state.config

        # If GEMINI_MSRL_ENV_FILE is set, the client will re-read the auth
        # token from that file (by mtime) on every outgoing request. This lets
        # us refresh tokens externally (e.g. `gcloud auth application-default
        # print-access-token > .env`) without restarting the server.
        client = _create_client()

        logger.info(
            f"Restoring Gemini MSRL tuning job from state: {state.tuning_job_name}"
        )

        instance = GeminiMsrlTrainer(
            config=config,
            client=client,
            state=state,
            state_store=state_store,
        )

        # A non-None `pending_train_op` means we never observed the op's
        # completion before the restart. We no longer re-spawn a poll here:
        # the service reconciles via `pending_train_op()` -> `wait()`, which is
        # the single restart-surviving completion path.

        # TPU allocation and initialization (JOB_STATE_RUNNING) can take a long
        # time. We don't block restore on it; the wait is deferred to the first
        # operation that needs a running job via `_ensure_running()`.
        return instance


# Register the factory
factory.register("gemini_msrl", GeminiMsrlTrainerFactory())
