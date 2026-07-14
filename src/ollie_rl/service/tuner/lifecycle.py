"""Tuner creation/registration."""

import logging
from typing import List, Optional

from ollie_rl.cookbook import Cookbook
from ollie_rl.db import DatumRowModel, TunerModel
from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.state_store import DbStateStore
from ollie_rl.trainer import factory as trainer_factory

logger = logging.getLogger(__name__)


class LifecycleMixin(TunerServiceBase):
    """Create and initialize new tuners."""

    async def create_tuner(
        self,
        recipe: str,
        name: str,
        train_datum_ids: List[str],
        trainer: str,
        eval_datum_ids: Optional[List[str]] = None,
        trainer_params: Optional[dict] = None,
    ) -> str:
        """
        Create and initialize a tuner using the Cookbook and register it.

        ``train_datum_ids`` become the training pool (dispensed, rewarded,
        consumed by train steps); ``eval_datum_ids`` are held out for
        per-checkpoint scoring only. A datum id must be train or eval, never
        both. Raises ``ValueError`` when ``train_datum_ids`` is empty or the
        two sets overlap (the single source of truth for this validation; the
        API layer maps it to a 400).
        """
        recipe_config = Cookbook.get(recipe)
        eval_datum_ids = eval_datum_ids or []
        if not train_datum_ids:
            raise ValueError("train_datum_ids must be non-empty")
        overlap = set(train_datum_ids) & set(eval_datum_ids)
        if overlap:
            raise ValueError(f"train/eval datum ids overlap: {sorted(overlap)}")
        factory = trainer_factory.get(trainer)  # validate now, fail fast

        if trainer == "gemini_msrl":
            # Gemini checkpoint materialization is configured at tuning-job
            # creation time. Align it with the recipe's sampler-promotion
            # cadence so promotion steps are also checkpoint-producing steps.
            trainer_params = dict(trainer_params or {})
            trainer_params.setdefault(
                "checkpoint_interval", recipe_config.sampler_promotion_every
            )

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
                for datum_id in train_datum_ids:
                    session.add(
                        DatumRowModel(
                            tuner_id=tuner_record.id,
                            datum_id=datum_id,
                            kind="train",
                        )
                    )
                for datum_id in eval_datum_ids:
                    session.add(
                        DatumRowModel(
                            tuner_id=tuner_record.id,
                            datum_id=datum_id,
                            kind="eval",
                        )
                    )

        tuner_id = tuner_record.id
        state_store = DbStateStore(tuner_id)
        trainer_instance = await factory.create(
            name, state_store, trainer_params=trainer_params
        )
        self.active_trainers[tuner_id] = trainer_instance

        logger.info(f"Successfully created tuner {tuner_id}")
        return tuner_id
