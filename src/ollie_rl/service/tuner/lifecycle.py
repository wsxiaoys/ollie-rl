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
        state_store = DbStateStore(tuner_id)
        trainer_instance = await factory.create(
            name, state_store, trainer_params=trainer_params
        )
        self.active_trainers[tuner_id] = trainer_instance

        logger.info(f"Successfully created tuner {tuner_id}")
        return tuner_id
