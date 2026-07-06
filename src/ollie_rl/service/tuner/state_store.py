"""Database-backed :class:`StateStore` used to persist trainer state."""

import logging
from typing import Optional

from sqlalchemy import select, update

from ollie_rl.db import TunerModel
from ollie_rl.db.connection import get_sessionmaker
from ollie_rl.trainer import StateStore

logger = logging.getLogger(__name__)


class DbStateStore(StateStore):
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
