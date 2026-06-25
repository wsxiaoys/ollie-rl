import logging
from typing import Dict, Optional, List
from pydantic import BaseModel

from .redis_client import get_redis_client
from ollie_rl.cookbook import Tuner, Cookbook

logger = logging.getLogger(__name__)


class TunerRecord(BaseModel):
    """
    Pydantic model representing a single persisted tuner's metadata and state.
    """

    kind: str
    state: str


class TunerStorage:
    """
    Handles both active in-memory tuners and their persistence to a Redis hash.
    Uses Redis in production, falling back to an in-memory Fakeredis client for development.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.client = get_redis_client(redis_url)
        self.active_tuners: Dict[str, Tuner] = {}

    def get(self, tuner_id: str) -> Optional[Tuner]:
        """
        Retrieve an active tuner instance by model ID.
        """
        return self.active_tuners.get(tuner_id)

    def list_keys(self) -> List[str]:
        """
        List all active model IDs in memory.
        """
        return list(self.active_tuners.keys())

    async def load_state(self) -> Dict[str, TunerRecord]:
        """
        Load the raw records from Redis.
        """
        try:
            records_raw = await self.client.hgetall("tuner:records")
            records: Dict[str, TunerRecord] = {}
            for tuner_id, val in records_raw.items():
                if isinstance(val, str):
                    m_id = (
                        tuner_id.decode()
                        if isinstance(tuner_id, bytes)
                        else str(tuner_id)
                    )
                    records[m_id] = TunerRecord.model_validate_json(val)
            return records
        except Exception as e:
            logger.exception("Failed to load persisted tuners from Redis")
            return {}

    async def restore_tuners(self) -> None:
        """
        Load the persisted state and restore active Tuner instances into memory using Cookbook.
        """
        records = await self.load_state()
        for tuner_id, record in records.items():
            try:
                logger.info(
                    f"Restoring tuner for model: {tuner_id} (kind: {record.kind})"
                )
                tuner = await Cookbook.restore(record.kind, record.state)
                self.active_tuners[tuner_id] = tuner
            except Exception as e:
                logger.exception(f"Failed to restore tuner for model: {tuner_id}")
        logger.info(f"Successfully restored {len(self.active_tuners)} tuners.")

    async def register_tuner(self, tuner_id: str, tuner: Tuner) -> None:
        """
        Register a new tuner instance, keeping it in memory and persisting it to storage.
        """
        self.active_tuners[tuner_id] = tuner
        await self.save_tuner(tuner_id, tuner)

    async def save_tuner(self, tuner_id: str, tuner: Tuner) -> None:
        """
        Save or update a single tuner in the persistent storage.
        """
        try:
            state_str = await tuner.save_state()
            record = TunerRecord(kind=tuner.kind, state=state_str)
            await self.client.hset("tuner:records", tuner_id, record.model_dump_json())
            logger.info(f"Successfully persisted tuner {tuner_id} to Redis")
        except Exception as e:
            logger.exception(f"Failed to save tuner for model: {tuner_id}")

    async def save_all_tuners(self) -> None:
        """
        Save/overwrite all active tuners in the persistent storage.
        """
        records: Dict[str, str] = {}
        for tuner_id, tuner in self.active_tuners.items():
            try:
                state_str = await tuner.save_state()
                record = TunerRecord(kind=tuner.kind, state=state_str)
                records[tuner_id] = record.model_dump_json()
            except Exception as e:
                logger.exception(f"Failed to save state for model: {tuner_id}")

        try:
            async with self.client.pipeline() as pipe:
                await pipe.delete("tuner:records")
                for k, v in records.items():
                    await pipe.hset("tuner:records", k, v)
                await pipe.execute()
            logger.info(f"Successfully persisted {len(records)} tuners to Redis")
        except Exception as e:
            logger.exception("Failed to write persisted tuners to Redis")

    async def delete_tuner(self, tuner_id: str) -> None:
        """
        Remove a tuner from both memory and persistent storage.
        """
        if tuner_id in self.active_tuners:
            del self.active_tuners[tuner_id]
        try:
            await self.client.hdel("tuner:records", tuner_id)
            logger.info(f"Successfully deleted tuner {tuner_id} from Redis")
        except Exception as e:
            logger.exception(
                f"Failed to delete tuner for model: {tuner_id} from storage"
            )

    async def close(self) -> None:
        """
        Close the Redis connection pool.
        """
        await self.client.aclose()
