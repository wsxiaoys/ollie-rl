import logging
from typing import List, Optional

from .redis_client import get_redis_client

logger = logging.getLogger(__name__)


class CompletionStorage:
    """
    Handles persistence of completion IDs within a chat ID or a tuner ID.
    Uses Redis in production, falling back to an in-memory Fakeredis client for development.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.client = get_redis_client(redis_url)

    async def record_completion(
        self,
        completion_id: str,
        tuner_id: str,
        chat_id: Optional[str] = None,
        ttl_seconds: int = 604800,
    ) -> None:
        """
        Record a completion ID under a tuner ID, and optionally a chat ID, and set/renew the TTL.
        """
        try:
            # We use a pipeline to perform sadd and expire atomically/efficiently
            async with self.client.pipeline() as pipe:
                tuner_key = f"tuner:{tuner_id}:completions"
                await pipe.sadd(tuner_key, completion_id)
                await pipe.expire(tuner_key, ttl_seconds)
                if chat_id:
                    chat_key = f"chat:{chat_id}:completions"
                    await pipe.sadd(chat_key, completion_id)
                    await pipe.expire(chat_key, ttl_seconds)
                await pipe.execute()
            logger.debug(
                f"Recorded completion {completion_id} (tuner: {tuner_id}, chat: {chat_id}) with TTL {ttl_seconds}s"
            )
        except Exception as e:
            logger.exception(
                f"Failed to record completion {completion_id} (tuner: {tuner_id}, chat: {chat_id}) in Redis"
            )

    async def get_chat_completions(self, chat_id: str) -> List[str]:
        """
        Retrieve all recorded completion IDs for a given chat ID.
        """
        key = f"chat:{chat_id}:completions"
        try:
            members = await self.client.smembers(key)
            return sorted([m for m in members if isinstance(m, str)])
        except Exception as e:
            logger.exception(
                f"Failed to retrieve completions for chat {chat_id} from Redis"
            )
            return []

    async def get_tuner_completions(self, tuner_id: str) -> List[str]:
        """
        Retrieve all recorded completion IDs for a given tuner ID.
        """
        key = f"tuner:{tuner_id}:completions"
        try:
            members = await self.client.smembers(key)
            return sorted([m for m in members if isinstance(m, str)])
        except Exception as e:
            logger.exception(
                f"Failed to retrieve completions for tuner {tuner_id} from Redis"
            )
            return []

    async def get_completions(self, chat_id: str) -> List[str]:
        """
        Retrieve all recorded completion IDs for a given chat ID. (Deprecated: use get_chat_completions)
        """
        return await self.get_chat_completions(chat_id)

    async def close(self) -> None:
        """
        Close the Redis connection pool.
        """
        await self.client.aclose()
