import logging
from typing import List, Optional

from .redis_client import get_redis_client

logger = logging.getLogger(__name__)


class ChatStorage:
    """
    Handles persistence of chat completion IDs within a chat ID.
    Uses Redis in production, falling back to an in-memory Fakeredis client for development.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.client = get_redis_client(redis_url)

    async def record_completion(
        self, chat_id: str, completion_id: str, ttl_seconds: int = 604800
    ) -> None:
        """
        Record a chat completion ID under a chat ID, and set/renew the 7-day TTL.
        """
        key = f"chat:{chat_id}:completions"
        try:
            # We use a pipeline to perform sadd and expire atomically/efficiently
            async with self.client.pipeline() as pipe:
                await pipe.sadd(key, completion_id)
                await pipe.expire(key, ttl_seconds)
                await pipe.execute()
            logger.debug(
                f"Recorded completion {completion_id} under chat {chat_id} with TTL {ttl_seconds}s"
            )
        except Exception as e:
            logger.exception(
                f"Failed to record completion {completion_id} for chat {chat_id} in Redis"
            )

    async def get_completions(self, chat_id: str) -> List[str]:
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

    async def close(self) -> None:
        """
        Close the Redis connection pool.
        """
        await self.client.aclose()
