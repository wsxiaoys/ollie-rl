import os
import logging
from typing import Optional
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


def get_redis_client(redis_url: Optional[str] = None) -> Redis:
    """
    Creates and returns a Redis client.
    Uses real Redis if redis_url or REDIS_URL environment variable is provided,
    otherwise falls back to an in-memory Fakeredis client.
    """
    url = redis_url or os.getenv("REDIS_URL")

    if url:
        import redis.asyncio as aioredis

        logger.info(f"Initializing real Redis client with URL: {url}")
        return aioredis.from_url(url, decode_responses=True)
    else:
        import fakeredis.aioredis

        logger.info("Initializing in-memory Fakeredis client for local development")
        return fakeredis.aioredis.FakeRedis(decode_responses=True)
