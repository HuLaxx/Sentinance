import os
from typing import Optional
import redis.asyncio as redis
import structlog

log = structlog.get_logger()

# Global Redis client
_redis_client: Optional[redis.Redis] = None


async def init_redis() -> None:
    """Initialize the global Redis client."""
    global _redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6380/0")
    try:
        _redis_client = redis.from_url(
            redis_url, 
            encoding="utf-8", 
            decode_responses=True
        )
        await _redis_client.ping()
        log.info("redis_connected", url=redis_url)
    except Exception as e:
        log.error("redis_connection_failed", error=str(e))
        raise e


async def close_redis() -> None:
    """Close the global Redis client."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        log.info("redis_closed")
        _redis_client = None


def get_redis() -> redis.Redis:
    """Get the global Redis client instance."""
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_redis() first.")
    return _redis_client
