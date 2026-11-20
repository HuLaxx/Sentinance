"""Fail-closed dependency readiness without external provider traffic."""

import asyncio

from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from core.redis import get_redis
from database import async_session


async def probe_readiness() -> dict:
    """Check necessary locally configured services; never disclose DSNs or secrets.

    This does not certify external market-provider availability, end-to-end
    stream freshness, or successful cloud deployment.
    """
    services = {"postgresql": "unavailable", "redis": "unavailable"}
    try:
        async with asyncio.timeout(2.0):
            async with async_session() as session:
                await session.execute(text("SELECT 1"))
        services["postgresql"] = "available"
    except (TimeoutError, OSError, SQLAlchemyError):
        pass

    try:
        async with asyncio.timeout(2.0):
            await get_redis().ping()
        services["redis"] = "available"
    except (TimeoutError, OSError, RedisError, RuntimeError):
        pass

    return {
        "ready": all(x == "available" for x in services.values()),
        "services": services,
    }
