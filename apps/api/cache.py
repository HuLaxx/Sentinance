"""
Redis Cache Service

Provides caching utilities for:
- Price data caching
- Semantic response caching
- Rate limiting
- Session management
"""

import hashlib
import json
import os
from typing import Any

import structlog

log = structlog.get_logger()

# Check if redis is available
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    log.warning("redis_not_installed", message="Install with: pip install redis")


# ============================================
# CONFIGURATION
# ============================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380")
CACHE_PREFIX = "sentinance:"
DEFAULT_TTL = 300  # 5 minutes


# ============================================
# CACHE CLIENT
# ============================================

_redis_client: redis.Redis | None = None


async def get_redis() -> redis.Redis | None:
    """Get or create Redis client."""
    global _redis_client

    if not REDIS_AVAILABLE:
        return None

    if _redis_client is None:
        # Only publish a Redis client after its health check succeeds.
        candidate = redis.from_url(REDIS_URL, decode_responses=True)
        try:
            await candidate.ping()
        except redis.RedisError as e:
            log.warning("redis_connection_failed", error_type=type(e).__name__)
            await candidate.aclose()
            return None
        _redis_client = candidate
        log.info("redis_connected")

    return _redis_client


async def close_redis():
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


# ============================================
# PRICE CACHING
# ============================================


async def cache_prices(prices: list[dict], ttl: int = 10) -> bool:
    """
    Cache price data.
    Short TTL since prices update frequently.
    """
    client = await get_redis()
    if not client:
        return False

    try:
        key = f"{CACHE_PREFIX}prices:current"
        await client.set(key, json.dumps(prices), ex=ttl)
        return True
    except redis.RedisError as e:
        log.warning("cache_prices_failed", error_type=type(e).__name__)
        return False


async def get_cached_prices() -> list[dict] | None:
    """Get cached prices."""
    client = await get_redis()
    if not client:
        return None

    try:
        key = f"{CACHE_PREFIX}prices:current"
        data = await client.get(key)
        return json.loads(data) if data else None
    except (redis.RedisError, json.JSONDecodeError) as e:
        log.warning("get_cached_prices_failed", error_type=type(e).__name__)
        return None


# ============================================
# SEMANTIC CACHE (AI Responses)
# ============================================


def _hash_query(query: str) -> str:
    """Create a hash for the query for cache lookup."""
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


async def cache_ai_response(query: str, response: dict, ttl: int = 3600) -> bool:
    """
    Cache AI response for a query.
    Longer TTL since AI analysis doesn't change rapidly.
    """
    client = await get_redis()
    if not client:
        return False

    try:
        key = f"{CACHE_PREFIX}ai:response:{_hash_query(query)}"
        await client.set(key, json.dumps(response), ex=ttl)
        log.debug("ai_response_cached", query_hash=key[-16:])
        return True
    except redis.RedisError as e:
        log.warning("cache_ai_response_failed", error_type=type(e).__name__)
        return False


async def get_cached_ai_response(query: str) -> dict | None:
    """Get cached AI response for a query."""
    client = await get_redis()
    if not client:
        return None

    try:
        key = f"{CACHE_PREFIX}ai:response:{_hash_query(query)}"
        data = await client.get(key)
        if data:
            log.debug("ai_cache_hit", query_hash=key[-16:])
            return json.loads(data)
        return None
    except (redis.RedisError, json.JSONDecodeError) as e:
        log.warning("get_cached_ai_response_failed", error_type=type(e).__name__)
        return None


# ============================================
# RATE LIMITING
# ============================================


async def check_rate_limit(
    key: str, max_requests: int = 100, window_seconds: int = 60
) -> tuple[bool, int]:
    """
    Apply the existing Redis counter/window limit.

    Deny when Redis is unavailable or returns a Redis error; a cache outage
    must not silently bypass rate limiting. This is currently a fixed-window
    counter with non-atomic first-use expiry, not a sliding-window guarantee.
    Caller wiring and distributed concurrency remain separate P0 gates.

    Returns:
        (is_allowed, remaining_requests)
    """
    if max_requests < 1 or window_seconds < 1:
        raise ValueError("Rate limit threshold and window must be positive")

    client = await get_redis()
    if client is None:
        log.error("rate_limit_unavailable_denied")
        return False, 0

    try:
        rate_key = f"{CACHE_PREFIX}ratelimit:{key}

        # Use sliding window
        current = await client.incr(rate_key)

        if current == 1:
            await client.expire(rate_key, window_seconds)

        remaining = max(0, max_requests - current)
        is_allowed = current <= max_requests

        if not is_allowed:
            log.warning("rate_limit_exceeded", key_hash=_hash_query(key), requests=current)

        return is_allowed, remaining
    except redis.RedisError as e:
        # An error after INCR/EXPIRE may leave an uncertain counter value.
        log.error("rate_limit_redis_failed_denied", error_type=type(e).__name__)
        return False, 0


# ============================================
# SESSION CACHE
# ============================================


async def set_session(session_id: str, data: dict, ttl: int = 86400) -> bool:
    """Store session data (24h default TTL)."""
    client = await get_redis()
    if not client:
        return False

    try:
        key = f"{CACHE_PREFIX}session:{session_id}"
        await client.set(key, json.dumps(data), ex=ttl)
        return True
    except redis.RedisError as e:
        log.warning("set_session_failed", error_type=type(e).__name__)
        return False


async def get_session(session_id: str) -> dict | None:
    """Get session data."""
    client = await get_redis()
    if not client:
        return None

    try:
        key = f"{CACHE_PREFIX}session:{session_id}"
        data = await client.get(key)
        return json.loads(data) if data else None
    except (redis.RedisError, json.JSONDecodeError) as e:
        log.warning("get_session_failed", error_type=type(e).__name__)
        return None


async def delete_session(session_id: str) -> bool:
    """Delete session data."""
    client = await get_redis()
    if not client:
        return False

    try:
        key = f"{CACHE_PREFIX}session:{session_id}"
        await client.delete(key)
        return True
    except redis.RedisError as e:
        log.warning("delete_session_failed", error_type=type(e).__name__)
        return False


# ============================================
# GENERIC CACHE OPERATIONS
# ============================================


async def cache_set(key: str, value: Any, ttl: int = DEFAULT_TTL) -> bool:
    """Set a value in cache."""
    client = await get_redis()
    if not client:
        return False

    try:
        full_key = f"{CACHE_PREFIX}{key}"
        await client.set(full_key, json.dumps(value), ex=ttl)
        return True
    except redis.RedisError as e:
        log.warning("cache_set_failed", error_type=type(e).__name__)
        return False


async def cache_get(key: str) -> Any | None:
    """Get a value from cache."""
    client = await get_redis()
    if not client:
        return None

    try:
        full_key = f"{CACHE_PREFIX}{key}"
        data = await client.get(full_key)
        return json.loads(data) if data else None
    except (redis.RedisError, json.JSONDecodeError) as e:
        log.warning("cache_get_failed", error_type=type(e).__name__)
        return None


async def cache_delete(key: str) -> bool:
    """Delete a value from cache."""
    client = await get_redis()
    if not client:
        return False

    try:
        full_key = f"{CACHE_PREFIX}{key}"
        await client.delete(full_key)
        return True
    except redis.RedisError as e:
        log.warning("cache_delete_failed", error_type=type(e).__name__)
        return False
