"""
Redis Cache Service

Provides caching utilities for:
- Price data caching
- Semantic response caching
- Rate limiting
- Session management
"""

import os
import json
import hashlib
from typing import Optional, Any
from datetime import timedelta
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

_redis_client: Optional[redis.Redis] = None


async def get_redis() -> Optional[redis.Redis]:
    """Get or create Redis client."""
    global _redis_client
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await _redis_client.ping()
            log.info("redis_connected", url=REDIS_URL)
        except Exception as e:
            log.warning("redis_connection_failed", error=str(e))
            _redis_client = None
    
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
    except Exception as e:
        log.warning("cache_prices_failed", error=str(e))
        return False


async def get_cached_prices() -> Optional[list[dict]]:
    """Get cached prices."""
    client = await get_redis()
    if not client:
        return None
    
    try:
        key = f"{CACHE_PREFIX}prices:current"
        data = await client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        log.warning("get_cached_prices_failed", error=str(e))
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
    except Exception as e:
        log.warning("cache_ai_response_failed", error=str(e))
        return False


async def get_cached_ai_response(query: str) -> Optional[dict]:
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
    except Exception as e:
        log.warning("get_cached_ai_response_failed", error=str(e))
        return None


# ============================================
# RATE LIMITING
# ============================================

async def check_rate_limit(
    key: str, 
    max_requests: int = 100, 
    window_seconds: int = 60
) -> tuple[bool, int]:
    """
    Check if rate limit is exceeded.
    
    Returns:
        (is_allowed, remaining_requests)
    """
    client = await get_redis()
    if not client:
        return True, max_requests  # Allow if Redis unavailable
    
    try:
        rate_key = f"{CACHE_PREFIX}ratelimit:{key}"
        
        # Use sliding window
        current = await client.incr(rate_key)
        
        if current == 1:
            await client.expire(rate_key, window_seconds)
        
        remaining = max(0, max_requests - current)
        is_allowed = current <= max_requests
        
        if not is_allowed:
            log.warning("rate_limit_exceeded", key=key, requests=current)
        
        return is_allowed, remaining
    except Exception as e:
        log.warning("rate_limit_check_failed", error=str(e))
        return True, max_requests


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
    except Exception as e:
        log.warning("set_session_failed", error=str(e))
        return False


async def get_session(session_id: str) -> Optional[dict]:
    """Get session data."""
    client = await get_redis()
    if not client:
        return None
    
    try:
        key = f"{CACHE_PREFIX}session:{session_id}"
        data = await client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        log.warning("get_session_failed", error=str(e))
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
    except Exception as e:
        log.warning("delete_session_failed", error=str(e))
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
    except Exception as e:
        log.warning("cache_set_failed", error=str(e))
        return False


async def cache_get(key: str) -> Optional[Any]:
    """Get a value from cache."""
    client = await get_redis()
    if not client:
        return None
    
    try:
        full_key = f"{CACHE_PREFIX}{key}"
        data = await client.get(full_key)
        return json.loads(data) if data else None
    except Exception as e:
        log.warning("cache_get_failed", error=str(e))
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
    except Exception as e:
        log.warning("cache_delete_failed", error=str(e))
        return False
