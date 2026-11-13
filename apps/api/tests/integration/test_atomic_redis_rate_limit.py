"""Real Redis integration tests for atomic fixed-window rate limiting.

Run in CI where the Redis 7 service is available; skip on developers' machines
without REDIS_URL. The unit tests still test failure handling without Redis.
"""

import asyncio
import os
from uuid import uuid4

import pytest
import redis.asyncio as redis

import cache


@pytest.mark.asyncio
async def test_atomic_counter_concurrency_and_expiry(monkeypatch):
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        pytest.skip("Real Redis is required for atomic integration test")

    client = redis.from_url(redis_url, decode_responses=True)
    key = f"sentinance-atomic-test-{uuid4().hex}"
    full_key = f"{cache.CACHE_PREFIX}ratelimit:{key}"
    try:
        await client.ping()
        monkeypatch.setattr(cache, "get_redis", lambda: asyncio.sleep(0, result=client))
        decisions = await asyncio.gather(
            *(cache.evaluate_rate_limit(key, 7, 45) for _ in range(20))
        )
        assert sum(decision.allowed for decision in decisions) == 7
        assert sum(not decision.allowed for decision in decisions) == 13
        assert all(decision.available for decision in decisions)
        assert await client.get(full_key) == "20"
        assert 0 < await client.ttl(full_key) <= 45
    finally:
        await client.delete(full_key)
        await client.aclose()


@pytest.mark.asyncio
async def test_atomic_counter_repairs_existing_unexpiring_key(monkeypatch):
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        pytest.skip("Real Redis is required for atomic integration test")

    client = redis.from_url(redis_url, decode_responses=True)
    key = f"sentinance-legacy-test-{uuid4().hex}"
    full_key = f"{cache.CACHE_PREFIX}ratelimit:{key}"
    try:
        await client.set(full_key, "2")
        assert await client.ttl(full_key) == -1
        monkeypatch.setattr(cache, "get_redis", lambda: asyncio.sleep(0, result=client))
        decision = await cache.evaluate_rate_limit(key, 10, 35)
        assert decision == cache.RateLimitDecision(True, 7, True)
        assert 0 < await client.ttl(full_key) <= 35
    finally:
        await client.delete(full_key)
        await client.aclose()
