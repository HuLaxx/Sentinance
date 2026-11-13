"""Regression tests for Redis outage, cache corruption and rate-limit failure policy."""

from unittest.mock import AsyncMock

import pytest
import redis.asyncio as redis

import cache


@pytest.mark.asyncio
async def test_unavailable_redis_cannot_bypass_rate_limit(monkeypatch):
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=None))
    assert await cache.check_rate_limit("tenant-a") == (False, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [redis.ConnectionError("offline"), redis.TimeoutError("expired deadline")],
)
async def test_rate_limit_fails_closed_on_redis_failures(monkeypatch, error):
    client = AsyncMock()
    client.incr.side_effect = error
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))
    assert await cache.check_rate_limit("tenant-a") == (False, 0)
    client.expire.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_limit_fails_closed_if_expiry_write_fails(monkeypatch):
    client = AsyncMock()
    client.incr.return_value = 1
    client.expire.side_effect = redis.ConnectionError("deadline failed")
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))
    assert await cache.check_rate_limit("tenant-a") == (False, 0)


@pytest.mark.asyncio
async def test_rate_limit_enforces_threshold_and_uses_first_request_ttl(monkeypatch):
    client = AsyncMock()
    client.incr.side_effect = [1, 2, 3]
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))

    assert await cache.check_rate_limit("tenant-a", max_requests=2) == (True, 1)
    assert await cache.check_rate_limit("tenant-a", max_requests=2) == (True, 0)
    assert await cache.check_rate_limit("tenant-a", max_requests=2) == (False, 0)

    client.expire.assert_awaited_once_with(
        "sentinance:ratelimit:tenant-a", 60
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_requests,window_seconds",
    [(0, 60), (-1, 60), (100, 0), (100, -5)],
)
async def test_invalid_limit_configuration_fails_fast(max_requests, window_seconds):
    with pytest.raises(ValueError, match="must be positive"):
        await cache.check_rate_limit(
            "tenant-a", max_requests=max_requests, window_seconds=window_seconds
        )


@pytest.mark.asyncio
async def test_programming_errors_are_not_disguised_as_redis_outages(monkeypatch):
    client = AsyncMock()
    client.incr.side_effect = RuntimeError("broken application code")
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))
    with pytest.raises(RuntimeError, match="broken application code"):
        await cache.check_rate_limit("tenant-a")


@pytest.mark.asyncio
async def test_failed_ping_does_not_publish_unhealthy_cached_client(monkeypatch):
    client = AsyncMock()
    client.ping.side_effect = redis.ConnectionError("offline")
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(cache, "REDIS_AVAILABLE", True)
    monkeypatch.setattr(cache.redis, "from_url", lambda *_args, **_kwargs: client)

    assert await cache.get_redis() is None
    assert cache._redis_client is None
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_successful_ping_publishes_healthy_client(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(cache, "_redis_client", None)
    monkeypatch.setattr(cache, "REDIS_AVAILABLE", True)
    monkeypatch.setattr(cache.redis, "from_url", lambda *_args, **_kwargs: client)

    assert await cache.get_redis() is client
    assert cache._redis_client is client
    client.ping.assert_awaited_once()
    client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_cached_price_json_does_not_become_market_data(monkeypatch):
    client = AsyncMock()
    client.get.return_value = "{invalid-json"
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))
    assert await cache.get_cached_prices() is None


@pytest.mark.asyncio
async def test_redis_cache_write_failure_returns_false(monkeypatch):
    client = AsyncMock()
    client.set.side_effect = redis.TimeoutError("cache timeout")
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))
    assert await cache.cache_prices([{"symbol": "BTCUSDT", "price": 100.0}]) is False


@pytest.mark.asyncio
async def test_unserializable_payload_is_not_mislabeled_as_redis_outage(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(cache, "get_redis", AsyncMock(return_value=client))
    with pytest.raises(TypeError):
        await cache.cache_set("invalid", {"not_json": object()})
    client.set.assert_not_awaited()
