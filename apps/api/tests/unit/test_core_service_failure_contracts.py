"""Offline service error-policy tests: typed provider outage vs code errors."""

import httpx
import pytest
import redis

import observability
from services import price_stream


@pytest.mark.asyncio
async def test_price_service_http_outage_does_not_create_fake_prices(monkeypatch):
    class BrokenProvider:
        async def get_prices(self):
            raise httpx.ConnectError("synthetic exchange outage")

    monkeypatch.setattr(price_stream, "get_connector", BrokenProvider)
    monkeypatch.setattr(price_stream, "CURRENT_PRICES", {})
    monkeypatch.setattr(price_stream, "PRICE_HISTORY", {})
    assert await price_stream.fetch_and_update_prices() == []
    assert price_stream.CURRENT_PRICES == {}
    assert price_stream.PRICE_HISTORY == {}


@pytest.mark.asyncio
async def test_price_service_programming_error_propagates(monkeypatch):
    class BrokenCode:
        async def get_prices(self):
            raise RuntimeError("bad connection handling implementation")

    monkeypatch.setattr(price_stream, "get_connector", BrokenCode)
    with pytest.raises(RuntimeError, match="implementation"):
        await price_stream.fetch_and_update_prices()


@pytest.mark.asyncio
async def test_price_service_missing_required_schema_does_not_silently_drop(
    monkeypatch,
):
    class MalformedProvider:
        async def get_prices(self):
            return [{"symbol": "BTCUSDT", "name": "BTC"}]

    monkeypatch.setattr(price_stream, "get_connector", MalformedProvider)
    monkeypatch.setattr(price_stream, "CURRENT_PRICES", {})
    monkeypatch.setattr(price_stream, "PRICE_HISTORY", {})
    with pytest.raises(KeyError, match="price"):
        await price_stream.fetch_and_update_prices()


def test_dependency_health_reports_redis_connection_outage(monkeypatch):
    class OfflineRedis:
        def __init__(self, **_kwargs):
            pass

        def ping(self):
            raise redis.exceptions.ConnectionError("synthetic offline")

    monkeypatch.setattr(redis, "Redis", OfflineRedis)
    health = observability.check_dependency_health()
    assert health["redis"] is False
    assert health["database"] is None


def test_dependency_health_does_not_disguise_programming_defect(monkeypatch):
    class BrokenRedis:
        def __init__(self, **_kwargs):
            pass

        def ping(self):
            raise RuntimeError("health-check code defect")

    monkeypatch.setattr(redis, "Redis", BrokenRedis)
    with pytest.raises(RuntimeError, match="health-check code defect"):
        observability.check_dependency_health()


def test_dependency_health_reports_healthy_redis(monkeypatch):
    class HealthyRedis:
        def __init__(self, **_kwargs):
            pass

        def ping(self):
            return True

    monkeypatch.setattr(redis, "Redis", HealthyRedis)
    health = observability.check_dependency_health()
    assert health["redis"] is True
