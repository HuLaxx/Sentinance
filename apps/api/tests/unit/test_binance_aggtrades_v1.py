"""Offline Binance aggregate-trade source contracts. No market network calls."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from binance_aggtrades_v1 import (
    BinanceAggregateTrades,
    MarketSourceGap,
    ProviderRateLimit,
)


def trades(*ids: int) -> list[dict]:
    return [
        {
            "a": ident,
            "p": "104.125",
            "q": "0.250",
            "f": ident * 2,
            "l": ident * 2 + 1,
            "T": 1794916800123 + i,
            "m": False,
        }
        for i, ident in enumerate(ids)
    ]


@pytest.mark.asyncio
async def test_provider_trade_ids_utc_and_fixture_provenance_preserved():
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=trades(500, 501))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client, mode="fixture")
        events = await source.fetch_page("BTCUSDT", from_id=500, limit=2)

    assert len(calls) == 1
    assert calls[0].url.path == "/api/v3/aggTrades"
    assert calls[0].url.params["fromId"] == "500"
    assert [event["event_id"] for event in events] == ["500", "501"]
    assert all(event["source"] == "binance_aggtrade" for event in events)
    assert all(event["provenance"]["mode"] == "fixture" for event in events)
    assert events[0]["provenance"]["aggregate_trade_id"] == 500
    assert datetime.fromisoformat(events[0]["timestamp"]).tzinfo == UTC
    assert events[0]["timestamp"].endswith("+00:00")
    assert events[0]["data"] == {"price": "104.125", "volume": "0.250"}


@pytest.mark.asyncio
async def test_complete_validated_page_is_published_with_broker_ack():
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=trades(500, 501))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client)
        producer = AsyncMock()
        assert await source.publish_page(
            producer, "BTCUSDT", from_id=500, limit=2
        ) == 2

    assert producer.send_price_event_v1.await_count == 2
    assert [
        call.args[0]["event_id"]
        for call in producer.send_price_event_v1.await_args_list
    ] == ["500", "501"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_rows",
    [
        trades(500, 502),
        trades(501),
        trades(500, 500),
        [{**trades(500)[0], "p": "-1"}],
        [{**trades(500)[0], "T": "2026-11-17"}],
        [{**trades(500)[0], "q": "NaN"}],
        [{"a": 500, "p": "100", "q": "1"}],
    ],
)
async def test_gap_or_invalid_page_sends_nothing(bad_rows):
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bad_rows)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client)
        producer = AsyncMock()
        with pytest.raises((ValueError, TypeError, MarketSourceGap)):
            await source.publish_page(producer, "BTCUSDT", from_id=500, limit=2)
        producer.send_price_event_v1.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [418, 429])
async def test_rate_limits_and_bans_stop_without_retries(status):
    count = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client, max_attempts=3)
        with pytest.raises(ProviderRateLimit):
            await source.fetch_page("BTCUSDT", from_id=500)
    assert count == 1


@pytest.mark.asyncio
async def test_http_403_does_not_retry():
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client)
        with pytest.raises(httpx.HTTPStatusError):
            await source.fetch_page("BTCUSDT", from_id=500)
    assert calls == 1


@pytest.mark.asyncio
async def test_transient_503_uses_bounded_retry(monkeypatch):
    count = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=trades(500))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client)
        delay = AsyncMock()
        monkeypatch.setattr(source, "_pause", delay)
        assert len(await source.fetch_page("BTCUSDT", from_id=500)) == 1
    assert count == 2
    delay.assert_awaited_once_with(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("symbol", "from_id", "limit"),
    [
        ("../../internal", 500, 2),
        ("BTCUSDT", -1, 2),
        ("BTCUSDT", True, 2),
        ("BTCUSDT", 0, 1001),
        ("BTCUSDT", 0, True),
    ],
)
async def test_invalid_fetch_args_reject_before_http(symbol, from_id, limit):
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError):
            await BinanceAggregateTrades(client).fetch_page(
                symbol, from_id=from_id, limit=limit
            )
    assert requests == 0
