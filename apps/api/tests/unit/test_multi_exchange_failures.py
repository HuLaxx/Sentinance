"""Provider failure-policy contracts; all fixtures use an in-memory HTTP transport."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

import multi_exchange


@pytest.mark.asyncio
async def test_binance_retains_valid_symbol_but_discards_rate_limited_and_malformed():
    def respond(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["symbol"]
        if symbol == "BTCUSDT":
            return httpx.Response(
                200,
                json={
                    "lastPrice": "100",
                    "priceChangePercent": "1.5",
                    "volume": "40",
                    "highPrice": "101",
                    "lowPrice": "98",
                },
            )
        if symbol == "ETHUSDT":
            return httpx.Response(429)
        if symbol == "SOLUSDT":
            return httpx.Response(200, json={"lastPrice": "bad-number"})
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        prices = await multi_exchange.fetch_binance_prices(client)

    assert list(prices) == ["BTCUSDT"]
    assert prices["BTCUSDT"]["price"] == 100.0
    stamp = datetime.fromisoformat(prices["BTCUSDT"]["timestamp"])
    assert stamp.utcoffset() == timedelta(0)


@pytest.mark.asyncio
async def test_coinbase_does_not_fabricate_stats_when_provider_is_rate_limited():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ticker"):
            return httpx.Response(200, json={"price": "100.0"})
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        prices = await multi_exchange.fetch_coinbase_prices(client)

    assert prices == {}


@pytest.mark.asyncio
async def test_kraken_outage_returns_no_observation():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _req: httpx.Response(503))
    ) as client:
        prices = await multi_exchange.fetch_kraken_prices(client)
    assert prices == {}


@pytest.mark.asyncio
async def test_adapter_programming_error_is_not_hidden_as_empty_exchange_data(
    monkeypatch,
):
    monkeypatch.setattr(
        multi_exchange,
        "fetch_binance_prices",
        AsyncMock(side_effect=RuntimeError("unexpected adapter bug")),
    )
    monkeypatch.setattr(
        multi_exchange, "fetch_coinbase_prices", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        multi_exchange, "fetch_kraken_prices", AsyncMock(return_value={})
    )
    with pytest.raises(RuntimeError, match="unexpected adapter bug"):
        await multi_exchange.fetch_all_exchanges()


@pytest.mark.asyncio
async def test_aggregated_fallback_has_one_observation_per_exchange(
    monkeypatch,
):
    cb = {"exchange": "coinbase", "symbol": "BTCUSDT", "price": 100.0}
    kr = {"exchange": "kraken", "symbol": "BTCUSDT", "price": 101.0}
    monkeypatch.setattr(
        multi_exchange, "fetch_binance_prices", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        multi_exchange,
        "fetch_coinbase_prices",
        AsyncMock(return_value={"BTCUSDT": cb}),
    )
    monkeypatch.setattr(
        multi_exchange,
        "fetch_kraken_prices",
        AsyncMock(return_value={"BTCUSDT": kr}),
    )

    aggregated, by_exchange = await multi_exchange.fetch_all_exchanges()

    assert aggregated["BTCUSDT"] == cb
    assert by_exchange["binance"] == []
    assert by_exchange["coinbase"] == [cb]
    assert by_exchange["kraken"] == [kr]


@pytest.mark.asyncio
async def test_provider_adapter_does_not_disguise_code_bug_as_network_outage():
    def raise_bug(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("provider wrapper bug")

    async with httpx.AsyncClient(transport=httpx.MockTransport(raise_bug)) as client:
        with pytest.raises(RuntimeError, match="provider wrapper bug"):
            await multi_exchange.fetch_kraken_prices(client)
