"""Regression tests for market-research provider failures and programmer errors.

All provider responses are synthetic. These tests do not validate exchange uptime,
licensing, freshness, live market prices or production delivery.
"""

from datetime import UTC, datetime

import httpx
import pytest

import agent
import multi_exchange
import news_scraper


def research_state(tool):
    return {
        "query": "BTC market overview",
        "messages": [],
        "plan": [tool],
        "research_data": None,
        "analysis": None,
        "confidence": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }


class StubClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, _type, _value, _traceback):
        return None

    async def get(self, *_args, **_kwargs):
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def install_client(monkeypatch, *, response=None, error=None):
    monkeypatch.setattr(
        agent.httpx, "AsyncClient", lambda: StubClient(response=response, error=error)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "data_key"),
    [
        ("get_order_book_depth", "order_book"),
        ("calculate_technical_indicators", "indicators"),
        ("get_price_history", "price_history"),
    ],
)
async def test_http_503_is_explicitly_unavailable(monkeypatch, tool, data_key):
    response = httpx.Response(
        503,
        request=httpx.Request("GET", "https://api.binance.com/api/v3/klines"),
    )
    install_client(monkeypatch, response=response)

    result = await agent.researcher_node(research_state(tool))
    assert "503" in result["research_data"][data_key]["error"]
    assert "price" not in result["research_data"][data_key]
    assert "high_7d" not in result["research_data"][data_key]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "data_key"),
    [
        ("get_order_book_depth", "order_book"),
        ("calculate_technical_indicators", "indicators"),
        ("get_price_history", "price_history"),
    ],
)
async def test_invalid_provider_json_is_not_treated_as_valid_market_data(
    monkeypatch, tool, data_key
):
    response = httpx.Response(
        200,
        content=b"{this-is-not-json",
        request=httpx.Request("GET", "https://api.binance.com/api/v3/klines"),
    )
    install_client(monkeypatch, response=response)

    result = await agent.researcher_node(research_state(tool))
    assert result["research_data"][data_key]["error"]
    assert len(result["research_data"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        "get_order_book_depth",
        "calculate_technical_indicators",
        "get_price_history",
    ],
)
async def test_unexpected_provider_programming_error_propagates(monkeypatch, tool):
    install_client(monkeypatch, error=RuntimeError("broken internal code"))

    with pytest.raises(RuntimeError, match="broken internal code"):
        await agent.researcher_node(research_state(tool))


@pytest.mark.asyncio
async def test_current_price_http_failure_is_not_an_invented_price(monkeypatch):
    async def unavailable(_client):
        raise httpx.ConnectError("exchange offline")

    monkeypatch.setattr(multi_exchange, "fetch_binance_prices", unavailable)
    install_client(monkeypatch)

    result = await agent.researcher_node(research_state("get_current_price"))
    price = result["research_data"]["current_price"]
    assert price["symbol"] == "BTCUSDT"
    assert "exchange offline" in price["error"]
    assert "price" not in price


@pytest.mark.asyncio
async def test_current_price_programming_error_propagates(monkeypatch):
    async def buggy(_client):
        raise RuntimeError("bug in adapter")

    monkeypatch.setattr(multi_exchange, "fetch_binance_prices", buggy)
    install_client(monkeypatch)

    with pytest.raises(RuntimeError, match="bug in adapter"):
        await agent.researcher_node(research_state("get_current_price"))


@pytest.mark.asyncio
async def test_news_http_failure_is_explicitly_unavailable(monkeypatch):
    async def unavailable(*, limit):
        raise httpx.ReadTimeout("news service timed out")

    monkeypatch.setattr(news_scraper, "get_latest_news", unavailable)

    result = await agent.researcher_node(research_state("get_news_sentiment"))
    assert "news service timed out" in result["research_data"]["sentiment"]["error"]


@pytest.mark.asyncio
async def test_news_programming_error_propagates(monkeypatch):
    async def buggy(*, limit):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(news_scraper, "get_latest_news", buggy)

    with pytest.raises(RuntimeError, match="unexpected parser failure"):
        await agent.researcher_node(research_state("get_news_sentiment"))
