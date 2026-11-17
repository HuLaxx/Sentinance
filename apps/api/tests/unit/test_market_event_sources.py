"""Offline source-ID and sequence-gap tests. No live Binance call in CI."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from market_event_sources import (
    BASE_URL,
    BinanceAggregateTradeSource,
    SourceGapError,
    publish_bounded_pages,
    validate_page,
)


def trade(identifier: int, *, price: str = "101.125") -> dict:
    return {"a": identifier, "p": price, "q": "2.5", "T": 1794913200000}


def client_for(payload, *, status=200):
    def responder(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith(BASE_URL)
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(responder))


def test_contiguous_trade_ids_become_source_owned_v1_envelopes():
    page = validate_page(
        [trade(50), trade(51)], symbol="BTCUSDT", from_id=50, page_size=2
    )
    assert page.complete is False
    assert page.next_from_id == 52
    assert [x["event_id"] for x in page.records] == ["50", "51"]
    assert all(x["source"] == "binance_aggtrades" for x in page.records)
    assert page.records[0]["data"] == {"price": "101.125", "volume": "2.5"}
    assert datetime.fromisoformat(page.records[0]["timestamp"]).tzinfo == UTC


@pytest.mark.parametrize(
    "data, expected",
    [
        ([trade(52)], 50),
        ([trade(50), trade(52)], 50),
        ([trade(50), trade(50)], 50),
        ([trade(49)], 50),
    ],
)
def test_sequence_gaps_duplicates_and_rewinds_fail_closed(data, expected):
    with pytest.raises(SourceGapError):
        validate_page(data, symbol="BTCUSDT", from_id=expected, page_size=2)


@pytest.mark.parametrize(
    "payload",
    [
        [{"a": True, "p": "1", "q": "1", "T": 1794913200000}],
        [{"a": 50, "p": "NaN", "q": "1", "T": 1794913200000}],
        [{"a": 50, "p": "1", "q": "-1", "T": 1794913200000}],
        [{"a": 50, "p": "1", "q": "1", "T": -1}],
        [{"a": 50, "p": "1", "q": "1", "T": "invalid"}],
        {"unexpected": "object"},
    ],
)
def test_malformed_provider_payload_is_not_published(payload):
    with pytest.raises((ValueError, TypeError)):
        validate_page(payload, symbol="BTCUSDT", from_id=50, page_size=2)


@pytest.mark.asyncio
async def test_bounded_paging_and_acknowledged_publishing_are_contiguous():
    calls = []

    def responder(request):
        cursor = int(request.url.params["fromId"])
        limit = int(request.url.params["limit"])
        calls.append((cursor, limit))
        if cursor == 50:
            return httpx.Response(200, json=[trade(50), trade(51)])
        return httpx.Response(200, json=[trade(52)])
    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
        source = BinanceAggregateTradeSource(client)
        publisher = AsyncMock()
        cursor = await publish_bounded_pages(
            source=source,
            producer=publisher,
            symbol="BTCUSDT",
            from_id=50,
            page_size=2,
            max_pages=3,
        )
    assert cursor == 53
    assert calls == [(50, 2), (52, 2)]
    assert publisher.send_price_v1.await_count == 3
    assert [
        c.kwargs["event_id"] for c in publisher.send_price_v1.await_args_list
    ] == ["50", "51", "52"]


@pytest.mark.asyncio
async def test_kafka_delivery_failure_does_not_skip_source_records():
    async with client_for([trade(40), trade(41)]) as client:
        producer = AsyncMock()
        producer.send_price_v1.side_effect = [None, TimeoutError("broker")]
        with pytest.raises(TimeoutError, match="broker"):
            await publish_bounded_pages(
                source=BinanceAggregateTradeSource(client),
                producer=producer,
                symbol="BTCUSDT",
                from_id=40,
                page_size=2,
            )
    assert producer.send_price_v1.await_count == 2


@pytest.mark.asyncio
async def test_provider_rate_limit_propagates_without_publish():
    async with client_for({"msg": "rate limit"}, status=429) as client:
        publisher = AsyncMock()
        with pytest.raises(httpx.HTTPStatusError):
            await publish_bounded_pages(
                source=BinanceAggregateTradeSource(client),
                producer=publisher,
                symbol="BTCUSDT",
                from_id=40,
            )
        publisher.send_price_v1.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_configuration_does_not_call_http():
    def no_network(_request):
        raise AssertionError("Network should not be accessed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_network)) as client:
        source = BinanceAggregateTradeSource(client)
        with pytest.raises(ValueError):
            await source.fetch_page(symbol="../ADMIN", from_id=5)
        with pytest.raises(ValueError):
            await source.fetch_page(symbol="BTCUSDT", from_id=-1)
        with pytest.raises(ValueError):
            await publish_bounded_pages(
                source=source,
                producer=AsyncMock(),
                symbol="BTCUSDT",
                from_id=0,
                max_pages=999,
            )


def test_empty_page_is_terminal_without_fabricating_new_ids():
    page = validate_page([], symbol="BTCUSDT", from_id=700, page_size=100)
    assert page.records == ()
    assert page.next_from_id == 700
    assert page.complete is True


def test_source_fixture_serialization_is_a_normal_json_event():
    record = validate_page(
        [trade(42)], symbol="ETHUSDT", from_id=42, page_size=1
    ).records[0]
    assert json.loads(json.dumps(record))["schema_version"] == 1
