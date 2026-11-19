"""Persistent checkpoint tests use real SQLite and offline provider fixtures."""

from unittest.mock import AsyncMock

import httpx
import pytest

from binance_aggtrades_v1 import BinanceAggregateTrades, MarketSourceGap
from source_checkpoint import CursorConflict, LocalCheckpoint, ingest_checkpointed


def trade(i):
    return {"a": i, "p": "100", "q": "1", "T": 1794916800123}


@pytest.mark.asyncio
async def test_resume_after_page_ack_and_empty_page(tmp_path):
    path = tmp_path / "cursor.sqlite"
    requested = []

    def response(request):
        cursor = int(request.url.params["fromId"])
        requested.append(cursor)
        return httpx.Response(200, json=[trade(10), trade(11)] if cursor == 10 else [])

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        source = BinanceAggregateTrades(client)
        producer = AsyncMock()
        for _ in range(2):
            result = await ingest_checkpointed(
                source=source,
                producer=producer,
                checkpoint=LocalCheckpoint(path),
                symbol="BTCUSDT",
                initial_id=10,
                limit=2,
            )
            assert result == 12
    assert requested == [10, 12]
    assert producer.send_price_event_v1.await_count == 2


@pytest.mark.asyncio
async def test_mid_page_failure_replays_from_same_cursor(tmp_path):
    path = tmp_path / "cursor.sqlite"
    requested = []

    def response(request):
        requested.append(request.url.params["fromId"])
        return httpx.Response(200, json=[trade(10), trade(11)])

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        source = BinanceAggregateTrades(client)
        producer = AsyncMock()
        producer.send_price_event_v1.side_effect = [None, TimeoutError("ack")]
        args = dict(
            source=source, producer=producer, symbol="BTCUSDT", initial_id=10, limit=2
        )
        with pytest.raises(TimeoutError, match="ack"):
            await ingest_checkpointed(checkpoint=LocalCheckpoint(path), **args)
        assert LocalCheckpoint(path).load("fixture", "BTCUSDT", 10) == 10
        producer.send_price_event_v1.side_effect = None
        assert await ingest_checkpointed(checkpoint=LocalCheckpoint(path), **args) == 12
    assert requested == ["10", "10"]


@pytest.mark.asyncio
async def test_gap_does_not_advance_or_publish(tmp_path):
    path = tmp_path / "cursor.sqlite"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[trade(10), trade(12)])
        )
    ) as client:
        producer = AsyncMock()
        with pytest.raises(MarketSourceGap):
            await ingest_checkpointed(
                source=BinanceAggregateTrades(client),
                producer=producer,
                checkpoint=LocalCheckpoint(path),
                symbol="BTCUSDT",
                initial_id=10,
                limit=2,
            )
        producer.send_price_event_v1.assert_not_awaited()
        assert LocalCheckpoint(path).load("fixture", "BTCUSDT", 10) == 10


def test_checkpoint_scopes_and_compare_and_set(tmp_path):
    path = tmp_path / "cursor.sqlite"
    a = LocalCheckpoint(path)
    b = LocalCheckpoint(path)
    assert a.load("fixture", "BTCUSDT", 10) == 10
    assert b.load("fixture", "BTCUSDT", 10) == 10
    a.advance("fixture", "BTCUSDT", 10, 12)
    with pytest.raises(CursorConflict):
        b.advance("fixture", "BTCUSDT", 10, 11)
    assert b.load("fixture", "BTCUSDT", 10) == 12
    assert b.load("provider", "BTCUSDT", 10) == 10
    assert b.load("fixture", "ETHUSDT", 10) == 10
    with pytest.raises(CursorConflict):
        b.load("fixture", "BTCUSDT", 99)
    with pytest.raises(ValueError):
        b.advance("fixture", "BTCUSDT", 12, 11)
