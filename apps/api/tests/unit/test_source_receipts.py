"""Offline receipt checks and bounded repair; no external network."""

from unittest.mock import AsyncMock

import httpx
import pytest

from binance_aggtrades_v1 import BinanceAggregateTrades
from source_receipts import ReceiptReport, reconcile_receipts, republish_missing


@pytest.mark.asyncio
async def test_receipt_missing_ranges_are_half_open_and_complete():
    db = AsyncMock()
    db.execute.return_value = [
        (str(i), {"provenance": {"mode": "fixture", "aggregate_trade_id": i}})
        for i in (11, 14)
    ]
    report = await reconcile_receipts(
        db, mode="fixture", symbol="BTCUSDT", start_id=10, stop_id=16
    )
    assert report.received_count == 2
    assert report.missing_ranges == ((10, 11), (12, 14), (15, 16))
    assert not report.complete
    db.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_reject_invalid_receipt_window_before_db_read():
    db = AsyncMock()
    for start, stop in [(0, 0), (0, 1001), (12, 10)]:
        with pytest.raises(ValueError):
            await reconcile_receipts(
                db, mode="fixture", symbol="BTCUSDT", start_id=start, stop_id=stop
            )
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_receipt_rejects_wrong_provenance():
    db = AsyncMock()
    db.execute.return_value = [
        ("10", {"provenance": {"mode": "provider", "aggregate_trade_id": 10}})
    ]
    with pytest.raises(ValueError, match="provenance"):
        await reconcile_receipts(
            db, mode="fixture", symbol="BTCUSDT", start_id=10, stop_id=11
        )


@pytest.mark.asyncio
async def test_repair_fetches_only_missing_ids_and_requires_full_page():
    calls = []

    def respond(request):
        start = int(request.url.params["fromId"])
        limit = int(request.url.params["limit"])
        calls.append((start, limit))
        return httpx.Response(
            200,
            json=[
                {"a": i, "p": "100", "q": "1", "T": 1794916800123}
                for i in range(start, start + limit)
            ],
        )

    report = ReceiptReport("fixture", "BTCUSDT", 10, 15, 2, ((10, 12), (14, 15)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        source = BinanceAggregateTrades(client)
        producer = AsyncMock()
        assert await republish_missing(report, source, producer) == 3
        assert calls == [(10, 2), (14, 1)]
        assert [
            call.args[0]["event_id"]
            for call in producer.send_price_event_v1.await_args_list
        ] == ["10", "11", "14"]
        with pytest.raises(ValueError, match="mode"):
            await republish_missing(
                report, BinanceAggregateTrades(client, mode="provider"), producer
            )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
    ) as client:
        producer = AsyncMock()
        with pytest.raises(ValueError, match="entire missing interval"):
            await republish_missing(report, BinanceAggregateTrades(client), producer)
        producer.send_price_event_v1.assert_not_awaited()
