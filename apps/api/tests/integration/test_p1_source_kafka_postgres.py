"""Fixture-sourced HTTP -> real Kafka -> real PostgreSQL acceptance test.

No live external exchange requests. Fixture provenance survives storage and
offset acknowledgement. This is not proof of Binance production access.
"""

import asyncio
import os
from decimal import Decimal
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select

from binance_aggtrades_v1 import BinanceAggregateTrades
from database import async_session
from durable_price_ingestion import DurablePriceSink, create_durable_price_consumer
from kafka_price_models import RawKafkaPriceEvent
from kafka_producer import PriceProducer
from routers.market_event_router import router as market_evidence_router
from source_checkpoint import LocalCheckpoint, ingest_checkpointed
from source_receipts import reconcile_receipts

pytestmark = pytest.mark.skipif(
    os.getenv("P1_BROKER_POSTGRES_INTEGRATION") != "1",
    reason="Needs disposable Kafka and PostgreSQL CI services",
)


@pytest.mark.asyncio
async def test_fixture_http_source_through_broker_into_postgres(tmp_path):
    identity = int(uuid4().hex[:11], 16)
    group = "sentinance-provider-it-" + uuid4().hex
    producer = PriceProducer(bootstrap_servers="127.0.0.1:9092")
    consumer = create_durable_price_consumer(
        bootstrap_servers="127.0.0.1:9092", group_id=group
    )
    stored_twice = asyncio.Event()
    seen = 0
    sink = DurablePriceSink()

    async def track_storage(record):
        nonlocal seen
        await sink.persist_record(record)
        if record.value.get("event_id") == str(identity):
            seen += 1
            if seen == 2:
                stored_twice.set()

    consumer.record_handler = track_storage

    def fixture_provider(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "BTCUSDT"
        if request.url.params["fromId"] == str(identity + 1):
            return httpx.Response(200, json=[])
        assert request.url.params["fromId"] == str(identity)
        return httpx.Response(
            200,
            json=[
                {
                    "a": identity,
                    "p": "202.123456789012",
                    "q": "0.750",
                    "f": 9,
                    "l": 9,
                    "T": 1794916800123,
                    "m": False,
                }
            ],
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(fixture_provider)
    ) as http_client:
        source = BinanceAggregateTrades(http_client, mode="fixture")
        await producer.start()
        try:
            await consumer.start()
            deadline = asyncio.get_running_loop().time() + 30
            while consumer.consumer is None or not consumer.consumer.assignment():
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError("Kafka consumer assignment timed out")
                await asyncio.sleep(0.1)
            cursor_path = tmp_path / "source.sqlite"
            for _ in range(2):
                assert (
                    await ingest_checkpointed(
                        source=source,
                        producer=producer,
                        checkpoint=LocalCheckpoint(cursor_path),
                        symbol="BTCUSDT",
                        initial_id=identity,
                        limit=1,
                    )
                    == identity + 1
                )
            assert LocalCheckpoint(cursor_path).load(
                "fixture", "BTCUSDT", identity
            ) == (identity + 1)

            row = None
            while row is None:
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError("Kafka -> PostgreSQL delivery timed out")
                async with async_session() as session:
                    row = await session.scalar(
                        select(RawKafkaPriceEvent).where(
                            RawKafkaPriceEvent.source == "binance_aggtrade_fixture",
                            RawKafkaPriceEvent.symbol == "BTCUSDT",
                            RawKafkaPriceEvent.source_event_id == str(identity),
                        )
                    )
                if row is None:
                    await asyncio.sleep(0.1)

            assert row.price == Decimal("202.123456789012")
            assert row.volume == Decimal("0.750")
            assert row.raw_event["provenance"]["mode"] == "fixture"
            assert row.raw_event["provenance"]["aggregate_trade_id"] == identity

            async with async_session() as session:
                report = await reconcile_receipts(
                    session,
                    mode="fixture",
                    symbol="BTCUSDT",
                    start_id=identity,
                    stop_id=identity + 2,
                )
            assert report.received_count == 1
            assert report.missing_ranges == ((identity + 1, identity + 2),)
            async with async_session() as session:
                complete = await reconcile_receipts(
                    session,
                    mode="fixture",
                    symbol="BTCUSDT",
                    start_id=identity,
                    stop_id=identity + 1,
                )
            assert complete.complete

            # Confirm the newly stored fixture reaches an actual FastAPI
            # endpoint through the real PostgreSQL query, not an in-memory mock.
            evidence_app = FastAPI()
            evidence_app.include_router(market_evidence_router)
            async with httpx.AsyncClient(
                app=evidence_app, base_url="http://test"
            ) as browser_api:
                response = await browser_api.get(
                    "/api/v1/market-events?symbol=BTCUSDT&limit=50"
                )
                assert response.status_code == 200
                candidates = [
                    item
                    for item in response.json()["events"]
                    if item["source_event_id"] == str(identity)
                ]
                assert len(candidates) == 1
                assert candidates[0]["source_mode"] == "fixture"
                assert candidates[0]["price"] == "202.123456789012"
                assert response.json()["live_feed_verified"] is False

            # A replay from the *same provider IDs* at another broker offset
            # must not insert a second PostgreSQL row.
            await source.publish_page(producer, "BTCUSDT", from_id=identity, limit=1)
            await asyncio.wait_for(stored_twice.wait(), timeout=30)
            async with async_session() as session:
                total = await session.scalar(
                    select(func.count())
                    .select_from(RawKafkaPriceEvent)
                    .where(
                        RawKafkaPriceEvent.source == "binance_aggtrade_fixture",
                        RawKafkaPriceEvent.symbol == "BTCUSDT",
                        RawKafkaPriceEvent.source_event_id == str(identity),
                    )
                )
                assert total == 1
            assert consumer._last_failure is None
        finally:
            await consumer.stop()
            await producer.stop()
