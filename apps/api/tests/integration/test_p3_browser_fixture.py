"""Fixture source -> real Kafka -> PostgreSQL -> Parquet -> forecast report."""

import asyncio
import hashlib
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select

from binance_aggtrades_v1 import BinanceAggregateTrades
from database import async_session, close_db
from durable_price_ingestion import DurablePriceSink, create_durable_price_consumer
from forecast_model import evaluate_model_snapshot, report_bytes
from kafka_price_models import RawKafkaPriceEvent
from kafka_producer import PriceProducer
from price_lake import export_ledger_snapshot
from source_checkpoint import LocalCheckpoint, ingest_checkpointed

pytestmark = pytest.mark.skipif(
    os.getenv("P3_BROWSER_FIXTURE_ENABLED") != "1",
    reason="Requires dedicated disposable Kafka/Postgres browser workflow",
)


@pytest.mark.asyncio
async def test_prepare_browser_evidence(tmp_path):
    start_id = 900001
    fixture = [
        {
            "a": start_id + i,
            "p": str(100 + i + i % 3),
            "q": "2",
            "T": 1794916800000 + i * 60000,
        }
        for i in range(24)
    ]

    def respond(request):
        assert request.url.params["symbol"] == "BTCUSDT"
        assert request.url.params["fromId"] == str(start_id)
        return httpx.Response(200, json=fixture)

    producer = PriceProducer(bootstrap_servers="127.0.0.1:9092")
    consumer = create_durable_price_consumer(
        bootstrap_servers="127.0.0.1:9092", group_id="browser-" + uuid4().hex
    )
    sink = DurablePriceSink()
    consumer.record_handler = sink.persist_record
    try:
        await producer.start()
        await consumer.start()
        deadline = asyncio.get_running_loop().time() + 30
        while not (consumer.consumer and consumer.consumer.assignment()):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            cursor = await ingest_checkpointed(
                source=BinanceAggregateTrades(client, mode="fixture"),
                producer=producer,
                checkpoint=LocalCheckpoint(tmp_path / "cursor.sqlite"),
                symbol="BTCUSDT",
                initial_id=start_id,
                limit=24,
            )
        assert cursor == start_id + 24
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            async with async_session() as session:
                count = await session.scalar(
                    select(func.count()).select_from(RawKafkaPriceEvent)
                )
            if count == 24:
                break
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.1)
    finally:
        await consumer.stop()
        await producer.stop()
    try:
        manifest = await export_ledger_snapshot(tmp_path)
        assert manifest["input_count"] == manifest["curated_minute_count"] == 24
        snapshot = tmp_path / "snapshots" / manifest["snapshot_id"]
        digest = hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
        report = evaluate_model_snapshot(
            snapshot,
            digest,
            "binance_aggtrade_fixture",
            "BTCUSDT",
            train_end=12,
            validation_end=18,
        )
        assert report["test_metrics"]["ridge_ar1"]["count"] == 6
        out = Path(os.environ["P3_BROWSER_REPORT_PATH"])
        content = report_bytes(report)
        out.write_bytes(content)
        out.with_suffix(".sha256").write_text(
            hashlib.sha256(content).hexdigest(), encoding="utf-8"
        )
    finally:
        await close_db()
