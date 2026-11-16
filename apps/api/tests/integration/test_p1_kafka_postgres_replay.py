"""Real Kafka + PostgreSQL: crash after DB commit, replay, no duplicate rows.

Runs only in the dedicated disposable Kafka/PostgreSQL CI job. It does not
prove provider integration or ML/frontend consumption.
"""

import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from database import async_session
from durable_price_ingestion import DurablePriceSink, create_durable_price_consumer
from kafka_price_models import RawKafkaPriceEvent
from kafka_producer import PriceProducer
from price_event_contract import PRICE_V1_TOPIC

pytestmark = pytest.mark.skipif(
    os.getenv("P1_BROKER_POSTGRES_INTEGRATION") != "1",
    reason="Requires disposable real Kafka and PostgreSQL broker CI",
)
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")


async def wait_for_assignment(consumer, *, timeout: float = 30) -> None:
    end = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < end:
        if consumer.consumer is not None and consumer.consumer.assignment():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("Kafka consumer did not receive a partition assignment")


async def count_records(event_id: str) -> int:
    async with async_session() as session:
        result = await session.scalar(
            select(func.count())
            .select_from(RawKafkaPriceEvent)
            .where(RawKafkaPriceEvent.source_event_id == event_id)
        )
        return int(result or 0)


@pytest.mark.asyncio
async def test_committed_postgres_record_survives_consumer_crash_and_replay():
    test_id = uuid4().hex
    group = "sentinance-v1-durable-test-" + test_id
    producer = PriceProducer(bootstrap_servers=BOOTSTRAP)
    sink = DurablePriceSink()
    first = create_durable_price_consumer(
        bootstrap_servers=BOOTSTRAP,
        group_id=group,
    )
    event_time = datetime(2026, 11, 16, 10, 0, tzinfo=UTC)
    first_written = asyncio.Event()

    async def commit_then_crash(record):
        await sink.persist_record(record)
        if record.value.get("event_id") == test_id:
            first_written.set()
            raise RuntimeError("synthetic crash after PostgreSQL, before Kafka ack")

    first.record_handler = commit_then_crash
    await producer.start()
    try:
        await first.start()
        await wait_for_assignment(first)
        await producer.send_price_v1(
            source="fixture_exchange",
            event_id=test_id,
            symbol="BTCUSDT",
            observed_at=event_time,
            price="201.125",
            volume="10",
        )
        await asyncio.wait_for(first_written.wait(), timeout=30)
        assert first._task is not None
        with pytest.raises(RuntimeError, match="synthetic crash after PostgreSQL"):
            await asyncio.wait_for(first._task, timeout=30)
        assert await count_records(test_id) == 1

        resumed = create_durable_price_consumer(
            bootstrap_servers=BOOTSTRAP,
            group_id=group,
        )
        replay_offsets = []
        processed = asyncio.Event()

        async def record_replay(record):
            await sink.persist_record(record)
            if record.value.get("event_id") == test_id:
                replay_offsets.append(record.offset)
                if len(replay_offsets) == 2:
                    processed.set()

        resumed.record_handler = record_replay
        try:
            await resumed.start()
            await wait_for_assignment(resumed)
            # This first delivery is the committed PostgreSQL event replay,
            # because its Kafka group offset was not acknowledged.
            deadline = asyncio.get_running_loop().time() + 30
            while not replay_offsets:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("Uncommitted Kafka event was not replayed")
                await asyncio.sleep(0.1)

            # A broker-acknowledged duplicate at a *different* Kafka offset
            # has the same source event ID and must not create a second row.
            await producer.send_price_v1(
                source="fixture_exchange",
                event_id=test_id,
                symbol="BTCUSDT",
                observed_at=event_time,
                price="201.125",
                volume="10",
            )
            await asyncio.wait_for(processed.wait(), timeout=30)
            assert len(replay_offsets) == 2
            assert replay_offsets[0] != replay_offsets[1]
            assert await count_records(test_id) == 1
            assert resumed._last_failure is None
        finally:
            await resumed.stop()
    finally:
        await producer.stop()

    assert await count_records(test_id) == 1
