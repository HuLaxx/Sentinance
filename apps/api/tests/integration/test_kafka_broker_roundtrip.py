"""Real-broker tests: only enabled in the dedicated Kafka CI job.

Exercise producer + broker + consumer and replay after an uncommitted handler
failure. No production broker, downstream database or exact-once guarantee.
"""

import asyncio
import os
from uuid import uuid4

import pytest

from kafka_consumer import PriceConsumer
from kafka_producer import PriceProducer

pytestmark = pytest.mark.skipif(
    os.getenv("KAFKA_BROKER_INTEGRATION") != "1",
    reason="Requires the ephemeral CI Kafka broker and KAFKA_BROKER_INTEGRATION=1",
)

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")


async def eventually(predicate, timeout=30):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.15)
    raise AssertionError("Kafka condition not met within bounded deadline")


async def assignment_ready(consumer):
    assert consumer.consumer is not None
    await eventually(lambda: _assigned(consumer))


async def _assigned(consumer):
    assert consumer.consumer is not None
    return bool(consumer.consumer.assignment())


async def committed_record(consumer):
    assert consumer.consumer is not None
    partitions = consumer.consumer.assignment()
    if not partitions:
        return False
    committed = await consumer.consumer.committed(next(iter(partitions)))
    return committed is not None and committed > 0


@pytest.mark.asyncio
async def test_real_broker_price_roundtrip_and_group_resume():
    ident = uuid4().hex
    group = "sentinance-it-" + ident
    seen = []
    signal = asyncio.Event()
    producer = PriceProducer(bootstrap_servers=BOOTSTRAP)
    first = PriceConsumer(
        bootstrap_servers=BOOTSTRAP,
        topics=["sentinance.prices"],
        group_id=group,
        auto_offset_reset="earliest",
    )

    async def capture(event):
        if event.get("data", {}).get("test_id") == ident:
            seen.append(event)
            signal.set()

    first.register_handler("price_update", capture)
    try:
        await producer.start()
        await first.start()
        await assignment_ready(first)
        await producer.send_price("BTCUSDT", {"price": 101, "test_id": ident})
        await asyncio.wait_for(signal.wait(), timeout=30)
        await eventually(lambda: committed_record(first))
    finally:
        await first.stop()

    second = PriceConsumer(
        bootstrap_servers=BOOTSTRAP,
        topics=["sentinance.prices"],
        group_id=group,
        auto_offset_reset="earliest",
    )
    signal.clear()
    second.register_handler("price_update", capture)
    try:
        await second.start()
        await assignment_ready(second)
        await asyncio.sleep(0.5)
        assert len(seen) == 1, "Committed message must not replay on resume"
        await producer.send_price("BTCUSDT", {"price": 102, "test_id": ident})
        await asyncio.wait_for(signal.wait(), timeout=30)
        assert len(seen) == 2
    finally:
        await second.stop()
        await producer.stop()


@pytest.mark.asyncio
async def test_real_broker_uncommitted_failure_replays_after_restart():
    ident = uuid4().hex
    topic = "sentinance.test." + ident
    group = "sentinance-it-" + ident
    producer = PriceProducer(bootstrap_servers=BOOTSTRAP)
    first = PriceConsumer(
        bootstrap_servers=BOOTSTRAP,
        topics=[topic],
        group_id=group,
        auto_offset_reset="earliest",
    )

    async def fail(event):
        raise ValueError("intentional uncommitted failure")

    first.register_handler("price_update", fail)
    await producer.start()
    try:
        await first.start()
        await assignment_ready(first)
        await producer._publish(
            topic,
            {"type": "price_update", "symbol": "BTCUSDT", "id": ident},
            key=ident,
        )
        assert first._task is not None
        with pytest.raises(ValueError, match="intentional uncommitted"):
            await asyncio.wait_for(first._task, timeout=30)
        await asyncio.sleep(0)
        assert isinstance(first._last_failure, ValueError)

        replayed = asyncio.Event()
        replay = []
        second = PriceConsumer(
            bootstrap_servers=BOOTSTRAP,
            topics=[topic],
            group_id=group,
            auto_offset_reset="earliest",
        )

        async def accept(event):
            replay.append(event["id"])
            replayed.set()

        second.register_handler("price_update", accept)
        try:
            await second.start()
            await asyncio.wait_for(replayed.wait(), timeout=30)
            assert replay == [ident]
            await eventually(lambda: committed_record(second))
        finally:
            await second.stop()
    finally:
        await producer.stop()
