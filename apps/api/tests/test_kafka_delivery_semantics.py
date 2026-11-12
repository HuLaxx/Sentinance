"""Kafka event acknowledgment contracts, without requiring a broker.

These tests check the real processor and consumer loop and do not claim
end-to-end broker delivery, deduplication, or replay persistence.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kafka_consumer import EventProcessor, PriceConsumer


@pytest.mark.asyncio
async def test_processor_awaits_async_handler():
    processed = []
    processor = EventProcessor()

    async def handler(event):
        processed.append(event["id"])

    processor.register("price_update", handler)
    await processor.process({"type": "price_update", "id": "evt-1"})
    assert processed == ["evt-1"]


@pytest.mark.asyncio
async def test_processor_rejects_unknown_event_instead_of_acknowledging():
    processor = EventProcessor()
    with pytest.raises(ValueError, match="Unsupported Kafka event"):
        await processor.process({"type": "future_v99", "id": "bad"})


def single_message_broker(event):
    broker = SimpleNamespace(commit=AsyncMock(), stop=AsyncMock())

    async def messages():
        yield SimpleNamespace(value=event, topic="sentinance.prices")

    broker.__aiter__ = lambda: messages()
    return broker


class FakeBroker:
    def __init__(self, event):
        self.event = event
        self.commit = AsyncMock()

    def __aiter__(self):
        async def messages():
            yield SimpleNamespace(value=self.event, topic="sentinance.prices")
        return messages()


@pytest.mark.asyncio
async def test_successful_event_is_committed_once():
    consumer = PriceConsumer()
    broker = FakeBroker({"type": "price_update", "id": "evt-ok"})
    consumer.consumer = broker
    consumer._running = True
    observed = []

    async def record(event):
        observed.append(event["id"])

    consumer.register_handler("price_update", record)
    await consumer._consume()
    assert observed == ["evt-ok"]
    broker.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_failing_handler_does_not_commit_event():
    consumer = PriceConsumer()
    broker = FakeBroker({"type": "price_update", "id": "evt-fail"})
    consumer.consumer = broker
    consumer._running = True

    async def fail(event):
        raise RuntimeError("simulated processing failure")

    consumer.register_handler("price_update", fail)
    with pytest.raises(RuntimeError, match="simulated processing failure"):
        await consumer._consume()
    broker.commit.assert_not_awaited()
