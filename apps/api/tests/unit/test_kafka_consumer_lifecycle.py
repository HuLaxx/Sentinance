"""Consumer lifecycle regression tests using synthetic brokers only."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiokafka.errors import KafkaConnectionError

import kafka_consumer


class BlockingBroker:
    def __init__(self):
        self.started = AsyncMock()
        self.stop = AsyncMock()
        self.commit = AsyncMock()
        self.gate = asyncio.Event()

    async def start(self):
        await self.started()

    def __aiter__(self):
        async def blocking():
            await self.gate.wait()
            if False:
                yield None

        return blocking()


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_awaits_worker(monkeypatch):
    broker = BlockingBroker()
    factory = Mock(return_value=broker)
    monkeypatch.setattr(kafka_consumer, "AIOKafkaConsumer", factory)
    monkeypatch.setattr(kafka_consumer, "KAFKA_AVAILABLE", True)
    consumer = kafka_consumer.PriceConsumer(group_id="fixture-group")

    await consumer.start()
    await consumer.start()
    assert factory.call_count == 1
    assert factory.call_args.kwargs["enable_auto_commit"] is False
    assert factory.call_args.kwargs["group_id"] == "fixture-group"
    assert consumer._task is not None

    await consumer.stop()
    assert consumer.consumer is None
    assert not consumer._running
    broker.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_broker_error_not_silenced(monkeypatch):
    broker = Mock()
    broker.start = AsyncMock(side_effect=KafkaConnectionError("offline"))
    monkeypatch.setattr(kafka_consumer, "AIOKafkaConsumer", Mock(return_value=broker))
    monkeypatch.setattr(kafka_consumer, "KAFKA_AVAILABLE", True)
    consumer = kafka_consumer.PriceConsumer()

    with pytest.raises(KafkaConnectionError, match="offline"):
        await consumer.start()
    assert not consumer._running
    assert consumer.consumer is None
    assert consumer._task is None


@pytest.mark.asyncio
async def test_missing_dependency_reports_failure(monkeypatch):
    monkeypatch.setattr(kafka_consumer, "KAFKA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="required"):
        await kafka_consumer.PriceConsumer().start()


@pytest.mark.asyncio
async def test_handler_failure_closes_client_and_exposes_task_failure(monkeypatch):
    log = Mock()
    monkeypatch.setattr(kafka_consumer, "log", log)

    class OneMessageBroker:
        def __init__(self):
            self.stop = AsyncMock()
            self.commit = AsyncMock()

        async def start(self):
            return None

        def __aiter__(self):
            async def records():
                yield SimpleNamespace(
                    topic="sentinance.prices",
                    partition=0,
                    offset=1,
                    value={"type": "price_update"},
                )

            return records()

    broker = OneMessageBroker()
    monkeypatch.setattr(kafka_consumer, "AIOKafkaConsumer", Mock(return_value=broker))
    monkeypatch.setattr(kafka_consumer, "KAFKA_AVAILABLE", True)
    consumer = kafka_consumer.PriceConsumer()

    async def fail(event):
        raise ValueError("handler failed")

    consumer.register_handler("price_update", fail)
    await consumer.start()
    task = consumer._task
    assert task is not None
    with pytest.raises(ValueError, match="handler failed"):
        await task
    await asyncio.sleep(0)

    broker.commit.assert_not_awaited()
    broker.stop.assert_awaited_once()
    assert consumer.consumer is None
    assert consumer._running is False
    assert isinstance(consumer._last_failure, ValueError)
    assert log.error.call_args.args[0] == "kafka_consumer_task_failed"
