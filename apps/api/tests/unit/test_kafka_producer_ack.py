"""Kafka producer acknowledgment/failure contracts using a mocked client.

No broker, storage, consumer delivery, replay, or Kafka exactly-once workflow
is demonstrated by these unit tests.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from aiokafka.errors import KafkaConnectionError

import kafka_producer


def ready_producer():
    producer = kafka_producer.PriceProducer()
    sender = Mock()
    sender.send_and_wait = AsyncMock()
    producer.producer = sender
    producer._started = True
    return producer, sender


@pytest.mark.asyncio
async def test_all_event_types_await_broker_acknowledgement():
    producer, sender = ready_producer()

    await producer.send_price("BTCUSDT", {"price": 123})
    await producer.send_alert("alert-1", {"fired": True})
    await producer.send_news({"title": "fixture"})

    calls = sender.send_and_wait.await_args_list
    assert len(calls) == 3
    assert [call.args[0] for call in calls] == [
        kafka_producer.PRICE_TOPIC,
        kafka_producer.ALERT_TOPIC,
        kafka_producer.NEWS_TOPIC,
    ]
    assert [call.kwargs["key"] for call in calls] == [
        "BTCUSDT",
        "alert-1",
        None,
    ]
    for call in calls:
        stamp = datetime.fromisoformat(call.kwargs["value"]["timestamp"])
        assert stamp.utcoffset().total_seconds() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("send_price", ("BTCUSDT", {"price": 1})),
        ("send_alert", ("alert-1", {"status": "triggered"})),
        ("send_news", ({"title": "fixture"},)),
    ],
)
async def test_stopped_producer_never_silently_drops_event(method, args):
    producer = kafka_producer.PriceProducer()
    with pytest.raises(RuntimeError, match="not started"):
        await getattr(producer, method)(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("send_price", ("BTCUSDT", {"price": 1})),
        ("send_alert", ("alert-1", {"status": "triggered"})),
        ("send_news", ({"title": "fixture"},)),
    ],
)
async def test_broker_delivery_error_reaches_caller(monkeypatch, method, args):
    producer, sender = ready_producer()
    sender.send_and_wait.side_effect = KafkaConnectionError("broker unavailable")
    log = Mock()
    monkeypatch.setattr(kafka_producer, "log", log)

    with pytest.raises(KafkaConnectionError, match="broker unavailable"):
        await getattr(producer, method)(*args)

    sender.send_and_wait.assert_awaited_once()
    assert log.error.call_args.args[0] == "kafka_delivery_failed"
    assert log.error.call_args.kwargs["topic"] in {
        kafka_producer.PRICE_TOPIC,
        kafka_producer.ALERT_TOPIC,
        kafka_producer.NEWS_TOPIC,
    }
    log.info.assert_not_called()
    log.debug.assert_not_called()


@pytest.mark.asyncio
async def test_send_does_not_finish_before_broker_ack():
    producer, sender = ready_producer()
    broker_acked = asyncio.Event()

    async def slow_ack(*args, **kwargs):
        await broker_acked.wait()

    sender.send_and_wait.side_effect = slow_ack
    task = asyncio.create_task(producer.send_price("BTCUSDT", {"price": 1}))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        broker_acked.set()
        await task
        sender.send_and_wait.assert_awaited_once()
    finally:
        broker_acked.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_requests_idempotent_all_replica_acknowledgment(monkeypatch):
    sender = Mock()
    sender.start = AsyncMock()
    sender.stop = AsyncMock()
    factory = Mock(return_value=sender)
    monkeypatch.setattr(kafka_producer, "AIOKafkaProducer", factory)
    monkeypatch.setattr(kafka_producer, "KAFKA_AVAILABLE", True)
    producer = kafka_producer.PriceProducer(bootstrap_servers="broker:9092")

    await producer.start()
    await producer.start()

    assert producer._started is True
    assert producer.producer is sender
    sender.start.assert_awaited_once()
    assert factory.call_count == 1
    assert factory.call_args.kwargs["acks"] == "all"
    assert factory.call_args.kwargs["enable_idempotence"] is True
    assert factory.call_args.kwargs["bootstrap_servers"] == "broker:9092"
    await producer.stop()
    sender.stop.assert_awaited_once()
    assert producer._started is False
    assert producer.producer is None


@pytest.mark.asyncio
async def test_startup_failure_is_raised_without_active_producer(monkeypatch):
    sender = Mock()
    sender.start = AsyncMock(side_effect=KafkaConnectionError("cannot connect"))
    monkeypatch.setattr(kafka_producer, "AIOKafkaProducer", Mock(return_value=sender))
    monkeypatch.setattr(kafka_producer, "KAFKA_AVAILABLE", True)
    producer = kafka_producer.PriceProducer()

    with pytest.raises(KafkaConnectionError, match="cannot connect"):
        await producer.start()

    assert producer.producer is None
    assert producer._started is False


@pytest.mark.asyncio
async def test_disabled_kafka_dependency_is_reported_not_silenced(monkeypatch):
    monkeypatch.setattr(kafka_producer, "KAFKA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="aiokafka must be installed"):
        await kafka_producer.PriceProducer().start()


@pytest.mark.asyncio
async def test_failed_factory_does_not_cache_broken_singleton(monkeypatch):
    class BrokenProducer:
        async def start(self):
            raise KafkaConnectionError("no cluster")

    monkeypatch.setattr(kafka_producer, "PriceProducer", BrokenProducer)
    monkeypatch.setattr(kafka_producer, "_producer", None)

    with pytest.raises(KafkaConnectionError, match="no cluster"):
        await kafka_producer.get_producer()
    assert kafka_producer._producer is None
