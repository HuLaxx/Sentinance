"""Bounded Redis WebSocket recovery contracts with synthetic transport.

No test here proves a real cross-process Redis relay or message replay.
Pub/Sub only delivers events to subscribers connected at publish time.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from managers import connection_manager


def fast_manager(*, retries=2):
    return connection_manager.ConnectionManager(
        max_redis_retries=retries,
        retry_base_seconds=0.001,
        retry_cap_seconds=0.01,
    )


@pytest.mark.asyncio
async def test_transient_outage_recovers_without_reporting_terminal_failure(
    monkeypatch,
):
    manager = fast_manager()
    calls = []
    resumed = asyncio.Event()
    keep_listening = asyncio.Event()

    async def intermittent_listener():
        calls.append(1)
        if len(calls) == 1:
            raise RedisConnectionError("synthetic Redis disconnection")
        resumed.set()
        await keep_listening.wait()

    monkeypatch.setattr(manager, "_redis_listener", intermittent_listener)
    await manager.start_listener()
    await asyncio.wait_for(resumed.wait(), timeout=1)

    assert len(calls) == 2
    assert manager.listener_retries == 1
    assert manager.listener_failure is None
    assert manager.pubsub_task is not None

    await manager.stop_listener()
    assert manager.pubsub_task is None
    assert manager.listener_failure is None


@pytest.mark.asyncio
async def test_permanent_redis_outage_exhausts_bounded_retries(monkeypatch):
    manager = fast_manager(retries=2)
    calls = []

    async def always_broken():
        calls.append(1)
        raise RedisConnectionError("synthetic outage")

    monkeypatch.setattr(manager, "_redis_listener", always_broken)
    await manager.start_listener()
    task = manager.pubsub_task
    assert task is not None

    with pytest.raises(RedisConnectionError, match="synthetic outage"):
        await asyncio.wait_for(task, timeout=1)
    await asyncio.sleep(0)
    assert len(calls) == 3
    assert manager.listener_retries == 2
    assert isinstance(manager.listener_failure, RedisConnectionError)
    assert manager.pubsub_task is None


@pytest.mark.asyncio
async def test_silent_subscriber_end_is_not_misreported_as_healthy(monkeypatch):
    manager = fast_manager(retries=0)

    async def ended():
        return None

    monkeypatch.setattr(manager, "_redis_listener", ended)
    await manager.start_listener()
    task = manager.pubsub_task
    assert task is not None
    with pytest.raises(RedisError, match="stream ended unexpectedly"):
        await task
    await asyncio.sleep(0)
    assert isinstance(manager.listener_failure, RedisError)


@pytest.mark.asyncio
async def test_programming_exception_propagates_without_reconnect(monkeypatch):
    manager = fast_manager(retries=3)
    calls = []

    async def buggy_listener():
        calls.append(1)
        raise ValueError("subscriber parser bug")

    monkeypatch.setattr(manager, "_redis_listener", buggy_listener)
    await manager.start_listener()
    task = manager.pubsub_task
    assert task is not None
    with pytest.raises(ValueError, match="parser bug"):
        await task
    assert calls == [1]
    assert manager.listener_retries == 0


@pytest.mark.asyncio
async def test_cancel_during_retry_wait_does_not_create_false_failure(monkeypatch):
    manager = connection_manager.ConnectionManager(
        max_redis_retries=4,
        retry_base_seconds=30,
        retry_cap_seconds=30,
    )
    retry_started = asyncio.Event()

    async def broken():
        raise RedisConnectionError("offline")

    logger = Mock()
    logger.warning.side_effect = lambda *_args, **_kwargs: retry_started.set()
    monkeypatch.setattr(connection_manager, "log", logger)
    monkeypatch.setattr(manager, "_redis_listener", broken)
    await manager.start_listener()
    await asyncio.wait_for(retry_started.wait(), timeout=1)
    await manager.stop_listener()
    assert manager.pubsub_task is None
    assert manager.listener_failure is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_redis_retries": -1},
        {"retry_base_seconds": 0},
        {"retry_base_seconds": 3, "retry_cap_seconds": 2},
    ],
)
def test_invalid_retry_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        connection_manager.ConnectionManager(**kwargs)


@pytest.mark.asyncio
async def test_two_local_listener_instances_receive_one_shared_broadcast(
    monkeypatch,
):
    """Synthetic shared broker; does not establish actual multi-process fanout."""
    class FakeBroker:
        def __init__(self):
            self.subscribers = []
            self.both_ready = asyncio.Event()

        def pubsub(self):
            broker = self

            class Subscriber:
                def __init__(self):
                    self.queue = asyncio.Queue()

                async def subscribe(self, _channel):
                    broker.subscribers.append(self)
                    if len(broker.subscribers) == 2:
                        broker.both_ready.set()

                async def listen(self):
                    while True:
                        yield await self.queue.get()

                async def aclose(self):
                    if self in broker.subscribers:
                        broker.subscribers.remove(self)

            return Subscriber()

        async def publish(self, _channel, payload):
            for subscriber in tuple(self.subscribers):
                await subscriber.queue.put({"type": "message", "data": payload})
            return len(self.subscribers)

    broker = FakeBroker()
    monkeypatch.setattr(connection_manager, "get_redis", lambda: broker)
    left = fast_manager()
    right = fast_manager()
    delivered_left = asyncio.Event()
    delivered_right = asyncio.Event()
    a = Mock()
    b = Mock()

    async def receive_left(_payload):
        delivered_left.set()

    async def receive_right(_payload):
        delivered_right.set()

    a.send_text = AsyncMock(side_effect=receive_left)
    b.send_text = AsyncMock(side_effect=receive_right)
    left.active_connections.add(a)
    right.active_connections.add(b)
    await left.start_listener()
    await right.start_listener()
    try:
        await asyncio.wait_for(broker.both_ready.wait(), timeout=1)
        await left.broadcast({"type": "test", "value": 42})
        await asyncio.wait_for(
            asyncio.gather(delivered_left.wait(), delivered_right.wait()), timeout=1
        )
        a.send_text.assert_awaited_once_with('{"type": "test", "value": 42}')
        b.send_text.assert_awaited_once_with('{"type": "test", "value": 42}')
        assert left.listener_ready is True
        assert right.listener_ready is True
    finally:
        await left.stop_listener()
        await right.stop_listener()
    assert left.listener_ready is False
    assert right.listener_ready is False
