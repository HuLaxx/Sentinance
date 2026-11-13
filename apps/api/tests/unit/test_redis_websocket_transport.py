"""Redis/WebSocket transport failure contracts with synthetic clients.

The tests deliberately distinguish expected disconnects/outages from code
errors. They do not exercise a live Redis server or multiple API replicas.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import WebSocketDisconnect
from redis.exceptions import ConnectionError as RedisConnectionError

import managers.connection_manager as connection_manager


class FakePubSub:
    def __init__(self, records=(), failure=None):
        self.records = records
        self.failure = failure
        self.subscribe = AsyncMock()
        self.aclose = AsyncMock()

    async def listen(self):
        for record in self.records:
            yield record
        if self.failure is not None:
            raise self.failure


def manager_with_pubsub(monkeypatch, pubsub):
    monkeypatch.setattr(
        connection_manager,
        "get_redis",
        lambda: SimpleNamespace(pubsub=lambda: pubsub),
    )
    return connection_manager.ConnectionManager()


@pytest.mark.asyncio
async def test_broadcast_sends_json_to_redis(monkeypatch):
    publish = AsyncMock(return_value=1)
    monkeypatch.setattr(
        connection_manager,
        "get_redis",
        lambda: SimpleNamespace(publish=publish),
    )
    manager = connection_manager.ConnectionManager()

    await manager.broadcast({"type": "update", "price": 101})
    publish.assert_awaited_once_with(
        "sentinance:updates", '{"type": "update", "price": 101}'
    )


@pytest.mark.asyncio
async def test_broadcast_redis_outage_is_not_silently_successful(monkeypatch):
    publish = AsyncMock(side_effect=RedisConnectionError("unavailable"))
    log = Mock()
    monkeypatch.setattr(connection_manager, "log", log)
    monkeypatch.setattr(
        connection_manager,
        "get_redis",
        lambda: SimpleNamespace(publish=publish),
    )
    manager = connection_manager.ConnectionManager()

    with pytest.raises(RedisConnectionError, match="unavailable"):
        await manager.broadcast({"type": "update"})
    assert log.error.call_args.args[0] == "redis_publish_failed"


@pytest.mark.asyncio
async def test_broadcast_programming_failure_bubbles(monkeypatch):
    publish = AsyncMock()
    monkeypatch.setattr(
        connection_manager,
        "get_redis",
        lambda: SimpleNamespace(publish=publish),
    )
    manager = connection_manager.ConnectionManager()

    with pytest.raises(TypeError):
        await manager.broadcast({"not_json": object()})
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_closed_websocket_is_removed_without_losing_healthy_client(
    monkeypatch,
):
    feed = FakePubSub(records=[{"type": "message", "data": '{"value":1}'}])
    manager = manager_with_pubsub(monkeypatch, feed)
    healthy = Mock()
    healthy.send_text = AsyncMock()
    closed = Mock()
    closed.send_text = AsyncMock(side_effect=WebSocketDisconnect(code=1006))
    manager.active_connections = {healthy, closed}

    await manager._redis_listener()

    healthy.send_text.assert_awaited_once_with('{"value":1}')
    assert healthy in manager.active_connections
    assert closed not in manager.active_connections
    feed.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_listener_redis_outage_bubbles_and_closes_pubsub(monkeypatch):
    feed = FakePubSub(failure=RedisConnectionError("feed stopped"))
    manager = manager_with_pubsub(monkeypatch, feed)

    with pytest.raises(RedisConnectionError, match="feed stopped"):
        await manager._redis_listener()
    feed.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_unexpected_websocket_error_is_not_hidden(monkeypatch):
    feed = FakePubSub(records=[{"type": "message", "data": "fixture"}])
    manager = manager_with_pubsub(monkeypatch, feed)
    broken = Mock()
    broken.send_text = AsyncMock(side_effect=ValueError("code bug"))
    manager.active_connections.add(broken)

    with pytest.raises(ValueError, match="code bug"):
        await manager._redis_listener()
    feed.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_listener_worker_failure_is_logged_and_exposed(monkeypatch):
    feed = FakePubSub(failure=RedisConnectionError("poll failed"))
    manager = manager_with_pubsub(monkeypatch, feed)
    log = Mock()
    monkeypatch.setattr(connection_manager, "log", log)

    await manager.start_listener()
    task = manager.pubsub_task
    assert task is not None
    with pytest.raises(RedisConnectionError, match="poll failed"):
        await task
    await asyncio.sleep(0)

    assert manager.pubsub_task is None
    assert isinstance(manager.listener_failure, RedisConnectionError)
    assert log.error.call_args.args[0] == "redis_listener_task_failed"
    feed.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_listener_waits_for_async_cleanup(monkeypatch):
    entered = asyncio.Event()
    blocker = asyncio.Event()

    class WaitingPubSub(FakePubSub):
        async def listen(self):
            entered.set()
            await blocker.wait()
            if False:
                yield None

    feed = WaitingPubSub()
    manager = manager_with_pubsub(monkeypatch, feed)
    await manager.start_listener()
    await asyncio.wait_for(entered.wait(), timeout=1)
    await manager.stop_listener()

    assert manager.pubsub_task is None
    feed.aclose.assert_awaited_once()
