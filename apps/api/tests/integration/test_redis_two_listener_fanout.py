"""Broker-backed fanout to two independent Pub/Sub subscribers.

Uses the ephemeral Redis service in API CI. A single process hosts both
ConnectionManager instances; this is not a multi-process deployment proof.
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock

import pytest
import redis.asyncio as redis
from redis.exceptions import RedisError

from managers import connection_manager


@pytest.mark.asyncio
async def test_two_independent_listeners_both_receive_broker_broadcast(monkeypatch):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = redis.from_url(redis_url, decode_responses=True)
    try:
        try:
            await asyncio.wait_for(client.ping(), timeout=2)
        except (RedisError, OSError, TimeoutError) as exc:
            pytest.skip(f"Real ephemeral Redis not available: {exc}")

        monkeypatch.setattr(connection_manager, "get_redis", lambda: client)
        left = connection_manager.ConnectionManager()
        right = connection_manager.ConnectionManager()
        seen_left = asyncio.Event()
        seen_right = asyncio.Event()
        left_ws = Mock()
        right_ws = Mock()

        async def received_left(_payload):
            seen_left.set()

        async def received_right(_payload):
            seen_right.set()

        left_ws.send_text = AsyncMock(side_effect=received_left)
        right_ws.send_text = AsyncMock(side_effect=received_right)
        left.active_connections.add(left_ws)
        right.active_connections.add(right_ws)

        await left.start_listener()
        await right.start_listener()
        try:
            for _ in range(100):
                if left.listener_ready and right.listener_ready:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Both Redis subscriptions did not become ready")

            await left.broadcast({"type": "integration", "value": 7})
            await asyncio.wait_for(
                asyncio.gather(seen_left.wait(), seen_right.wait()), timeout=3
            )
            payload = '{"type": "integration", "value": 7}'
            left_ws.send_text.assert_awaited_once_with(payload)
            right_ws.send_text.assert_awaited_once_with(payload)
            assert left.listener_failure is None
            assert right.listener_failure is None
        finally:
            await left.stop_listener()
            await right.stop_listener()
    finally:
        await client.aclose()
