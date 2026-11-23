"""Isolated Redis Pub/Sub WebSocket listener process for fault-injection CI.

Run only from verify_redis_restart_processes.py. Each worker has its own
Python interpreter, Redis client, listener task and synthetic WebSocket.
"""

import asyncio
import json
import os
import signal
from typing import cast

import redis.asyncio as redis
from fastapi import WebSocket

from managers import connection_manager


def emit(event: str, **details: object) -> None:
    print(
        json.dumps({"event": event, "pid": os.getpid(), **details}),
        flush=True,
    )


class RecordingWebSocket:
    async def send_text(self, payload: str) -> None:
        emit("message", payload=json.loads(payload))


async def main() -> None:
    url = os.environ["SENTINANCE_RESTART_REDIS_URL"]
    client = redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=1,
    )
    # Intentionally uses an isolated Redis connection per OS process.
    connection_manager.get_redis = lambda: client
    manager = connection_manager.ConnectionManager(
        max_redis_retries=30,
        retry_base_seconds=0.1,
        retry_cap_seconds=0.4,
    )
    # The synthetic recording socket implements send_text but is not a
    # framework-created WebSocket instance.
    manager.active_connections.add(cast(WebSocket, RecordingWebSocket()))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    await manager.start_listener()
    was_ready = False

    try:
        while not stop.is_set():
            ready = manager.listener_ready
            if ready and not was_ready:
                emit("ready", retries=manager.listener_retries)
            elif was_ready and not ready:
                emit("offline", retries=manager.listener_retries)
            was_ready = ready

            if manager.listener_failure is not None:
                emit(
                    "fatal",
                    cause=type(manager.listener_failure).__name__,
                )
                raise RuntimeError("Redis Pub/Sub listener permanently failed")
            await asyncio.sleep(0.02)
    finally:
        await manager.stop_listener()
        await client.aclose()
        emit("stopped")


if __name__ == "__main__":
    asyncio.run(main())
