import asyncio
import json

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from redis.exceptions import RedisError

from core.redis import get_redis

log = structlog.get_logger()


class ConnectionManager:
    """
    Manages WebSocket connections and synchronizes messages across multiple
    API replicas using Redis Pub/Sub.
    """

    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.pubsub_task: asyncio.Task[None] | None = None
        self.listener_failure: BaseException | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        log.info("websocket_connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        log.info(
            "websocket_disconnected", total_connections=len(self.active_connections)
        )

    async def broadcast(self, message: dict) -> None:
        """Publish to all replicas; do not silently accept failed delivery."""
        client = get_redis()
        payload = json.dumps(message)
        try:
            await client.publish("sentinance:updates", payload)
        except RedisError as exc:
            log.error("redis_publish_failed", error=str(exc))
            raise

    async def _redis_listener(self) -> None:
        """Forward Redis messages to local connections and release Pub/Sub."""
        client = get_redis()
        pubsub = client.pubsub()
        try:
            await pubsub.subscribe("sentinance:updates")
            log.info("redis_pubsub_subscribed")

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                payload = message["data"]
                disconnected = set()
                for connection in tuple(self.active_connections):
                    try:
                        await connection.send_text(payload)
                    except (WebSocketDisconnect, OSError):
                        # Ordinary closed sockets are not Redis transport faults.
                        disconnected.add(connection)

                for connection in disconnected:
                    self.active_connections.discard(connection)
        except asyncio.CancelledError:
            log.info("redis_listener_cancelled")
            raise
        except RedisError as exc:
            log.error("redis_listener_error", error=str(exc))
            raise
        finally:
            await pubsub.aclose()

    def _report_listener_exit(self, task: asyncio.Task[None]) -> None:
        if self.pubsub_task is task:
            self.pubsub_task = None
        if task.cancelled():
            return
        failure = task.exception()
        if failure is not None:
            self.listener_failure = failure
            log.error("redis_listener_task_failed", error=str(failure))

    async def start_listener(self) -> None:
        """Start one listener task and preserve any asynchronous failure."""
        if self.pubsub_task is None:
            self.listener_failure = None
            task = asyncio.create_task(self._redis_listener())
            self.pubsub_task = task
            task.add_done_callback(self._report_listener_exit)

    async def stop_listener(self) -> None:
        """Cancel and await the listener, making shutdown deterministic."""
        task = self.pubsub_task
        self.pubsub_task = None
        if task is not None:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# Global instance
manager = ConnectionManager()
