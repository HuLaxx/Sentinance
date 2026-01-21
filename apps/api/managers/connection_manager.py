import asyncio
import json
from typing import Set
from fastapi import WebSocket
import structlog
from core.redis import get_redis

log = structlog.get_logger()

class ConnectionManager:
    """
    Manages WebSocket connections and synchronizes messages across multiple
    API replicas using Redis Pub/Sub.
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.pubsub_task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        log.info("websocket_connected", total_connections=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        log.info("websocket_disconnected", total_connections=len(self.active_connections))

    async def broadcast(self, message: dict):
        """
        Publish message to Redis so ALL replicas can receive it
        and forward it to their connected clients.
        """
        try:
            redis = get_redis()
            await redis.publish("sentinance:updates", json.dumps(message))
        except Exception as e:
            log.error("redis_publish_failed", error=str(e))

    async def _redis_listener(self):
        """
        Background task: Listens to Redis 'sentinance:updates' channel
        and forwards messages to local WebSocket clients.
        """
        redis = get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe("sentinance:updates")
        
        log.info("redis_pubsub_subscribed")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    payload = message["data"]
                    # Forward to all local websockets
                    disconnected = set()
                    for connection in self.active_connections:
                        try:
                            # Payload is already a JSON string from the publisher
                            # We send it directly as text/json
                            await connection.send_text(payload)
                        except Exception:
                            disconnected.add(connection)
                    
                    for conn in disconnected:
                        self.active_connections.discard(conn)
        except asyncio.CancelledError:
            log.info("redis_listener_cancelled")
        except Exception as e:
            log.error("redis_listener_error", error=str(e))
        finally:
            await pubsub.unsubscribe("sentinance:updates")

    async def start_listener(self):
        """Start the Redis subscriber task."""
        if self.pubsub_task is None:
            self.pubsub_task = asyncio.create_task(self._redis_listener())

    async def stop_listener(self):
        """Stop the Redis subscriber task."""
        if self.pubsub_task:
            self.pubsub_task.cancel()
            try:
                await self.pubsub_task
            except asyncio.CancelledError:
                pass
            self.pubsub_task = None

# Global instance
manager = ConnectionManager()
