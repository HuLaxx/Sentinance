"""
Kafka Consumer

Consumes events from Kafka topics for processing.
"""

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

log = structlog.get_logger()

# Check if aiokafka is available
try:
    from aiokafka import AIOKafkaConsumer, TopicPartition
    from aiokafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    log.warning("aiokafka_not_installed")


# ============================================
# CONFIGURATION
# ============================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PRICE_TOPIC = "sentinance.prices"
ALERT_TOPIC = "sentinance.alerts"


# ============================================
# EVENT HANDLERS
# ============================================

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class EventProcessor:
    """Processes events from Kafka topics."""

    def __init__(self):
        self.handlers: dict[str, list[EventHandler]] = {}

    def register(self, event_type: str, handler: EventHandler):
        """Register a handler for an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        log.info("handler_registered", event_type=event_type)

    async def process(self, event: dict[str, Any]):
        """Process an event by calling registered handlers."""
        event_type = event.get("type")
        if not event_type or event_type not in self.handlers:
            raise ValueError(f"Unsupported Kafka event type: {event_type!r}")
        for handler in self.handlers[event_type]:
            result = handler(event)
            if inspect.isawaitable(result):
                await result


# ============================================
# CONSUMER
# ============================================


class PriceConsumer:
    """Manages one Kafka consumer loop with explicit offset acknowledgments.

    Handler failures leave that record uncommitted. Broker replay also requires
    an appropriate group offset/reset policy and idempotent downstream handlers.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topics: list[str] | None = None,
        group_id: str = "sentinance-api",
        auto_offset_reset: str = "latest",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or [PRICE_TOPIC, ALERT_TOPIC]
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset
        self.consumer: AIOKafkaConsumer | None = None
        self.processor = EventProcessor()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._last_failure: BaseException | None = None

    def register_handler(self, event_type: str, handler: EventHandler):
        """Register a handler for Kafka events."""
        self.processor.register(event_type, handler)

    async def start(self) -> None:
        """Start a managed polling task, surfacing startup failures."""
        if self._running:
            return
        if not KAFKA_AVAILABLE:
            raise RuntimeError("aiokafka is required to consume Kafka events")

        candidate = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset=self.auto_offset_reset,
            enable_auto_commit=False,
        )
        try:
            await candidate.start()
        except (KafkaError, TimeoutError) as exc:
            log.error("kafka_consumer_start_failed", error=str(exc))
            raise

        self.consumer = candidate
        self._running = True
        self._last_failure = None
        self._task = asyncio.create_task(self._run_managed(), name="kafka-consumer")
        self._task.add_done_callback(self._report_task_failure)
        log.info("kafka_consumer_started", topics=self.topics, group=self.group_id)

    def _report_task_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        failure = task.exception()
        if failure is not None:
            self._last_failure = failure
            log.error("kafka_consumer_task_failed", error=str(failure))

    async def _run_managed(self) -> None:
        """Always close the underlying client, even after handler failure."""
        try:
            await self._consume()
        finally:
            self._running = False
            client = self.consumer
            self.consumer = None
            if client is not None:
                await client.stop()

    async def stop(self) -> None:
        """Cancel and await the worker before returning."""
        self._running = False
        task = self._task
        self._task = None

        if task is not None:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # A worker may never have started; explicitly close that client.
        client = self.consumer
        self.consumer = None
        if client is not None:
            await client.stop()
        log.info("kafka_consumer_stopped")

    async def _consume(self) -> None:
        """Commit only the offset of a successfully processed record."""
        client = self.consumer
        if client is None:
            return

        async for message in client:
            if not self._running:
                break

            event = message.value
            log.debug(
                "event_received",
                topic=message.topic,
                event_type=event.get("type"),
            )
            await self.processor.process(event)
            # Explicit next offset; never commit another partition's
            # unprocessed position or acknowledge a failed handler.
            partition = TopicPartition(message.topic, message.partition)
            await client.commit({partition: message.offset + 1})


# ============================================
# DEFAULT HANDLERS
# ============================================


async def handle_price_update(event: dict[str, Any]):
    """Handle price update events."""
    data = event.get("data", {})
    symbol = event.get("symbol")
    log.debug("price_update_processed", symbol=symbol, price=data.get("price"))


async def handle_alert_triggered(event: dict[str, Any]):
    """Handle alert triggered events."""
    alert_id = event.get("alert_id")
    data = event.get("data", {})
    log.info("alert_triggered_processed", alert_id=alert_id, symbol=data.get("symbol"))


# ============================================
# FACTORY
# ============================================


def create_consumer_with_handlers() -> PriceConsumer:
    """Create a consumer with default handlers registered."""
    consumer = PriceConsumer()
    consumer.register_handler("price_update", handle_price_update)
    consumer.register_handler("alert_triggered", handle_alert_triggered)
    return consumer
