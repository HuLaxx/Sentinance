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
    from aiokafka import AIOKafkaConsumer

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
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                log.error("handler_error", event_type=event_type, error=str(exc))
                raise


# ============================================
# CONSUMER
# ============================================


class PriceConsumer:
    """Kafka consumer for price and alert events."""

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topics: list[str] | None = None,
        group_id: str = "sentinance-api",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or [PRICE_TOPIC, ALERT_TOPIC]
        self.group_id = group_id
        self.consumer: AIOKafkaConsumer | None = None
        self.processor = EventProcessor()
        self._running = False

    def register_handler(self, event_type: str, handler: EventHandler):
        """Register an event handler."""
        self.processor.register(event_type, handler)

    async def start(self):
        """Start the consumer."""
        if not KAFKA_AVAILABLE:
            log.warning("kafka_consumer_disabled", reason="aiokafka not installed")
            return

        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=False,
            )
            await self.consumer.start()
            self._running = True
            log.info("kafka_consumer_started", topics=self.topics)

            # Start consuming
            asyncio.create_task(self._consume())
        except Exception as e:
            log.error("kafka_consumer_start_failed", error=str(e))

    async def stop(self):
        """Stop the consumer."""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
            log.info("kafka_consumer_stopped")

    async def _consume(self):
        """Consume messages from Kafka."""
        if not self.consumer:
            return

        try:
            async for message in self.consumer:
                if not self._running:
                    break

                try:
                    event = message.value
                    log.debug(
                        "event_received",
                        topic=message.topic,
                        event_type=event.get("type"),
                    )
                    await self.processor.process(event)
                    # At-least-once processing: acknowledge only successful handlers.
                    await self.consumer.commit()
                except Exception as exc:
                    log.error("event_processing_failed_uncommitted", error=str(exc))
                    raise
        except Exception as e:
            log.error("consume_error", error=str(e))
            # Preserve failure visibility; an unacknowledged offset must be replayed.
            raise


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
