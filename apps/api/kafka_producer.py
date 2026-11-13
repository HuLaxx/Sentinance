"""
Kafka Producer

Produces price update events to Kafka topics.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import structlog

log = structlog.get_logger()

# Check if aiokafka is available
try:
    from aiokafka import AIOKafkaProducer
    from aiokafka.errors import KafkaError

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    log.warning("aiokafka_not_installed", message="Install with: pip install aiokafka")


# ============================================
# CONFIGURATION
# ============================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
PRICE_TOPIC = "sentinance.prices"
ALERT_TOPIC = "sentinance.alerts"
NEWS_TOPIC = "sentinance.news"


# ============================================
# PRODUCER
# ============================================


class PriceProducer:
    """Publish broker-acknowledged Kafka events.

    A successful call means a broker acknowledgment was received, not merely
    that the event was queued into the local producer buffer. It is *not* an
    end-to-end guarantee that downstream consumers processed the event.
    """

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self.producer: AIOKafkaProducer | None = None
        self._started = False

    async def start(self) -> None:
        """Open a producer or fail visibly; never cache a failed startup."""
        if self._started:
            return
        if not KAFKA_AVAILABLE:
            raise RuntimeError("aiokafka must be installed to publish Kafka events")

        candidate = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            enable_idempotence=True,
        )
        try:
            await candidate.start()
        except (KafkaError, TimeoutError) as exc:
            log.error("kafka_producer_start_failed", error=str(exc))
            raise

        self.producer = candidate
        self._started = True
        log.info("kafka_producer_started", servers=self.bootstrap_servers)

    async def stop(self) -> None:
        """Close a started producer and clear state after successful stop."""
        if self.producer is not None:
            await self.producer.stop()
            self.producer = None
        self._started = False
        log.info("kafka_producer_stopped")

    async def _publish(self, topic: str, event: dict, key: str | None = None) -> None:
        if not self._started or self.producer is None:
            raise RuntimeError("Kafka producer is not started")

        try:
            # send() only buffers locally; send_and_wait awaits broker ack.
            await self.producer.send_and_wait(topic, key=key, value=event)
        except (KafkaError, TimeoutError) as exc:
            # Do not turn delivery uncertainty into a successful return.
            # An application-level retry may duplicate a broker-written record.
            log.error("kafka_delivery_failed", topic=topic, error=str(exc))
            raise

    async def send_price(self, symbol: str, price_data: dict) -> None:
        """Publish a price event, returning only after Kafka acknowledges it."""
        event = {
            "type": "price_update",
            "symbol": symbol,
            "data": price_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publish(PRICE_TOPIC, event, key=symbol)
        log.debug("price_event_acknowledged", symbol=symbol)

    async def send_alert(self, alert_id: str, alert_data: dict) -> None:
        """Publish an alert event, returning only after Kafka acknowledges it."""
        event = {
            "type": "alert_triggered",
            "alert_id": alert_id,
            "data": alert_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publish(ALERT_TOPIC, event, key=alert_id)
        log.info("alert_event_acknowledged", alert_id=alert_id)

    async def send_news(self, article: dict) -> None:
        """Publish a news event, returning only after Kafka acknowledges it."""
        event = {
            "type": "news_article",
            "data": article,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await self._publish(NEWS_TOPIC, event)
        log.debug("news_event_acknowledged")


# Singleton instance
_producer: PriceProducer | None = None


async def get_producer() -> PriceProducer:
    """Get or create the Kafka producer."""
    global _producer
    if _producer is None:
        candidate = PriceProducer()
        await candidate.start()
        _producer = candidate
    return _producer


async def close_producer():
    """Close the Kafka producer."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
