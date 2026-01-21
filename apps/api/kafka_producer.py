"""
Kafka Producer

Produces price update events to Kafka topics.
"""

import asyncio
import os
import json
from typing import Optional
from datetime import datetime
import structlog

log = structlog.get_logger()

# Check if aiokafka is available
try:
    from aiokafka import AIOKafkaProducer
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
    """Kafka producer for price events."""
    
    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[AIOKafkaProducer] = None
        self._started = False
    
    async def start(self):
        """Start the producer."""
        if not KAFKA_AVAILABLE:
            log.warning("kafka_producer_disabled", reason="aiokafka not installed")
            return
        
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            await self.producer.start()
            self._started = True
            log.info("kafka_producer_started", servers=self.bootstrap_servers)
        except Exception as e:
            log.error("kafka_producer_start_failed", error=str(e))
    
    async def stop(self):
        """Stop the producer."""
        if self.producer and self._started:
            await self.producer.stop()
            self._started = False
            log.info("kafka_producer_stopped")
    
    async def send_price(self, symbol: str, price_data: dict):
        """Send a price update event."""
        if not self._started or not self.producer:
            return
        
        try:
            event = {
                "type": "price_update",
                "symbol": symbol,
                "data": price_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.producer.send(PRICE_TOPIC, key=symbol, value=event)
            log.debug("price_event_sent", symbol=symbol)
        except Exception as e:
            log.warning("price_event_failed", symbol=symbol, error=str(e))
    
    async def send_alert(self, alert_id: str, alert_data: dict):
        """Send an alert triggered event."""
        if not self._started or not self.producer:
            return
        
        try:
            event = {
                "type": "alert_triggered",
                "alert_id": alert_id,
                "data": alert_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.producer.send(ALERT_TOPIC, key=alert_id, value=event)
            log.info("alert_event_sent", alert_id=alert_id)
        except Exception as e:
            log.warning("alert_event_failed", alert_id=alert_id, error=str(e))
    
    async def send_news(self, article: dict):
        """Send a news article event."""
        if not self._started or not self.producer:
            return
        
        try:
            event = {
                "type": "news_article",
                "data": article,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.producer.send(NEWS_TOPIC, value=event)
            log.debug("news_event_sent")
        except Exception as e:
            log.warning("news_event_failed", error=str(e))


# Singleton instance
_producer: Optional[PriceProducer] = None


async def get_producer() -> PriceProducer:
    """Get or create the Kafka producer."""
    global _producer
    if _producer is None:
        _producer = PriceProducer()
        await _producer.start()
    return _producer


async def close_producer():
    """Close the Kafka producer."""
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
