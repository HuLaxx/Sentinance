import asyncio
from datetime import datetime
import structlog
from exchange_connector import get_connector
from alerts_service import get_alerts_service
from managers.connection_manager import manager

log = structlog.get_logger()

# Shared state (still useful for API lookups, but NOT for broadcasting reliance)
CURRENT_PRICES: dict = {}
PRICE_HISTORY: dict = {}
MAX_HISTORY_POINTS = 20000
PRICE_UPDATE_INTERVAL_SECONDS = 5

async def fetch_and_update_prices():
    """Fetch prices from Binance (crypto) and yFinance (indices), update local cache."""
    try:
        connector = get_connector()
        prices = await connector.get_prices()
        
        for price in prices:
            symbol = price["symbol"]
            CURRENT_PRICES[symbol] = {
                "symbol": symbol,
                "name": price["name"],
                "price": price["price"],
                "change_24h": price["priceChangePercent"],
                "volume": price.get("volume", 0),
                "high": price.get("high", 0),
                "low": price.get("low", 0),
            }
            
            # Record to price history
            if symbol not in PRICE_HISTORY:
                PRICE_HISTORY[symbol] = []
            PRICE_HISTORY[symbol].append({
                "price": price["price"],
                "volume": price.get("volume", 0),
                "timestamp": datetime.utcnow().isoformat(),
            })
            # Keep only last N points
            if len(PRICE_HISTORY[symbol]) > MAX_HISTORY_POINTS:
                PRICE_HISTORY[symbol] = PRICE_HISTORY[symbol][-MAX_HISTORY_POINTS:]
        
        log.info("prices_updated", count=len(prices))
        return prices
    except Exception as e:
        log.error("price_fetch_failed", error=str(e))
        return []


async def price_stream_task():
    """
    Background task that fetches prices and publishes updates via Redis.
    Note: In a multi-replica setup, this task will run on ALL replicas.
    This is redundant fetching (fetching 2x if 2 replicas), but acceptable 
    for simple architecture to ensure reliability if one pod dies.
    (A leader election system would be better for perfect efficiency, but complexity is higher).
    """
    triggered_alerts_queue = []

    while True:
        # Fetch real prices
        prices = await fetch_and_update_prices()
        
        # Check alerts
        if CURRENT_PRICES:
            alerts_service = get_alerts_service()
            triggered = alerts_service.check_alerts(CURRENT_PRICES)
            
            for alert in triggered:
                triggered_alerts_queue.append({
                    "type": "alert_triggered",
                    "alert": {
                        "id": alert.id,
                        "symbol": alert.symbol,
                        "alert_type": alert.alert_type,
                        "target_value": alert.target_value,
                        "message": alert.message or f"{alert.symbol} hit {alert.alert_type} target!",
                        "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None
                    }
                })
                log.info("alert_notification_queued", alert_id=alert.id)
        
        # Broadcast via Redis (Manager handles the publishing)
        if prices:
            await manager.broadcast({
                "type": "price_update",
                "prices": prices
            })
        
        while triggered_alerts_queue:
            notification = triggered_alerts_queue.pop(0)
            await manager.broadcast(notification)
        
        await asyncio.sleep(PRICE_UPDATE_INTERVAL_SECONDS)
