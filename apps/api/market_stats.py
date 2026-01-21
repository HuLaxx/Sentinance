"""
Market Statistics Service

Calculates market-wide statistics and aggregations.
"""

from datetime import datetime, timedelta
from typing import Optional
import structlog

log = structlog.get_logger()


# ============================================
# MARKET STATS
# ============================================

class MarketStats:
    """Aggregated market statistics."""
    
    def __init__(self):
        self.total_market_cap = 0.0
        self.total_volume_24h = 0.0
        self.btc_dominance = 0.0
        self.eth_dominance = 0.0
        self.fear_greed_index = 50
        self.active_coins = 0
        self.gainers_losers = {"gainers": 0, "losers": 0}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "total_market_cap": self.total_market_cap,
            "total_volume_24h": self.total_volume_24h,
            "btc_dominance": self.btc_dominance,
            "eth_dominance": self.eth_dominance,
            "fear_greed_index": self.fear_greed_index,
            "fear_greed_label": self._get_fear_greed_label(),
            "active_coins": self.active_coins,
            "gainers": self.gainers_losers["gainers"],
            "losers": self.gainers_losers["losers"],
            "timestamp": self.timestamp.isoformat(),
        }
    
    def _get_fear_greed_label(self) -> str:
        if self.fear_greed_index <= 25:
            return "Extreme Fear"
        elif self.fear_greed_index <= 45:
            return "Fear"
        elif self.fear_greed_index <= 55:
            return "Neutral"
        elif self.fear_greed_index <= 75:
            return "Greed"
        else:
            return "Extreme Greed"


def calculate_market_stats(prices: dict) -> MarketStats:
    """
    Calculate market statistics from current prices.
    
    Args:
        prices: Dictionary of current prices by symbol
    
    Returns:
        MarketStats object with aggregated data
    """
    stats = MarketStats()
    
    if not prices:
        return stats
    
    total_volume = 0
    gainers = 0
    losers = 0
    btc_cap = 0
    eth_cap = 0
    
    for symbol, data in prices.items():
        volume = data.get("volume", 0)
        change = data.get("change_24h", 0)
        
        total_volume += volume
        
        if change > 0:
            gainers += 1
        elif change < 0:
            losers += 1
        
        # Track BTC and ETH for dominance
        if symbol == "BTCUSDT":
            btc_cap = data.get("price", 0) * 19_500_000  # Approx circulating supply
        elif symbol == "ETHUSDT":
            eth_cap = data.get("price", 0) * 120_000_000  # Approx circulating supply
    
    # Simulate total market cap (in production, fetch from API)
    stats.total_market_cap = 1_870_000_000_000  # $1.87T
    stats.total_volume_24h = total_volume
    
    # Calculate dominance
    if stats.total_market_cap > 0:
        stats.btc_dominance = round((btc_cap / stats.total_market_cap) * 100, 1)
        stats.eth_dominance = round((eth_cap / stats.total_market_cap) * 100, 1)
    
    stats.active_coins = len(prices)
    stats.gainers_losers = {"gainers": gainers, "losers": losers}
    
    # Calculate fear/greed based on market sentiment
    stats.fear_greed_index = calculate_fear_greed(prices)
    
    return stats


def calculate_fear_greed(prices: dict) -> int:
    """
    Calculate Fear & Greed Index based on market data.
    
    Factors:
    - Price momentum (weighted average of 24h changes)
    - Volatility
    - Volume trends
    
    Returns: 0-100 (0 = Extreme Fear, 100 = Extreme Greed)
    """
    if not prices:
        return 50
    
    # Calculate average momentum
    changes = [p.get("change_24h", 0) for p in prices.values()]
    avg_change = sum(changes) / len(changes) if changes else 0
    
    # Map -10% to +10% change to 0-100 index
    # -10% or worse = 0 (Extreme Fear)
    # +10% or better = 100 (Extreme Greed)
    momentum_score = max(0, min(100, (avg_change + 10) * 5))
    
    # Calculate volatility factor (high volatility = more fear)
    volatility = max(changes) - min(changes) if changes else 0
    volatility_score = max(0, min(100, 100 - volatility * 5))
    
    # Weighted average
    index = int(momentum_score * 0.6 + volatility_score * 0.4)
    
    return max(0, min(100, index))


# ============================================
# TOP MOVERS
# ============================================

def get_top_movers(prices: dict, limit: int = 5) -> dict:
    """
    Get top gainers and losers.
    
    Returns:
        {
            "gainers": [...],
            "losers": [...]
        }
    """
    sorted_by_change = sorted(
        prices.values(),
        key=lambda p: p.get("change_24h", 0),
        reverse=True
    )
    
    gainers = [
        {
            "symbol": p.get("symbol"),
            "price": p.get("price"),
            "change": p.get("change_24h"),
        }
        for p in sorted_by_change[:limit]
        if p.get("change_24h", 0) > 0
    ]
    
    losers = [
        {
            "symbol": p.get("symbol"),
            "price": p.get("price"),
            "change": p.get("change_24h"),
        }
        for p in reversed(sorted_by_change[-limit:])
        if p.get("change_24h", 0) < 0
    ]
    
    return {"gainers": gainers, "losers": losers}
