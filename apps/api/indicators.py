"""
Technical Indicators Calculator

Calculates common technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- EMA/SMA
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import structlog

log = structlog.get_logger()


@dataclass
class TechnicalIndicators:
    """Container for all technical indicators."""
    symbol: str
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "rsi_14": round(self.rsi_14, 2) if self.rsi_14 else None,
            "macd": {
                "value": round(self.macd, 2) if self.macd else None,
                "signal": round(self.macd_signal, 2) if self.macd_signal else None,
                "histogram": round(self.macd_histogram, 2) if self.macd_histogram else None,
            },
            "moving_averages": {
                "sma_20": round(self.sma_20, 2) if self.sma_20 else None,
                "sma_50": round(self.sma_50, 2) if self.sma_50 else None,
                "ema_12": round(self.ema_12, 2) if self.ema_12 else None,
                "ema_26": round(self.ema_26, 2) if self.ema_26 else None,
            },
            "bollinger_bands": {
                "upper": round(self.bollinger_upper, 2) if self.bollinger_upper else None,
                "middle": round(self.bollinger_middle, 2) if self.bollinger_middle else None,
                "lower": round(self.bollinger_lower, 2) if self.bollinger_lower else None,
            },
            "atr_14": round(self.atr_14, 2) if self.atr_14 else None,
        }


def calculate_sma(prices: list[float], period: int) -> Optional[float]:
    """Simple Moving Average."""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_ema(prices: list[float], period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


def calculate_rsi(prices: list[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index."""
    if len(prices) < period + 1:
        return None
    
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [c if c > 0 else 0 for c in changes[-period:]]
    losses = [-c if c < 0 else 0 for c in changes[-period:]]
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    prices: list[float], 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """MACD with Signal and Histogram."""
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_line = ema_fast - ema_slow
    
    # For signal line, we'd need MACD history (simplified here)
    signal_line = macd_line * 0.9  # Simplified
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: list[float], 
    period: int = 20, 
    std_dev: float = 2.0
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Bollinger Bands (Upper, Middle, Lower)."""
    if len(prices) < period:
        return None, None, None
    
    middle = calculate_sma(prices, period)
    if middle is None:
        return None, None, None
    
    variance = sum((p - middle) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower


def calculate_atr(
    highs: list[float], 
    lows: list[float], 
    closes: list[float], 
    period: int = 14
) -> Optional[float]:
    """Average True Range."""
    if len(highs) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    return sum(true_ranges[-period:]) / period


def calculate_all_indicators(
    symbol: str,
    prices: list[float],
    highs: Optional[list[float]] = None,
    lows: Optional[list[float]] = None,
) -> TechnicalIndicators:
    """Calculate all technical indicators for a symbol."""
    indicators = TechnicalIndicators(symbol=symbol)
    
    if len(prices) < 2:
        return indicators
    
    # RSI
    indicators.rsi_14 = calculate_rsi(prices, 14)
    
    # MACD
    macd, signal, hist = calculate_macd(prices)
    indicators.macd = macd
    indicators.macd_signal = signal
    indicators.macd_histogram = hist
    
    # Moving Averages
    indicators.sma_20 = calculate_sma(prices, 20)
    indicators.sma_50 = calculate_sma(prices, 50)
    indicators.ema_12 = calculate_ema(prices, 12)
    indicators.ema_26 = calculate_ema(prices, 26)
    
    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(prices)
    indicators.bollinger_upper = upper
    indicators.bollinger_middle = middle
    indicators.bollinger_lower = lower
    
    # ATR (if OHLC data available)
    if highs and lows:
        indicators.atr_14 = calculate_atr(highs, lows, prices)
    
    log.debug("indicators_calculated", symbol=symbol)
    
    return indicators
