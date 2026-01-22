"""
Unit tests for Technical Indicators module.

Tests all indicator calculations for correctness:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence) 
- Bollinger Bands
- SMA/EMA
- ATR

Coverage target: 100% of indicators.py
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_all_indicators,
    TechnicalIndicators
)


# ============================================
# SMA TESTS
# ============================================

class TestSMA:
    """Tests for Simple Moving Average calculation."""
    
    def test_sma_with_sufficient_data(self, uptrend_prices):
        """SMA should calculate correctly with enough data."""
        sma = calculate_sma(uptrend_prices, 20)
        assert sma is not None
        assert isinstance(sma, float)
    
    def test_sma_insufficient_data(self, short_prices):
        """SMA should return None with insufficient data."""
        sma = calculate_sma(short_prices, 20)
        assert sma is None
    
    def test_sma_exact_calculation(self):
        """SMA should be the average of last N prices."""
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        sma = calculate_sma(prices, 3)
        # Last 3 prices: 30, 40, 50 → avg = 40
        assert sma == pytest.approx(40.0, rel=0.01)
    
    def test_sma_period_equals_length(self):
        """SMA with period = data length should be overall average."""
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        sma = calculate_sma(prices, 5)
        assert sma == pytest.approx(30.0, rel=0.01)


# ============================================
# EMA TESTS
# ============================================

class TestEMA:
    """Tests for Exponential Moving Average calculation."""
    
    def test_ema_with_sufficient_data(self, uptrend_prices):
        """EMA should calculate correctly with enough data."""
        ema = calculate_ema(uptrend_prices, 12)
        assert ema is not None
        assert isinstance(ema, float)
    
    def test_ema_insufficient_data(self, short_prices):
        """EMA should return None with insufficient data."""
        ema = calculate_ema(short_prices, 20)
        assert ema is None
    
    def test_ema_weights_recent_prices(self, uptrend_prices):
        """EMA should be higher than SMA in uptrend (weights recent prices more)."""
        ema = calculate_ema(uptrend_prices, 20)
        sma = calculate_sma(uptrend_prices, 20)
        # In uptrend, EMA > SMA because recent prices are higher
        assert ema > sma


# ============================================
# RSI TESTS
# ============================================

class TestRSI:
    """Tests for Relative Strength Index calculation."""
    
    def test_rsi_range(self, uptrend_prices):
        """RSI should always be between 0 and 100."""
        rsi = calculate_rsi(uptrend_prices, 14)
        assert rsi is not None
        assert 0 <= rsi <= 100
    
    def test_rsi_uptrend_high(self, uptrend_prices):
        """RSI should be high (>70) in strong uptrend."""
        rsi = calculate_rsi(uptrend_prices, 14)
        assert rsi > 70  # Overbought territory
    
    def test_rsi_downtrend_low(self, downtrend_prices):
        """RSI should be low (<30) in strong downtrend."""
        rsi = calculate_rsi(downtrend_prices, 14)
        assert rsi < 30  # Oversold territory
    
    def test_rsi_insufficient_data(self, short_prices):
        """RSI should return None with insufficient data."""
        rsi = calculate_rsi(short_prices, 14)
        assert rsi is None
    
    def test_rsi_all_gains(self):
        """RSI should be 100 when there are only gains."""
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                  110, 111, 112, 113, 114, 115]
        rsi = calculate_rsi(prices, 14)
        assert rsi == 100.0


# ============================================
# MACD TESTS (CRITICAL - Bug Was Fixed Here)
# ============================================

class TestMACD:
    """Tests for MACD calculation - especially signal line."""
    
    def test_macd_returns_three_values(self, uptrend_prices):
        """MACD should return tuple of (macd, signal, histogram)."""
        result = calculate_macd(uptrend_prices)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_macd_with_sufficient_data(self, uptrend_prices):
        """MACD should calculate correctly with enough data."""
        macd, signal, histogram = calculate_macd(uptrend_prices)
        assert macd is not None
        assert signal is not None
        assert histogram is not None
    
    def test_macd_signal_is_not_simplified(self, uptrend_prices):
        """Signal line should NOT be MACD * 0.9 (the old bug)."""
        macd, signal, histogram = calculate_macd(uptrend_prices)
        
        # The old bug was: signal = macd * 0.9
        # This should NOT be true anymore
        simplified_signal = macd * 0.9
        assert abs(signal - simplified_signal) > 0.001, \
            "Signal line should be EMA of MACD history, not MACD * 0.9"
    
    def test_macd_histogram_is_difference(self, uptrend_prices):
        """Histogram should equal MACD line minus signal line."""
        macd, signal, histogram = calculate_macd(uptrend_prices)
        expected_histogram = macd - signal
        assert histogram == pytest.approx(expected_histogram, rel=0.001)
    
    def test_macd_insufficient_data(self):
        """MACD should return None with insufficient data."""
        short_prices = [100 + i for i in range(30)]  # Need 35 for default settings
        macd, signal, histogram = calculate_macd(short_prices)
        assert macd is None
        assert signal is None
        assert histogram is None
    
    def test_macd_in_uptrend_positive(self, uptrend_prices):
        """MACD should be positive in uptrend."""
        macd, signal, histogram = calculate_macd(uptrend_prices)
        assert macd > 0
    
    def test_macd_custom_periods(self, btc_like_prices):
        """MACD should work with custom periods."""
        macd, signal, histogram = calculate_macd(
            btc_like_prices, fast=8, slow=21, signal=5
        )
        assert macd is not None
        assert signal is not None


# ============================================
# BOLLINGER BANDS TESTS
# ============================================

class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""
    
    def test_bollinger_returns_three_bands(self, uptrend_prices):
        """Bollinger should return (upper, middle, lower)."""
        result = calculate_bollinger_bands(uptrend_prices)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_bollinger_ordering(self, uptrend_prices):
        """Upper > Middle > Lower should always be true."""
        upper, middle, lower = calculate_bollinger_bands(uptrend_prices)
        assert upper > middle
        assert middle > lower
    
    def test_bollinger_middle_is_sma(self, uptrend_prices):
        """Middle band should equal SMA(20)."""
        upper, middle, lower = calculate_bollinger_bands(uptrend_prices, period=20)
        sma = calculate_sma(uptrend_prices, 20)
        assert middle == pytest.approx(sma, rel=0.001)
    
    def test_bollinger_wider_in_volatile_market(self, volatile_prices, sideways_prices):
        """Bands should be wider in volatile market."""
        vol_upper, vol_mid, vol_lower = calculate_bollinger_bands(volatile_prices)
        side_upper, side_mid, side_lower = calculate_bollinger_bands(sideways_prices)
        
        vol_width = vol_upper - vol_lower
        side_width = side_upper - side_lower
        
        # Volatile market should have wider bands
        assert vol_width > side_width


# ============================================
# ATR TESTS
# ============================================

class TestATR:
    """Tests for Average True Range calculation."""
    
    def test_atr_with_ohlc_data(self):
        """ATR should calculate with proper OHLC data."""
        # Generate OHLC data
        closes = [100 + i for i in range(20)]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        
        atr = calculate_atr(highs, lows, closes, period=14)
        assert atr is not None
        assert atr > 0
    
    def test_atr_insufficient_data(self):
        """ATR should return None with insufficient data."""
        closes = [100, 101, 102]
        highs = [102, 103, 104]
        lows = [98, 99, 100]
        
        atr = calculate_atr(highs, lows, closes, period=14)
        assert atr is None


# ============================================
# INTEGRATION: calculate_all_indicators
# ============================================

class TestCalculateAllIndicators:
    """Tests for the main calculate_all_indicators function."""
    
    def test_returns_technical_indicators_object(self, btc_like_prices):
        """Should return TechnicalIndicators dataclass."""
        result = calculate_all_indicators("BTCUSDT", btc_like_prices)
        assert isinstance(result, TechnicalIndicators)
    
    def test_all_fields_populated(self, btc_like_prices):
        """With enough data, all fields should be populated."""
        result = calculate_all_indicators("BTCUSDT", btc_like_prices)
        
        assert result.symbol == "BTCUSDT"
        assert result.rsi_14 is not None
        assert result.macd is not None
        assert result.macd_signal is not None
        assert result.sma_20 is not None
        assert result.ema_12 is not None
        assert result.bollinger_upper is not None
    
    def test_to_dict_format(self, btc_like_prices):
        """to_dict should return properly formatted dict."""
        result = calculate_all_indicators("BTCUSDT", btc_like_prices)
        d = result.to_dict()
        
        assert "symbol" in d
        assert "rsi_14" in d
        assert "macd" in d
        assert "value" in d["macd"]
        assert "signal" in d["macd"]
        assert "histogram" in d["macd"]
        assert "moving_averages" in d
        assert "bollinger_bands" in d
    
    def test_handles_short_data(self, short_prices):
        """Should handle insufficient data gracefully."""
        result = calculate_all_indicators("BTCUSDT", short_prices)
        assert result.symbol == "BTCUSDT"
        # Most indicators should be None
        assert result.rsi_14 is None
        assert result.macd is None
