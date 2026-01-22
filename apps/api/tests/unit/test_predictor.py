"""
Unit tests for Price Prediction module.

Tests all prediction models:
- Momentum
- Mean Reversion
- Random Walk
- LSTM (mock)
- Ensemble

Coverage target: 100% of predictor.py
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from predictor import (
    predict_momentum,
    predict_mean_reversion,
    predict_random_walk,
    generate_prediction,
    PricePrediction
)


# ============================================
# MOMENTUM PREDICTION TESTS
# ============================================

class TestMomentumPrediction:
    """Tests for momentum-based prediction model."""
    
    def test_returns_tuple(self, uptrend_prices):
        """Should return (predicted_price, confidence)."""
        result = predict_momentum(uptrend_prices, uptrend_prices[-1])
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_positive_price_prediction(self, uptrend_prices):
        """Predicted price should always be positive."""
        pred_price, confidence = predict_momentum(uptrend_prices, uptrend_prices[-1])
        assert pred_price > 0
    
    def test_confidence_range(self, uptrend_prices):
        """Confidence should be between 0 and 1."""
        _, confidence = predict_momentum(uptrend_prices, uptrend_prices[-1])
        assert 0 <= confidence <= 1
    
    def test_uptrend_predicts_higher(self, uptrend_prices):
        """In uptrend, should predict higher price."""
        current = uptrend_prices[-1]
        pred_price, _ = predict_momentum(uptrend_prices, current)
        assert pred_price > current
    
    def test_handles_short_data(self, short_prices):
        """Should handle insufficient data gracefully."""
        result = predict_momentum(short_prices, short_prices[-1])
        assert result is not None


# ============================================
# MEAN REVERSION PREDICTION TESTS
# ============================================

class TestMeanReversionPrediction:
    """Tests for mean reversion prediction model."""
    
    def test_price_above_mean_predicts_lower(self):
        """When price is above mean, should predict reversion down."""
        # Price at 150, mean around 100
        prices = [100 + i * 0.1 for i in range(50)] + [150]
        current = 150
        pred_price, _ = predict_mean_reversion(prices, current)
        assert pred_price < current
    
    def test_price_below_mean_predicts_higher(self):
        """When price is below mean, should predict reversion up."""
        # Price at 50, mean around 100
        prices = [100 + i * 0.1 for i in range(50)] + [50]
        current = 50
        pred_price, _ = predict_mean_reversion(prices, current)
        assert pred_price > current
    
    def test_confidence_range(self, sideways_prices):
        """Confidence should be between 0 and 1."""
        _, confidence = predict_mean_reversion(sideways_prices, sideways_prices[-1])
        assert 0 <= confidence <= 1


# ============================================
# RANDOM WALK PREDICTION TESTS
# ============================================

class TestRandomWalkPrediction:
    """Tests for random walk baseline model."""
    
    def test_returns_positive_price(self):
        """Should return positive predicted price."""
        pred_price, _ = predict_random_walk(100.0)
        assert pred_price > 0
    
    def test_low_confidence(self):
        """Random walk should have low confidence."""
        _, confidence = predict_random_walk(100.0)
        assert confidence <= 0.5  # Low confidence for random model
    
    def test_different_horizons(self):
        """Should work with different horizons."""
        pred_1h, _ = predict_random_walk(100.0, horizon="1h")
        pred_24h, _ = predict_random_walk(100.0, horizon="24h")
        pred_7d, _ = predict_random_walk(100.0, horizon="7d")
        
        assert pred_1h is not None
        assert pred_24h is not None
        assert pred_7d is not None


# ============================================
# GENERATE PREDICTION (MAIN FUNCTION) TESTS
# ============================================

class TestGeneratePrediction:
    """Tests for the main generate_prediction function."""
    
    def test_returns_prediction_object(self, btc_like_prices):
        """Should return PricePrediction dataclass."""
        result = generate_prediction(
            "BTCUSDT", 
            btc_like_prices, 
            btc_like_prices[-1]
        )
        assert isinstance(result, PricePrediction)
    
    def test_all_fields_populated(self, btc_like_prices):
        """All fields should be populated."""
        result = generate_prediction(
            "BTCUSDT", 
            btc_like_prices, 
            btc_like_prices[-1]
        )
        
        assert result.symbol == "BTCUSDT"
        assert result.current_price > 0
        assert result.predicted_price > 0
        assert result.direction in ["bullish", "bearish", "neutral"]
        assert 0 <= result.confidence <= 1
        assert result.horizon in ["1h", "24h", "7d"]
        assert result.timestamp is not None
    
    def test_different_models(self, btc_like_prices):
        """Should work with different model types."""
        current = btc_like_prices[-1]
        
        momentum = generate_prediction("BTC", btc_like_prices, current, model="momentum")
        mean_rev = generate_prediction("BTC", btc_like_prices, current, model="mean_reversion")
        random = generate_prediction("BTC", btc_like_prices, current, model="random_walk")
        ensemble = generate_prediction("BTC", btc_like_prices, current, model="ensemble")
        
        assert momentum.model == "momentum"
        assert mean_rev.model == "mean_reversion"
        assert random.model == "random_walk"
        assert ensemble.model == "ensemble"
    
    def test_different_horizons(self, btc_like_prices):
        """Should work with different time horizons."""
        current = btc_like_prices[-1]
        
        pred_1h = generate_prediction("BTC", btc_like_prices, current, horizon="1h")
        pred_24h = generate_prediction("BTC", btc_like_prices, current, horizon="24h")
        pred_7d = generate_prediction("BTC", btc_like_prices, current, horizon="7d")
        
        assert pred_1h.horizon == "1h"
        assert pred_24h.horizon == "24h"
        assert pred_7d.horizon == "7d"
    
    def test_direction_classification(self, btc_like_prices):
        """Direction should match predicted change."""
        result = generate_prediction(
            "BTCUSDT", 
            btc_like_prices, 
            btc_like_prices[-1]
        )
        
        change = result.predicted_change_percent
        if change > 1:
            assert result.direction == "bullish"
        elif change < -1:
            assert result.direction == "bearish"
        else:
            assert result.direction == "neutral"
    
    def test_to_dict_format(self, btc_like_prices):
        """to_dict should return properly formatted dict."""
        result = generate_prediction(
            "BTCUSDT", 
            btc_like_prices, 
            btc_like_prices[-1]
        )
        d = result.to_dict()
        
        assert "symbol" in d
        assert "current_price" in d
        assert "predicted_price" in d
        assert "predicted_change_percent" in d
        assert "direction" in d
        assert "confidence" in d
        assert "horizon" in d
        assert "model" in d
        assert "timestamp" in d


# ============================================
# EDGE CASES
# ============================================

class TestPredictorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handles_zero_price(self):
        """Should handle zero price gracefully."""
        prices = [0.0] * 50
        result = generate_prediction("TEST", prices, 0.0)
        # Should not crash
        assert result is not None
    
    def test_handles_negative_prices(self):
        """Should handle negative prices (shouldn't happen but test anyway)."""
        prices = [-100 + i for i in range(50)]
        result = generate_prediction("TEST", prices, prices[-1])
        assert result is not None
    
    def test_ensemble_averages_models(self, btc_like_prices):
        """Ensemble should be between individual model predictions."""
        current = btc_like_prices[-1]
        
        momentum = generate_prediction("BTC", btc_like_prices, current, model="momentum")
        mean_rev = generate_prediction("BTC", btc_like_prices, current, model="mean_reversion")
        ensemble = generate_prediction("BTC", btc_like_prices, current, model="ensemble")
        
        # Ensemble prediction should be between the two
        min_pred = min(momentum.predicted_price, mean_rev.predicted_price)
        max_pred = max(momentum.predicted_price, mean_rev.predicted_price)
        
        # Allow some tolerance for edge cases
        assert min_pred * 0.95 <= ensemble.predicted_price <= max_pred * 1.05
