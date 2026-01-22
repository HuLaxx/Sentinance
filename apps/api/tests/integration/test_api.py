"""
Integration tests for FastAPI endpoints.

Tests all REST API endpoints:
- Health checks
- Price endpoints
- Indicators endpoints
- Prediction endpoints
- Chat endpoint
- Alerts endpoints

Coverage target: All API routes in main.py
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi.testclient import TestClient


# ============================================
# HEALTH ENDPOINT TESTS
# ============================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_returns_200(self, api_client):
        """GET /health should return 200."""
        response = api_client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_ok_status(self, api_client):
        """GET /health should return status: ok."""
        response = api_client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
    
    def test_liveness_returns_200(self, api_client):
        """GET /healthz should return 200."""
        response = api_client.get("/healthz")
        assert response.status_code == 200
    
    def test_readiness_returns_200(self, api_client):
        """GET /ready should return 200."""
        response = api_client.get("/ready")
        assert response.status_code == 200


# ============================================
# PRICE ENDPOINT TESTS
# ============================================

class TestPriceEndpoints:
    """Tests for price-related endpoints."""
    
    def test_list_prices_returns_200(self, api_client):
        """GET /api/prices should return 200."""
        response = api_client.get("/api/prices")
        assert response.status_code == 200
    
    def test_list_prices_returns_array(self, api_client):
        """GET /api/prices should return prices array."""
        response = api_client.get("/api/prices")
        data = response.json()
        assert "prices" in data
        assert isinstance(data["prices"], list)
    
    def test_get_single_price_btc(self, api_client):
        """GET /api/prices/BTCUSDT should return BTC price."""
        response = api_client.get("/api/prices/BTCUSDT")
        # May return 200 with price or 404 if not cached
        assert response.status_code in [200, 404]
    
    def test_price_history_returns_array(self, api_client):
        """GET /api/prices/BTCUSDT/history should return history."""
        response = api_client.get("/api/prices/BTCUSDT/history")
        assert response.status_code in [200, 404]


# ============================================
# INDICATORS ENDPOINT TESTS
# ============================================

class TestIndicatorsEndpoints:
    """Tests for technical indicators endpoints."""
    
    def test_get_indicators_returns_200(self, api_client):
        """GET /api/indicators/BTCUSDT should return 200."""
        response = api_client.get("/api/indicators/BTCUSDT")
        assert response.status_code == 200
    
    def test_indicators_has_rsi(self, api_client):
        """Response should include RSI."""
        response = api_client.get("/api/indicators/BTCUSDT")
        data = response.json()
        assert "rsi_14" in data
    
    def test_indicators_has_macd(self, api_client):
        """Response should include MACD."""
        response = api_client.get("/api/indicators/BTCUSDT")
        data = response.json()
        assert "macd" in data
        assert "value" in data["macd"]
        assert "signal" in data["macd"]
        assert "histogram" in data["macd"]
    
    def test_indicators_has_bollinger(self, api_client):
        """Response should include Bollinger Bands."""
        response = api_client.get("/api/indicators/BTCUSDT")
        data = response.json()
        assert "bollinger_bands" in data


# ============================================
# PREDICTION ENDPOINT TESTS
# ============================================

class TestPredictionEndpoints:
    """Tests for price prediction endpoints."""
    
    def test_get_prediction_returns_200(self, api_client):
        """GET /api/predict/BTCUSDT should return 200."""
        response = api_client.get("/api/predict/BTCUSDT")
        assert response.status_code == 200
    
    def test_prediction_with_horizon(self, api_client):
        """Should accept horizon parameter."""
        response = api_client.get("/api/predict/BTCUSDT?horizon=1h")
        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == "1h"
    
    def test_prediction_with_model(self, api_client):
        """Should accept model parameter."""
        response = api_client.get("/api/predict/BTCUSDT?model=momentum")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "momentum"
    
    def test_prediction_has_required_fields(self, api_client):
        """Response should have all required fields."""
        response = api_client.get("/api/predict/BTCUSDT")
        data = response.json()
        
        required_fields = [
            "symbol", "current_price", "predicted_price",
            "predicted_change_percent", "direction", "confidence",
            "horizon", "model", "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ============================================
# CHAT ENDPOINT TESTS
# ============================================

class TestChatEndpoints:
    """Tests for AI chat endpoints."""
    
    def test_chat_returns_200(self, api_client):
        """POST /api/chat should return 200."""
        response = api_client.post("/api/chat", json={
            "message": "What is Bitcoin?",
            "history": []
        })
        assert response.status_code == 200
    
    def test_chat_returns_content(self, api_client):
        """Response should have content field."""
        response = api_client.post("/api/chat", json={
            "message": "What is BTC price?",
            "history": []
        })
        data = response.json()
        assert "content" in data
        assert len(data["content"]) > 0
    
    def test_chat_returns_metadata(self, api_client):
        """Response should have metadata."""
        response = api_client.post("/api/chat", json={
            "message": "Analyze BTC",
            "history": []
        })
        data = response.json()
        assert "metadata" in data
    
    def test_chat_suggestions(self, api_client):
        """GET /api/chat/suggestions should return suggestions."""
        response = api_client.get("/api/chat/suggestions")
        assert response.status_code == 200


# ============================================
# ALERTS ENDPOINT TESTS
# ============================================

class TestAlertsEndpoints:
    """Tests for price alerts endpoints."""
    
    def test_create_alert_returns_200(self, api_client):
        """POST /api/alerts should create alert."""
        response = api_client.post("/api/alerts", json={
            "symbol": "BTCUSDT",
            "alert_type": "price_above",
            "target_value": 100000
        })
        assert response.status_code == 200
    
    def test_create_alert_returns_alert_object(self, api_client):
        """Response should return created alert."""
        response = api_client.post("/api/alerts", json={
            "symbol": "BTCUSDT",
            "alert_type": "price_below",
            "target_value": 80000
        })
        data = response.json()
        assert "id" in data
        assert data["symbol"] == "BTCUSDT"
        assert data["alert_type"] == "price_below"


# ============================================
# ERROR HANDLING TESTS
# ============================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_on_unknown_route(self, api_client):
        """Unknown routes should return 404."""
        response = api_client.get("/api/unknown/route")
        assert response.status_code == 404
    
    def test_chat_requires_message(self, api_client):
        """POST /api/chat without message should fail."""
        response = api_client.post("/api/chat", json={})
        assert response.status_code in [400, 422]
