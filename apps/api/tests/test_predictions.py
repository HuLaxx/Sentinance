"""
Predictions Endpoint Tests

Tests for ML predictions, LSTM model, and technical indicators.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_predict_endpoint_btc(client: TestClient):
    """Test prediction endpoint for Bitcoin."""
    response = client.get("/api/predict/BTCUSDT")
    assert response.status_code in [200, 404, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "symbol" in data or "prediction" in data


def test_predict_endpoint_eth(client: TestClient):
    """Test prediction endpoint for Ethereum."""
    response = client.get("/api/predict/ETHUSDT")
    assert response.status_code in [200, 404, 503]


def test_predict_invalid_symbol(client: TestClient):
    """Test prediction with invalid symbol."""
    response = client.get("/api/predict/INVALIDSYMBOL")
    assert response.status_code in [200, 400, 404]


def test_indicators_endpoint_btc(client: TestClient):
    """Test technical indicators endpoint for Bitcoin."""
    response = client.get("/api/indicators/BTCUSDT")
    assert response.status_code in [200, 404, 503]
    
    if response.status_code == 200:
        data = response.json()
        # Should contain RSI, MACD, or other indicators
        assert isinstance(data, dict)


def test_indicators_endpoint_eth(client: TestClient):
    """Test technical indicators endpoint for Ethereum."""
    response = client.get("/api/indicators/ETHUSDT")
    assert response.status_code in [200, 404, 503]


class TestPredictorModule:
    """Unit tests for predictor module."""
    
    def test_predictor_import(self):
        """Test predictor can be imported."""
        from predictor import generate_prediction
        assert generate_prediction is not None
    
    def test_indicators_import(self):
        """Test indicators module can be imported."""
        from indicators import calculate_all_indicators
        assert calculate_all_indicators is not None


class TestLSTMModel:
    """Unit tests for LSTM model."""
    
    def test_model_config(self):
        """Test model config can be created."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "ml"
        ))
        from lstm_model import ModelConfig
        
        config = ModelConfig()
        assert config.input_size == 5
        assert config.hidden_size == 64
        assert config.num_layers == 2
        assert config.sequence_length == 60
    
    def test_lstm_model_creation(self):
        """Test LSTM model can be instantiated."""
        try:
            from lstm_model import LSTMModel, ModelConfig, TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                pytest.skip("PyTorch not installed")
            
            config = ModelConfig()
            model = LSTMModel(config)
            assert model is not None
        except ImportError:
            pytest.skip("LSTM model dependencies not available")
