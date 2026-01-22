"""
Price Prediction Service

Simple rule-based and statistical predictions.
In production, replace with trained ML models.
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import random
import structlog
from models.lstm_model import LSTMPredictor

log = structlog.get_logger()

# Global mock model instance
lstm_model = LSTMPredictor()
lstm_model.warmup()


@dataclass
class PricePrediction:
    """Price prediction result."""
    symbol: str
    current_price: float
    predicted_price: float
    predicted_change_percent: float
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    horizon: str  # '1h', '24h', '7d'
    model: str
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "predicted_price": round(self.predicted_price, 2),
            "predicted_change_percent": round(self.predicted_change_percent, 2),
            "direction": self.direction,
            "confidence": round(self.confidence, 2),
            "horizon": self.horizon,
            "model": self.model,
            "timestamp": self.timestamp,
        }


def predict_momentum(
    prices: list[float], 
    current_price: float,
    horizon: str = "24h"
) -> tuple[float, float]:
    """
    Simple momentum-based prediction.
    Returns (predicted_price, confidence).
    """
    if len(prices) < 5:
        return current_price if current_price > 0 else 1.0, 0.3
    
    # Handle zero prices
    if prices[-5] == 0 or current_price == 0:
        return current_price if current_price > 0 else 1.0, 0.3
    
    # Calculate recent momentum
    recent_change = (prices[-1] - prices[-5]) / prices[-5]
    
    # Project forward
    if horizon == "1h":
        factor = 0.1
    elif horizon == "24h":
        factor = 1.0
    else:  # 7d
        factor = 3.0
    
    predicted_change = recent_change * factor
    predicted_price = current_price * (1 + predicted_change)
    
    # Confidence based on volatility
    min_price = min(prices[-5:])
    if min_price > 0:
        volatility = max(prices[-5:]) / min_price - 1
    else:
        volatility = 0.1
    confidence = max(0.3, min(0.8, 1 - volatility * 2))
    
    return predicted_price, confidence


def predict_mean_reversion(
    prices: list[float],
    current_price: float,
    horizon: str = "24h"
) -> tuple[float, float]:
    """
    Mean reversion prediction.
    Assumes price will revert to moving average.
    """
    if len(prices) < 20:
        return current_price if current_price > 0 else 1.0, 0.3
    
    mean_price = sum(prices[-20:]) / 20
    
    # Handle zero or near-zero mean price to avoid ZeroDivisionError
    if mean_price == 0 or abs(mean_price) < 1e-10:
        return current_price if current_price > 0 else 1.0, 0.3
    
    deviation = (current_price - mean_price) / mean_price
    
    # Predict reversion
    if horizon == "1h":
        reversion_factor = 0.1
    elif horizon == "24h":
        reversion_factor = 0.3
    else:
        reversion_factor = 0.5
    
    predicted_price = current_price * (1 - deviation * reversion_factor)
    confidence = min(0.7, abs(deviation) * 2 + 0.4)
    
    return predicted_price, confidence


def predict_random_walk(
    current_price: float,
    volatility: float = 0.02,
    horizon: str = "24h"
) -> tuple[float, float]:
    """
    Random walk with drift.
    Baseline model for comparison.
    """
    # Handle zero price
    if current_price <= 0:
        return 1.0, 0.3
    
    if horizon == "1h":
        steps = 1
    elif horizon == "24h":
        steps = 24
    else:
        steps = 168
    
    # Random walk
    drift = 0.001  # Small positive drift
    random_component = random.gauss(0, volatility) * (steps ** 0.5)
    predicted_change = drift * steps + random_component
    
    predicted_price = current_price * (1 + predicted_change)
    confidence = 0.3  # Random walk has low confidence
    
    return predicted_price, confidence


def generate_prediction(
    symbol: str,
    prices: list[float],
    current_price: float,
    horizon: str = "24h",
    model: str = "ensemble"
) -> PricePrediction:
    """
    Generate price prediction using specified model.
    
    Models:
    - momentum: Based on recent price trend
    - mean_reversion: Assumes price reverts to mean
    - random_walk: Baseline random model
    - lstm: Deep learning model (mock)
    - ensemble: Weighted average of all models
    """
    # Handle edge case of zero/negative prices
    if current_price <= 0:
        current_price = 1.0
    
    if model == "lstm":
        # Use our new LSTM mock
        output = lstm_model.predict(prices, horizon)
        pred_price = output.price
        confidence = output.confidence
    elif model == "momentum":
        pred_price, confidence = predict_momentum(prices, current_price, horizon)
    elif model == "mean_reversion":
        pred_price, confidence = predict_mean_reversion(prices, current_price, horizon)
    elif model == "random_walk":
        pred_price, confidence = predict_random_walk(current_price, horizon=horizon)
    else:  # ensemble
        # Weighted average of models
        m_price, m_conf = predict_momentum(prices, current_price, horizon)
        r_price, r_conf = predict_mean_reversion(prices, current_price, horizon)
        
        # Weight by confidence
        total_conf = m_conf + r_conf
        if total_conf > 0:
            pred_price = (m_price * m_conf + r_price * r_conf) / total_conf
        else:
            pred_price = current_price
        confidence = (m_conf + r_conf) / 2
    
    # Calculate change
    if current_price > 0:
        change_percent = ((pred_price - current_price) / current_price) * 100
    else:
        change_percent = 0.0
    
    # Determine direction
    if change_percent > 1:
        direction = "bullish"
    elif change_percent < -1:
        direction = "bearish"
    else:
        direction = "neutral"
    
    return PricePrediction(
        symbol=symbol,
        current_price=current_price,
        predicted_price=pred_price,
        predicted_change_percent=change_percent,
        direction=direction,
        confidence=confidence,
        horizon=horizon,
        model=model,
        timestamp=datetime.utcnow().isoformat(),
    )
