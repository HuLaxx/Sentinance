"""
LSTM Price Predictor (Mock)

This module implements a mock LSTM predictor for demonstration purposes.
In a real production environment, this would load a PyTorch model artifact
(e.g. from TorchServe or ONNX) and run inference on GPU.

For this portfolio project, we simulate the *behavior* of an LSTM:
- Takes a sequence of prices
- Captures momentum and volatility features
- Outputs a prediction with a confidence score
"""

import math
import random
import structlog
import numpy as np
from dataclasses import dataclass

log = structlog.get_logger()

@dataclass
class ModelOutput:
    price: float
    confidence: float
    direction: str
    features_used: list[str]

class LSTMPredictor:
    def __init__(self, model_path: str = "models/lstm_v1.pt"):
        self.model_path = model_path
        self._initialized = False
        log.info("Initialized LSTM Predictor (Mock mode)", path=model_path)

    def warmup(self):
        """Simulate loading model weights."""
        if not self._initialized:
            # Simulate latency of loading a model
            self._initialized = True
            log.info("Model weights loaded to memory")

    def predict(self, prices: list[float], horizon: str = "24h") -> ModelOutput:
        """
        Run inference on price sequence.
        
        Real logic would be:
        1. Normalize input window (e.g. min-max scaling)
        2. Tensor conversion
        3. Forward pass: output = model(input)
        4. Denormalize output
        """
        if not prices:
            return ModelOutput(0.0, 0.0, "neutral", [])

        # Feature engineering (simulated layer inputs)
        recent = np.array(prices[-30:]) if len(prices) >= 30 else np.array(prices)
        
        # 1. Trend component (simulating LSTM hidden state memory)
        # Calculate exponential weighted moving average
        weights = np.exp(np.linspace(-1., 0., len(recent)))
        weights /= weights.sum()
        weighted_avg = np.sum(recent * weights)
        
        current_price = prices[-1]
        
        # 2. Volatility component
        volatility = np.std(recent) / np.mean(recent) if len(recent) > 1 else 0.01
        
        # 3. Random noise (simulating uncertainty/dropout)
        noise = np.random.normal(0, volatility * 0.5)
        
        # Prediction logic
        if horizon == "1h":
            factor = 1.001
        elif horizon == "24h":
            factor = 1.02  # Bias slightly bullish for demo
        else:
            factor = 1.05

        # Combine inputs to form prediction
        # If trend is positive, LSTM amplifies it
        trend_strength = (weighted_avg - current_price) / current_price
        
        prediction = current_price * (1 + trend_strength + (factor - 1) + noise)
        
        # Confidence is inverse to volatility
        confidence = max(0.1, min(0.95, 1.0 - (volatility * 5)))
        
        direction = "bullish" if prediction > current_price else "bearish"
        
        return ModelOutput(
            price=prediction,
            confidence=confidence,
            direction=direction,
            features_used=["close_price_30d", "volatility_window", "rsi_14"]
        )
