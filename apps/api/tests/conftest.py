"""
Pytest fixtures for Sentinance API tests.

Provides reusable test fixtures for:
- Price data generation
- API client
- Database connections
- Redis connections
"""
import pytest
import asyncio
from typing import List, Dict
from fastapi.testclient import TestClient
import httpx


# ============================================
# PRICE DATA FIXTURES
# ============================================

@pytest.fixture
def uptrend_prices() -> List[float]:
    """Generate 100 prices in an uptrend for testing."""
    base = 100.0
    return [base + i * 0.5 + (i % 5) * 0.1 for i in range(100)]


@pytest.fixture
def downtrend_prices() -> List[float]:
    """Generate 100 prices in a downtrend for testing."""
    base = 200.0
    return [base - i * 0.5 - (i % 5) * 0.1 for i in range(100)]


@pytest.fixture
def sideways_prices() -> List[float]:
    """Generate 100 prices in sideways movement."""
    import math
    base = 100.0
    return [base + 5 * math.sin(i / 5) for i in range(100)]


@pytest.fixture
def volatile_prices() -> List[float]:
    """Generate 100 highly volatile prices."""
    import random
    random.seed(42)  # Reproducible
    base = 100.0
    prices = [base]
    for _ in range(99):
        change = random.uniform(-5, 5)
        prices.append(prices[-1] + change)
    return prices


@pytest.fixture
def short_prices() -> List[float]:
    """Generate insufficient prices (too few for most indicators)."""
    return [100.0, 101.0, 102.0]


@pytest.fixture
def btc_like_prices() -> List[float]:
    """Generate BTC-like price data (high values, moderate volatility)."""
    import random
    random.seed(42)
    base = 90000.0
    prices = [base]
    for _ in range(199):
        change_pct = random.uniform(-0.02, 0.025)  # -2% to +2.5%
        prices.append(prices[-1] * (1 + change_pct))
    return prices


# ============================================
# API CLIENT FIXTURES
# ============================================

@pytest.fixture
def api_client():
    """Create a test client for the FastAPI app."""
    from main import app
    return TestClient(app)


@pytest.fixture
def client():
    """Alias for api_client - for backward compatibility with existing tests."""
    from main import app
    return TestClient(app)


@pytest.fixture
def async_api_client():
    """Create an async test client."""
    from main import app
    return httpx.AsyncClient(app=app, base_url="http://test")


# ============================================
# MOCK DATA FIXTURES
# ============================================

@pytest.fixture
def mock_price_response() -> Dict:
    """Mock response from price API."""
    return {
        "symbol": "BTCUSDT",
        "name": "Bitcoin",
        "price": 90234.56,
        "priceChangePercent": 2.45,
        "volume": 1234567890.12,
        "high_24h": 91000.00,
        "low_24h": 89000.00,
    }


@pytest.fixture
def mock_indicators_response() -> Dict:
    """Mock response from indicators API."""
    return {
        "symbol": "BTCUSDT",
        "rsi_14": 65.43,
        "macd": {
            "value": 123.45,
            "signal": 120.00,
            "histogram": 3.45,
        },
        "moving_averages": {
            "sma_20": 89500.00,
            "sma_50": 88000.00,
            "ema_12": 89800.00,
            "ema_26": 89200.00,
        },
        "bollinger_bands": {
            "upper": 91500.00,
            "middle": 89500.00,
            "lower": 87500.00,
        },
    }


# ============================================
# ASYNC EVENT LOOP
# ============================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
