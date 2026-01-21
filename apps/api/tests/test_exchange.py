import pytest
import respx
from httpx import Response
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange_connector import BinanceConnector, CRYPTO_SYMBOLS

@pytest.mark.asyncio
async def test_get_prices_success():
    """Test successful price fetching with mocked response."""
    connector = BinanceConnector(symbols=["BTCUSDT", "ETHUSDT"])
    
    # Mock Binance API response for 24h ticker
    mock_24h_btc = {
        "symbol": "BTCUSDT",
        "lastPrice": "50000.00",
        "priceChangePercent": "2.5",
        "volume": "1000.0",
        "highPrice": "51000.00",
        "lowPrice": "49000.00"
    }
    
    mock_24h_eth = {
        "symbol": "ETHUSDT",
        "lastPrice": "3000.00",
        "priceChangePercent": "-1.2",
        "volume": "5000.0",
        "highPrice": "3100.00",
        "lowPrice": "2900.00"
    }

    with respx.mock(base_url="https://api.binance.com") as respx_mock:
        respx_mock.get("/api/v3/ticker/24hr", params={"symbol": "BTCUSDT"}).mock(return_value=Response(200, json=mock_24h_btc))
        respx_mock.get("/api/v3/ticker/24hr", params={"symbol": "ETHUSDT"}).mock(return_value=Response(200, json=mock_24h_eth))
        
        prices = await connector.get_prices()
        
        assert isinstance(prices, list)
        assert len(prices) == 2
        
        btc_data = next((p for p in prices if p["symbol"] == "BTCUSDT"), None)
        assert btc_data is not None
        assert btc_data["price"] == 50000.00
        assert btc_data["priceChangePercent"] == 2.5
        assert btc_data["asset_type"] == "crypto"

@pytest.mark.asyncio
async def test_get_prices_failure():
    """Test graceful handling of API failure."""
    connector = BinanceConnector(symbols=["BTCUSDT"])
    
    with respx.mock(base_url="https://api.binance.com") as respx_mock:
        respx_mock.get("/api/v3/ticker/24hr").mock(return_value=Response(500))
        
        prices = await connector.get_prices()
        
        # Should return list with valid results only. If all fail, empty list.
        assert prices == []

def test_crypto_symbols_configured():
    """Test that default crypto symbols are configured correctly."""
    assert "BTCUSDT" in CRYPTO_SYMBOLS
    assert "ETHUSDT" in CRYPTO_SYMBOLS
    assert "SOLUSDT" in CRYPTO_SYMBOLS
    assert "XRPUSDT" in CRYPTO_SYMBOLS
    assert len(CRYPTO_SYMBOLS) == 4
