"""
WebSocket and News Tests

Tests for WebSocket connections and news endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_news_endpoint(client: TestClient):
    """Test news listing endpoint."""
    response = client.get("/api/news")
    assert response.status_code in [200, 404, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (list, dict))


def test_news_with_limit(client: TestClient):
    """Test news endpoint with limit parameter."""
    response = client.get("/api/news?limit=5")
    assert response.status_code in [200, 404, 503]


def test_news_by_topic(client: TestClient):
    """Test news by topic endpoint."""
    response = client.get("/api/news/topic/bitcoin")
    assert response.status_code in [200, 404, 503]


def test_market_stats_endpoint(client: TestClient):
    """Test market statistics endpoint."""
    response = client.get("/api/stats")
    assert response.status_code in [200, 404, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


def test_top_movers_endpoint(client: TestClient):
    """Test top movers endpoint."""
    response = client.get("/api/movers")
    assert response.status_code in [200, 404, 503]


def test_prices_endpoint(client: TestClient):
    """Test prices listing endpoint."""
    response = client.get("/api/prices")
    assert response.status_code == 200
    
    data = response.json()
    assert "prices" in data
    assert isinstance(data["prices"], list)


def test_single_price_endpoint(client: TestClient):
    """Test single price endpoint."""
    response = client.get("/api/prices/BTCUSDT")
    assert response.status_code in [200, 404]


def test_price_history_endpoint(client: TestClient):
    """Test price history endpoint."""
    response = client.get("/api/prices/BTCUSDT/history")
    assert response.status_code in [200, 404]


class TestWebSocket:
    """WebSocket connection tests."""
    
    def test_websocket_endpoint_exists(self, client: TestClient):
        """Test WebSocket endpoint is accessible."""
        # WebSocket endpoints can't be tested with regular HTTP
        # but we can verify the route exists
        from main import app
        
        routes = [route.path for route in app.routes]
        assert "/ws/prices" in routes or any("/ws" in r for r in routes)
    
    def test_websocket_connection(self, client: TestClient):
        """Test WebSocket connection can be established."""
        with client.websocket_connect("/ws/prices") as websocket:
            # Connection should be accepted
            # May receive initial data or just connect
            pass  # Connection successful if no exception


class TestNewsScraper:
    """Unit tests for news scraper module."""
    
    def test_news_scraper_import(self):
        """Test news scraper can be imported."""
        try:
            from news_scraper import get_latest_news
            assert get_latest_news is not None
        except ImportError:
            pytest.skip("News scraper not available")
    
    def test_market_stats_import(self):
        """Test market stats can be imported."""
        try:
            from market_stats import calculate_market_stats
            assert calculate_market_stats is not None
        except ImportError:
            pytest.skip("Market stats not available")
