from fastapi.testclient import TestClient
import pytest

def test_health_check(client: TestClient):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data

def test_health_live(client: TestClient):
    """Test the liveness probe."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

def test_get_prices(client: TestClient):
    """Test fetching public price data."""
    response = client.get("/api/prices")
    assert response.status_code == 200
    data = response.json()
    assert "prices" in data
    assert isinstance(data["prices"], list)
    # Since we might rely on live data or cache, just check structure
    if len(data["prices"]) > 0:
        item = data["prices"][0]
        assert "symbol" in item
        assert "price" in item
        assert "change_24h" in item
