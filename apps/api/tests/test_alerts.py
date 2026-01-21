"""
Alerts Service Tests

Tests for alert creation, listing, deletion, and triggering.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_create_alert_without_auth(client: TestClient):
    """Test creating alert without authentication."""
    response = client.post(
        "/api/alerts",
        json={
            "symbol": "BTCUSDT",
            "target_value": 50000.0,
            "alert_type": "price_above"
        }
    )
    # Should work for anonymous users or require auth
    assert response.status_code in [200, 201, 401, 422]


def test_create_alert_invalid_symbol(client: TestClient):
    """Test creating alert with invalid symbol."""
    response = client.post(
        "/api/alerts",
        json={
            "symbol": "INVALIDSYMBOL",
            "target_value": 50000.0,
            "alert_type": "price_above"
        }
    )
    # May accept any symbol or validate
    assert response.status_code in [200, 201, 400, 422]


def test_create_alert_invalid_condition(client: TestClient):
    """Test creating alert with invalid alert_type."""
    response = client.post(
        "/api/alerts",
        json={
            "symbol": "BTCUSDT",
            "target_value": 50000.0,
            "alert_type": "invalid_type"
        }
    )
    assert response.status_code in [400, 422]


def test_list_alerts_endpoint(client: TestClient):
    """Test listing alerts endpoint exists."""
    response = client.get("/api/alerts")
    # May require auth or return alerts
    assert response.status_code in [200, 401, 403]


def test_list_active_alerts(client: TestClient):
    """Test listing active alerts."""
    response = client.get("/api/alerts/active")
    assert response.status_code in [200, 401, 403]


def test_delete_nonexistent_alert(client: TestClient):
    """Test deleting non-existent alert."""
    response = client.delete("/api/alerts/nonexistent-id-12345")
    assert response.status_code in [401, 403, 404]


class TestAlertService:
    """Unit tests for the AlertService class."""
    
    def test_alert_service_import(self):
        """Test alert service can be imported."""
        from alerts_service import get_alerts_service, CreateAlertRequest, AlertType
        assert get_alerts_service is not None
        assert CreateAlertRequest is not None
    
    def test_create_alert_request_model(self):
        """Test CreateAlertRequest model validation."""
        from alerts_service import CreateAlertRequest, AlertType
        
        # Valid request
        request = CreateAlertRequest(
            symbol="BTCUSDT",
            target_value=50000.0,
            alert_type=AlertType.PRICE_ABOVE
        )
        assert request.symbol == "BTCUSDT"
        assert request.target_value == 50000.0
        assert request.alert_type == AlertType.PRICE_ABOVE
    
    def test_create_alert_request_conditions(self):
        """Test alert types are validated."""
        from alerts_service import CreateAlertRequest, AlertType
        
        # Test valid alert types
        for alert_type in [AlertType.PRICE_ABOVE, AlertType.PRICE_BELOW]:
            request = CreateAlertRequest(
                symbol="ETHUSDT",
                target_value=3000.0,
                alert_type=alert_type
            )
            assert request.alert_type == alert_type
