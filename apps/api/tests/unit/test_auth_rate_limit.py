"""Actual FastAPI auth-boundary tests for fail-closed rate limiting."""

from hashlib import sha256
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import auth_rate_limit
from cache import RateLimitDecision

LOGIN = {"email": "missing@example.test", "password": "invalid-password"}
REGISTER = {"email": "fresh-rate@example.test", "password": "TestPassword123!"}
REFRESH = {"refresh_token": "invalid-refresh-token"}


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/auth/login", LOGIN),
        ("/api/auth/register", REGISTER),
        ("/api/auth/refresh", REFRESH),
    ],
)
def test_auth_endpoints_fail_closed_when_redis_is_unavailable(
    client: TestClient, monkeypatch, path, body
):
    lookup = AsyncMock(return_value=RateLimitDecision(False, 0, False))
    monkeypatch.setattr(auth_rate_limit, "evaluate_rate_limit", lookup)

    response = client.post(path, json=body)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.json()["detail"] == "Authentication temporarily unavailable"
    lookup.assert_awaited_once()


def test_login_is_rejected_with_429_after_atomic_limit(
    client: TestClient, monkeypatch
):
    lookup = AsyncMock(return_value=RateLimitDecision(False, 0, True))
    monkeypatch.setattr(auth_rate_limit, "evaluate_rate_limit", lookup)

    response = client.post("/api/auth/login", json=LOGIN)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json()["detail"] == "Too many authentication attempts"
    lookup.assert_awaited_once()


def test_allowed_rate_limit_does_not_bypass_actual_authentication(
    client: TestClient, monkeypatch
):
    lookup = AsyncMock(return_value=RateLimitDecision(True, 19, True))
    monkeypatch.setattr(auth_rate_limit, "evaluate_rate_limit", lookup)

    response = client.post("/api/auth/login", json=LOGIN)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"
    lookup.assert_awaited_once()


def test_auth_limit_uses_peer_address_not_spoofable_forwarded_header(
    client: TestClient, monkeypatch
):
    lookup = AsyncMock(return_value=RateLimitDecision(True, 19, True))
    monkeypatch.setattr(auth_rate_limit, "evaluate_rate_limit", lookup)

    for fake_ip in ("1.1.1.1", "2.2.2.2"):
        response = client.post(
            "/api/auth/login",
            json=LOGIN,
            headers={"X-Forwarded-For": fake_ip},
        )
        assert response.status_code == 401

    expected_key = sha256(b"/api/auth/login|testclient").hexdigest()
    assert lookup.await_count == 2
    for call in lookup.await_args_list:
        assert call.args == (expected_key,)
        assert call.kwargs == {"max_requests": 20, "window_seconds": 60}


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/auth/login", LOGIN),
        ("/api/auth/register", REGISTER),
        ("/api/auth/refresh", REFRESH),
    ],
)
def test_all_token_issuing_routes_call_rate_limit(
    client: TestClient, monkeypatch, path, body
):
    lookup = AsyncMock(return_value=RateLimitDecision(False, 0, True))
    monkeypatch.setattr(auth_rate_limit, "evaluate_rate_limit", lookup)
    response = client.post(path, json=body)
    assert response.status_code == 429
    lookup.assert_awaited_once()
