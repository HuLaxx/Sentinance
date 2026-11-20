"""Readiness requires actual Postgres + Redis; liveness is independent."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient


def test_dependency_outage_returns_503_and_liveness_stays_200(
    monkeypatch,
) -> None:
    from main import app

    probe = AsyncMock(
        return_value={
            "ready": False,
            "services": {"postgresql": "unavailable", "redis": "available"},
        }
    )
    monkeypatch.setattr("main.probe_readiness", probe)
    client = TestClient(app, raise_server_exceptions=False)
    for route in ("/health/ready", "/ready"):
        response = client.get(route)
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert "password" not in response.text
    assert client.get("/health/live").status_code == 200


def test_dependencies_healthy_return_200(monkeypatch) -> None:
    from main import app

    monkeypatch.setattr(
        "main.probe_readiness",
        AsyncMock(
            return_value={
                "ready": True,
                "services": {"postgresql": "available", "redis": "available"},
            }
        ),
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


async def test_probe_readiness_reports_real_uninitialized_redis(
    monkeypatch,
) -> None:
    from readiness import probe_readiness

    monkeypatch.setattr(
        "readiness.get_redis",
        lambda: (_ for _ in ()).throw(RuntimeError("Redis uninitialized")),
    )
    result = await probe_readiness()
    assert result["ready"] is False
    assert result["services"]["redis"] == "unavailable"
    assert result["services"]["postgresql"] in ("available", "unavailable")
