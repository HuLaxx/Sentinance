"""ASGI endpoint accepts no user-controlled report path or digest."""

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from experiment_registry import register_report
from forecast_model import evaluate_model
from routers.forecast_evidence_router import router


def fixture_report():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    points = [(start + timedelta(minutes=i), 100.0 + i) for i in range(24)]
    result = evaluate_model(points, train_end=12, validation_end=18)
    result.update({"symbol": "BTCUSDT", "manifest_sha256": "a" * 64})
    return result


def test_forecast_api_is_disabled_and_fail_closed(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.delenv("SENTINANCE_FORECAST_REPORT_PATH", raising=False)
    monkeypatch.delenv("SENTINANCE_FORECAST_REPORT_SHA256", raising=False)
    assert client.get("/api/v1/forecast-evidence").status_code == 503
    digest = register_report(tmp_path, fixture_report())
    path = tmp_path / (digest + ".json")
    monkeypatch.setenv("SENTINANCE_FORECAST_REPORT_PATH", str(path))
    monkeypatch.setenv("SENTINANCE_FORECAST_REPORT_SHA256", digest)
    response = client.get("/api/v1/forecast-evidence?path=/etc/passwd")
    assert response.status_code == 200
    assert response.json()["citation"] == "sha256:" + digest
    assert response.json()["live_market_verified"] is False
    path.write_text("tampered")
    assert client.get("/api/v1/forecast-evidence").status_code == 503


def test_main_api_registers_forecast_evidence_route():
    from main import app

    assert "/api/v1/forecast-evidence" in app.openapi()["paths"]
