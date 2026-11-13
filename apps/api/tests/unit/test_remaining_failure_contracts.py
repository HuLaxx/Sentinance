"""Offline checks for the final explicit failure boundaries.

These fixtures do not contact exchanges or run actual training.
"""

import httpx
import pytest
import requests

import deep_analysis
import download_data
import download_global_data
import train_model
from anomaly_alerts import AlertManager, AlertSeverity, AlertType


def disconnected_request():
    raise httpx.ConnectError("synthetic market outage")


@pytest.mark.parametrize(
    "module,kwargs",
    [
        (download_data, {"symbol": "BTCUSDT", "start_date": "2026-11-13"}),
        (train_model, {"symbol": "BTCUSDT", "days": 1}),
    ],
)
def test_historical_download_failure_never_returns_truncated_success(
    monkeypatch, module, kwargs
):
    monkeypatch.setattr(module.httpx, "get", lambda *_args, **_kwargs: disconnected_request())
    with pytest.raises(httpx.ConnectError, match="synthetic market outage"):
        (
            module.download_all_candles(**kwargs)
            if module is download_data
            else module.download_historical_data(**kwargs)
        )


def test_historical_download_programming_fault_propagates(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise RuntimeError("broken parser")

    monkeypatch.setattr(train_model.httpx, "get", unexpected)
    with pytest.raises(RuntimeError, match="broken parser"):
        train_model.download_historical_data(days=1)


def test_global_market_provider_exception_is_reported_without_data(monkeypatch):
    import yfinance

    def unavailable(_symbol):
        raise requests.ConnectionError("synthetic remote outage")

    monkeypatch.setattr(yfinance, "Ticker", unavailable)
    assert (
        download_global_data.download_stock_data(
            "^GSPC", {"name": "S&P", "country": "US", "type": "index"}
        )
        is None
    )


def test_global_market_programming_error_is_not_disguised_as_missing_data(monkeypatch):
    import yfinance

    def buggy(_symbol):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(yfinance, "Ticker", buggy)
    with pytest.raises(RuntimeError, match="implementation failure"):
        download_global_data.download_stock_data(
            "^GSPC", {"name": "S&P", "country": "US", "type": "index"}
        )


def test_deep_diagnostic_transport_fault_is_not_a_success():
    def unavailable():
        raise requests.ConnectionError("local API unavailable")

    ok, reason = deep_analysis.analyze_module("API", unavailable, {}, [])
    assert ok is False
    assert "unavailable" in reason


def test_deep_diagnostic_programming_defect_is_not_swallowed():
    def buggy():
        raise RuntimeError("test fixture has bug")

    with pytest.raises(RuntimeError, match="fixture has bug"):
        deep_analysis.analyze_module("API", buggy, {}, [])


@pytest.mark.asyncio
async def test_alert_subscriber_failure_is_isolated_from_healthy_subscribers():
    manager = AlertManager()
    delivered = []

    async def failed(_alert):
        raise RuntimeError("offline subscriber")

    async def healthy(alert):
        delivered.append(alert.id)

    manager.subscribe(failed)
    manager.subscribe(healthy)
    created = await manager.create_alert(
        alert_type=AlertType.PRICE_SPIKE,
        symbol="BTCUSDT",
        title="Synthetic alert",
        message="Test only",
        severity=AlertSeverity.INFO,
    )
    assert created is not None
    assert delivered == [created.id]
