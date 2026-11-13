"""Regression tests for UTC alert timestamps and cross-offset filtering."""

from datetime import UTC, datetime, timedelta

import pytest

from anomaly_alerts import (
    AlertManager,
    AlertSeverity,
    AlertType,
    MarketAlert,
    _timestamp_utc,
)
from anomaly_detection import (
    detect_manipulation_pattern,
    detect_price_spike,
    detect_spread_anomaly,
    detect_volume_anomaly,
)


@pytest.mark.parametrize(
    "detector,args",
    [
        (detect_price_spike, ([100.0, 111.0],)),
        (detect_volume_anomaly, (1000.0, 100.0)),
        (detect_spread_anomaly, (100.0, 101.0)),
    ],
)
def test_anomaly_timestamps_include_utc_offset(detector, args):
    anomaly = detector(*args)
    assert anomaly is not None
    timestamp = datetime.fromisoformat(anomaly.timestamp)
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_heuristic_manipulation_alert_uses_aware_utc_time():
    prices = [100.0] * 9 + [80.0]
    volumes = [1.0] * 9 + [10.0]
    anomalies = detect_manipulation_pattern(prices, volumes)
    assert anomalies
    for anomaly in anomalies:
        assert datetime.fromisoformat(anomaly.timestamp).utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "value",
    [
        "2026-11-13T10:30:00",
        "2026-11-13T10:30:00+00:00",
        "2026-11-13T05:30:00-05:00",
        "2026-11-13T10:30:00Z",
    ],
)
def test_alert_timestamp_normalizes_legacy_and_explicit_timezones(value):
    expected = datetime(2026, 11, 13, 10, 30, tzinfo=UTC)
    assert _timestamp_utc(value) == expected


def test_recent_alerts_normalize_offsets_before_sorting_and_filtering():
    manager = AlertManager()
    now = datetime.now(UTC)
    early = now - timedelta(minutes=45)
    late = now - timedelta(minutes=15)
    expired = now - timedelta(days=2)

    def alert(key: str, date_string: str) -> MarketAlert:
        return MarketAlert(
            id=key,
            type=AlertType.PRICE_SPIKE,
            severity=AlertSeverity.WARNING,
            symbol="BTCUSDT",
            title=key,
            message="Synthetic fixture",
            data={},
            timestamp=date_string,
        )

    manager.alerts = [
        alert(
            "late-negative-offset", late.astimezone(timedelta_timezone(-5)).isoformat()
        ),
        alert("expired", expired.isoformat()),
        alert("legacy-utc", early.replace(tzinfo=None).isoformat()),
    ]
    result = manager.get_recent_alerts(hours=24)
    assert [entry.id for entry in result] == ["late-negative-offset", "legacy-utc"]


def timedelta_timezone(hours: int):
    """Produce a fixed offset for testing timestamp normalization."""
    from datetime import timezone

    return timezone(timedelta(hours=hours))


@pytest.mark.asyncio
async def test_created_alert_is_utc_aware_and_cooldown_is_active():
    manager = AlertManager()
    created = await manager.create_alert(
        AlertType.PRICE_SPIKE,
        "BTCUSDT",
        "Synthetic spike",
        "Fixture-only signal",
    )
    assert created is not None
    assert datetime.fromisoformat(created.timestamp).utcoffset() == timedelta(0)
    assert manager._check_cooldown(AlertType.PRICE_SPIKE, "BTCUSDT")
    repeated = await manager.create_alert(
        AlertType.PRICE_SPIKE,
        "BTCUSDT",
        "Repeated",
        "Should be rate limited",
    )
    assert repeated is None


def test_legacy_naive_utc_cooldown_does_not_raise_type_error():
    manager = AlertManager()
    key = f"{AlertType.PRICE_SPIKE.value}:BTCUSDT"
    manager.cooldowns[key] = (datetime.now(UTC) + timedelta(minutes=5)).replace(
        tzinfo=None
    )
    assert manager._check_cooldown(AlertType.PRICE_SPIKE, "BTCUSDT")


@pytest.mark.asyncio
async def test_broadcast_failure_does_not_stop_other_subscribers():
    manager = AlertManager()
    observed = []

    def broken(_alert):
        raise RuntimeError("subscriber unavailable")

    def healthy(alert):
        observed.append(alert.symbol)

    manager.subscribe(broken)
    manager.subscribe(healthy)
    created = await manager.create_alert(
        AlertType.VOLUME_SURGE,
        "ETHUSDT",
        "Volume",
        "Synthetic event",
    )
    assert created is not None
    assert observed == ["ETHUSDT"]
