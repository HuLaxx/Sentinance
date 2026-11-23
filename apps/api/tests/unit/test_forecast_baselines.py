"""Deterministic acceptance tests for strictly past-only price baselines."""

from datetime import UTC, datetime, timedelta

import pytest

from forecast_baselines import evaluate_baselines


def _points(values: list[float]) -> list[tuple[datetime, float]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [(start + timedelta(minutes=i), value) for i, value in enumerate(values)]


def test_constant_series_has_zero_error() -> None:
    result = evaluate_baselines(_points([10.0] * 6), warmup=3, window=2)
    for metric in result["metrics"].values():
        assert metric == {"count": 3, "mae": 0.0, "rmse": 0.0, "mean_error": 0.0}


def test_walk_forward_has_known_predictions_and_errors() -> None:
    result = evaluate_baselines(_points([10.0, 20.0, 30.0, 40.0]), warmup=2, window=2)
    assert [p["last_close"] for p in result["predictions"]] == [20.0, 30.0]
    assert [p["moving_average"] for p in result["predictions"]] == [15.0, 25.0]
    assert result["metrics"]["last_close"]["mae"] == 10.0
    assert result["metrics"]["moving_average"]["rmse"] == 15.0
    for p in result["predictions"]:
        assert p["history_through_utc"] < p["target_utc"]


def test_later_actual_cannot_change_earlier_forecast() -> None:
    a = evaluate_baselines(_points([10.0, 20.0, 30.0, 40.0]), warmup=2, window=2)
    b = evaluate_baselines(_points([10.0, 20.0, 999.0, 40.0]), warmup=2, window=2)
    for key in ("last_close", "moving_average"):
        assert a["predictions"][0][key] == b["predictions"][0][key]


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_reject_invalid_prices(value: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        evaluate_baselines(_points([10.0, 20.0, value]), warmup=2, window=2)


def test_reject_gap_duplicate_and_naive_timestamp() -> None:
    points = _points([10.0, 20.0, 30.0, 40.0])
    for bad in (points[:2] + points[3:], points[:2] + points[1:]):
        with pytest.raises(ValueError, match="minute bars"):
            evaluate_baselines(bad, warmup=2, window=2)
    points[0] = (points[0][0].replace(tzinfo=None), 10.0)
    with pytest.raises(ValueError, match="UTC minute"):
        evaluate_baselines(points, warmup=2, window=2)


def test_reject_insufficient_history() -> None:
    with pytest.raises(ValueError, match="warmup"):
        evaluate_baselines(_points([10.0, 20.0]), warmup=2, window=2)
