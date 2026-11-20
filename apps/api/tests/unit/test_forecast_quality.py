"""Exercise forecast quality underperformance, dependence and invalid input."""
from datetime import UTC, datetime, timedelta

import pytest

from forecast_model import evaluate_model
from forecast_quality import assess_report


def _series(prices):
    start = datetime(2026, 11, 20, tzinfo=UTC)
    return [(start + timedelta(minutes=i), float(p)) for i, p in enumerate(prices)]


def test_assessment_reproducible_and_metrics_agree_with_forecast():
    report = evaluate_model(
        _series([100 + i + (i % 4) for i in range(48)]),
        train_end=24,
        validation_end=32,
    )
    quality = assess_report(report)
    assert quality == assess_report(report)
    assert quality["test_count"] == 16
    for name in ("last_close", "moving_average", "ridge_ar1"):
        assert quality["metrics"][name]["mae"] == pytest.approx(
            report["test_metrics"][name]["mae"]
        )
        assert quality["metrics"][name]["rmse"] == pytest.approx(
            report["test_metrics"][name]["rmse"]
        )
    lo, hi = quality["model_minus_naive_mae_interval_heuristic"]
    assert lo <= hi


def test_explicit_underperformance_flag_for_bad_model():
    rows = [
        {"actual": 10.0, "last_close": 10.0, "moving_average": 10.0,
         "ridge_ar1": 30.0}
        for _ in range(10)
    ]
    result = assess_report({"version": "ridge_log_return_ar1_v1", "predictions": rows})
    assert result["model_underperforms_naive_on_test"] is True
    assert result["model_minus_naive_mae"] == 20
    assert result["model_minus_naive_mae_interval_heuristic"] == [20, 20]


def test_fail_closed_on_corrupt_and_small_reports():
    with pytest.raises(ValueError):
        assess_report({"version": "unsupported", "predictions": []})
    with pytest.raises(ValueError):
        assess_report({"version": "ridge_log_return_ar1_v1", "predictions": []})
    rows = [
        {"actual": 10.0, "last_close": 10.0, "moving_average": 10.0,
         "ridge_ar1": float("nan")}
        for _ in range(3)
    ]
    with pytest.raises(ValueError, match="invalid prices"):
        assess_report({"version": "ridge_log_return_ar1_v1", "predictions": rows})
