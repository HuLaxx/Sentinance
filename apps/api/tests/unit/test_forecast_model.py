"""Deterministic chronological evaluation, leakage and baseline comparisons."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from forecast_model import evaluate_model, report_bytes


def series(values):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [(start + timedelta(minutes=i), v) for i, v in enumerate(values)]


def test_constant_series_zero_error_and_same_target_set():
    result = evaluate_model(series([100.0] * 16), train_end=8, validation_end=12)
    assert result["split"]["test_rows"] == 4
    assert set(result["test_metrics"]) == {
        "last_close",
        "moving_average",
        "ridge_ar1",
    }
    for metric in result["test_metrics"].values():
        assert metric["count"] == 4
        assert metric["mae"] == pytest.approx(0, abs=1e-10)


def test_test_values_cannot_change_model_selection_or_earlier_predictions():
    values = [100.0 + i + (i % 3) for i in range(20)]
    a = evaluate_model(series(values), train_end=10, validation_end=15)
    values[15] += 30
    values[18] += 90
    b = evaluate_model(series(values), train_end=10, validation_end=15)
    assert a["model"] == b["model"]
    assert a["validation_candidates"] == b["validation_candidates"]
    for key in ("ridge_ar1", "last_close", "moving_average"):
        assert a["predictions"][0][key] == b["predictions"][0][key]
    assert a["predictions"][0]["actual"] != b["predictions"][0]["actual"]
    assert a["model"]["fit_through_utc"] < a["split"]["test_start_utc"]


def test_ridge_learns_synthetic_constant_log_return():
    points = series([100.0 * math.exp(i * 0.01) for i in range(24)])
    result = evaluate_model(points, train_end=12, validation_end=18)
    assert result["test_metrics"]["ridge_ar1"]["mae"] < 1e-8
    assert result["test_metrics"]["last_close"]["mae"] > 1
    assert report_bytes(result) == report_bytes(
        evaluate_model(points, train_end=12, validation_end=18)
    )


@pytest.mark.parametrize("train, validation", [(7, 12), (8, 9), (8, 15)])
def test_invalid_split_fails_closed(train, validation):
    with pytest.raises(ValueError, match="train rows"):
        evaluate_model(series([100.0] * 16), train_end=train, validation_end=validation)


def test_gap_is_rejected():
    points = series([100.0] * 17)
    del points[9]
    with pytest.raises(ValueError, match="minute bars"):
        evaluate_model(points, train_end=8, validation_end=12)
