"""Chronological ridge AR(1) benchmark on log returns, not a trading signal."""

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from forecast_baselines import Point, evaluate_baselines, load_snapshot_points

ALPHAS = (0.0, 1e-6, 1e-4, 1e-2)
VERSION = "ridge_log_return_ar1_v1"


def fit_model(
    pairs: Sequence[tuple[float, float]], alpha: float
) -> tuple[float, float]:
    """Fit intercept plus one lagged log return; ridge on slope only."""
    n = len(pairs)
    if n < 2 or alpha not in ALPHAS:
        raise ValueError("Invalid model fitting configuration")
    mx = math.fsum(x for x, _ in pairs) / n
    my = math.fsum(y for _, y in pairs) / n
    variance = math.fsum((x - mx) ** 2 for x, _ in pairs) / n
    covariance = math.fsum((x - mx) * (y - my) for x, y in pairs) / n
    slope = covariance / (variance + alpha) if variance + alpha else 0.0
    return my - slope * mx, slope


def _forecast(logs: Sequence[float], i: int, intercept: float, slope: float) -> float:
    predicted = logs[i - 1] + intercept + slope * (logs[i - 1] - logs[i - 2])
    if not -740 < predicted < math.log(1e18):
        raise ValueError("Model forecast outside supported finite price range")
    return math.exp(predicted)


def _metrics(errors: Sequence[float]) -> dict[str, float | int]:
    n = len(errors)
    return {
        "count": n,
        "mae": math.fsum(abs(x) for x in errors) / n,
        "rmse": math.sqrt(math.fsum(x * x for x in errors) / n),
        "mean_error": math.fsum(errors) / n,
    }


def evaluate_model(
    points: Sequence[Point], *, train_end: int, validation_end: int
) -> dict[str, Any]:
    """Fixed contiguous splits; actual prior close observed before each target.

    Train targets [2, train_end), validation [train_end, validation_end),
    test [validation_end, len(points)). Select alpha on validation only,
    then refit once on pre-test rows. No test-target fitting or tuning.
    """
    if not 8 <= train_end <= validation_end - 2 <= len(points) - 4:
        raise ValueError("Require >=8 train rows and >=2 validation and test rows")
    # Existing baseline validator rejects gaps and invalid/naive timestamps.
    baseline = evaluate_baselines(points, warmup=validation_end, window=5)
    logs = [math.log(value) for _, value in points]

    def pairs_until(stop: int) -> list[tuple[float, float]]:
        return [
            (logs[i - 1] - logs[i - 2], logs[i] - logs[i - 1]) for i in range(2, stop)
        ]

    candidates: list[dict[str, float]] = []
    for alpha in ALPHAS:
        intercept, slope = fit_model(pairs_until(train_end), alpha)
        errors = [
            _forecast(logs, i, intercept, slope) - points[i][1]
            for i in range(train_end, validation_end)
        ]
        candidates.append(
            {
                "alpha": alpha,
                "validation_mae": math.fsum(abs(x) for x in errors) / len(errors),
            }
        )
    winner = min(candidates, key=lambda row: (row["validation_mae"], -row["alpha"]))
    selected = winner["alpha"]
    intercept, slope = fit_model(pairs_until(validation_end), selected)
    predictions: list[dict[str, Any]] = []
    errors = []
    for i, row in enumerate(baseline["predictions"], start=validation_end):
        estimate = _forecast(logs, i, intercept, slope)
        errors.append(estimate - points[i][1])
        predictions.append({**row, "ridge_ar1": estimate})
    metrics = {**baseline["metrics"], "ridge_ar1": _metrics(errors)}
    return {
        "version": VERSION,
        "split": {
            "train_rows": train_end,
            "validation_rows": validation_end - train_end,
            "test_rows": len(points) - validation_end,
            "train_through_utc": points[train_end - 1][0].isoformat(),
            "validation_through_utc": points[validation_end - 1][0].isoformat(),
            "test_start_utc": points[validation_end][0].isoformat(),
            "test_end_utc": points[-1][0].isoformat(),
        },
        "validation_candidates": candidates,
        "model": {
            "alpha": selected,
            "intercept": intercept,
            "slope": slope,
            "fit_through_utc": points[validation_end - 1][0].isoformat(),
        },
        "test_metrics": metrics,
        "predictions": predictions,
        "limitation": (
            "Event-time historical one-step replay with observed previous closes; "
            "not live arrival-time or profitability evidence."
        ),
    }


def evaluate_model_snapshot(
    snapshot: Path,
    manifest_sha256: str,
    source: str,
    symbol: str,
    *,
    train_end: int,
    validation_end: int,
) -> dict[str, Any]:
    points, provenance = load_snapshot_points(snapshot, manifest_sha256, source, symbol)
    report = evaluate_model(points, train_end=train_end, validation_end=validation_end)
    report.update(provenance)
    return report


def report_bytes(report: dict[str, Any]) -> bytes:
    """Deterministic artifact encoding; digest identifies content, not authorship."""
    return (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--validation-end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_model_snapshot(
        args.snapshot,
        args.manifest_sha256,
        args.source,
        args.symbol,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    content = report_bytes(report)
    # Exclusive creation protects prior evaluation evidence.
    with args.output.open("xb") as file:
        file.write(content)
    print(hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    main()
