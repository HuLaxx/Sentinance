"""Auditable synthetic price forecast errors and dependence-aware uncertainty.

Post-hoc model assessment only; no data or hyperparameter fitting here.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from experiment_registry import read_report

METHODS = ("last_close", "moving_average", "ridge_ar1")


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("Forecast evidence must contain numeric predictions")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or result >= 1e18:
        raise ValueError("Forecast evidence contains invalid prices")
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, index)]


def _block_interval(values: list[float], *, iterations: int = 250) -> list[float]:
    """Percentile interval for paired MAE delta, sampling consecutive test blocks.

    Heuristic, not an inferential guarantee on a small or synthetic sample.
    """
    n = len(values)
    width = min(5, n)
    rng = random.Random(20261120)
    means = []
    for _ in range(iterations):
        chosen: list[float] = []
        while len(chosen) < n:
            start = rng.randrange(n - width + 1)
            chosen.extend(values[start : start + width])
        means.append(math.fsum(chosen[:n]) / n)
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def assess_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("version") != "ridge_log_return_ar1_v1":
        raise ValueError("Unsupported evaluated model version")
    rows = report.get("predictions")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 10_000:
        raise ValueError("Requires 2 to 10000 held-out observations")
    actual = [_finite(row["actual"]) for row in rows]
    errors: dict[str, list[float]] = {
        method: [_finite(row[method]) - observed for row, observed in zip(rows, actual)]
        for method in METHODS
    }

    def describe(values: list[float]) -> dict[str, float]:
        absolute = [abs(x) for x in values]
        return {
            "mae": math.fsum(absolute) / len(absolute),
            "rmse": math.sqrt(math.fsum(x * x for x in values) / len(values)),
            "median_absolute_error": _percentile(absolute, 0.5),
            "p95_absolute_error": _percentile(absolute, 0.95),
            "max_absolute_error": max(absolute),
        }

    metrics = {method: describe(errors[method]) for method in METHODS}
    model_delta = [
        abs(model) - abs(naive)
        for model, naive in zip(errors["ridge_ar1"], errors["last_close"])
    ]
    return {
        "evaluation": "paired_block_resampling_synthetic_v1",
        "test_count": len(rows),
        "metrics": metrics,
        "model_minus_naive_mae": math.fsum(model_delta) / len(model_delta),
        "model_minus_naive_mae_interval_heuristic": _block_interval(model_delta),
        "model_underperforms_naive_on_test": (
            metrics["ridge_ar1"]["mae"] > metrics["last_close"]["mae"]
        ),
        "methodology": {
            "bootstrap_seed": 20261120,
            "bootstrap_repetitions": 250,
            "bootstrap_block_length": min(5, len(rows)),
            "unit": "source price quote currency",
            "caveat": (
                "Synthetic historical event-time replay; this interval is a "
                "heuristic and cannot establish live forecast skill, calibrated "
                "uncertainty or investment profitability."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assessment = assess_report(read_report(args.report, args.sha256))
    content = (json.dumps(assessment, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(content)
    print(hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    main()
