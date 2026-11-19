"""Offline next-minute close baselines; not a trading signal or an LSTM."""

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

Point = tuple[datetime, float]


def evaluate_baselines(
    points: Sequence[Point], *, warmup: int, window: int
) -> dict[str, Any]:
    """Fixed one-step walk-forward: every prediction sees only earlier closes."""
    if window < 1 or warmup < window or len(points) <= warmup:
        raise ValueError("Require 1 <= window <= warmup < number of points")
    for i, (timestamp, close) in enumerate(points):
        if (
            timestamp.utcoffset() != timedelta(0)
            or timestamp.second
            or timestamp.microsecond
            or not math.isfinite(close)
            or close <= 0
            or close >= 1e18
        ):
            raise ValueError("Require UTC minute anchors and finite positive closes")
        if i and timestamp - points[i - 1][0] != timedelta(minutes=1):
            raise ValueError("Missing, duplicated or unordered minute bars")
    predictions: list[dict[str, Any]] = []
    for i in range(warmup, len(points)):
        timestamp, actual = points[i]
        last = points[i - 1][1]
        mean = math.fsum(p[1] for p in points[i - window : i]) / window
        predictions.append(
            {
                "target_utc": timestamp.astimezone(UTC).isoformat(),
                "history_through_utc": points[i - 1][0].astimezone(UTC).isoformat(),
                "actual": actual,
                "last_close": last,
                "moving_average": mean,
            }
        )
    metrics = {}
    for method in ("last_close", "moving_average"):
        errors = [float(row[method]) - float(row["actual"]) for row in predictions]
        count = len(errors)
        metrics[method] = {
            "count": count,
            "mae": math.fsum(abs(error) for error in errors) / count,
            "rmse": math.sqrt(math.fsum(error * error for error in errors) / count),
            "mean_error": math.fsum(errors) / count,
        }
    return {
        "evaluation": "one_step_observed_history_v1",
        "window": window,
        "warmup": warmup,
        "metrics": metrics,
        "predictions": predictions,
    }


def evaluate_snapshot(
    snapshot: Path,
    manifest_sha256: str,
    source: str,
    symbol: str,
    warmup: int,
    window: int,
) -> dict[str, Any]:
    """Use only a manifest whose SHA-256 was supplied independently."""
    import pyarrow.parquet as pq

    from price_lake import verify_snapshot

    actual_hash = hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    if actual_hash != manifest_sha256:
        raise ValueError("Snapshot manifest hash does not match the pinned hash")
    manifest = verify_snapshot(snapshot)
    table = pq.read_table(
        snapshot / "curated" / "minute_bars.parquet",
        columns=["source", "symbol", "minute_utc", "close"],
        filters=[("source", "=", source), ("symbol", "=", symbol)],
    )
    points = sorted(
        (row["minute_utc"], float(row["close"])) for row in table.to_pylist()
    )
    report = evaluate_baselines(points, warmup=warmup, window=window)
    report["snapshot_id"] = manifest["snapshot_id"]
    report["manifest_sha256"] = actual_hash
    report["source"] = source
    report["symbol"] = symbol
    report["limitation"] = (
        "Historical event-time replay; not proof of live arrival-time availability, "
        "real-world forecasting skill, trading profitability or model superiority."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    result = evaluate_snapshot(
        args.snapshot,
        args.manifest_sha256,
        args.source,
        args.symbol,
        args.warmup,
        args.window,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
