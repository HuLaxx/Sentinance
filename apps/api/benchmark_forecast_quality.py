"""Deterministic synthetic 3000-bar benchmark; not a live-feed SLA."""

import json
import math
from datetime import UTC, datetime, timedelta
from time import perf_counter

from forecast_model import evaluate_model
from forecast_quality import assess_report


def main() -> None:
    n = 3000
    start = datetime(2026, 1, 1, tzinfo=UTC)
    points = [
        (start + timedelta(minutes=i), 100.0 + i * 0.005 + math.sin(i / 9))
        for i in range(n)
    ]
    t0 = perf_counter()
    report = evaluate_model(points, train_end=1500, validation_end=2000)
    quality = assess_report(report)
    elapsed_ms = round((perf_counter() - t0) * 1000, 3)
    assert quality["test_count"] == 1000
    assert elapsed_ms < 15000, "Synthetic benchmark exceeded 15 s CI budget"
    print(
        json.dumps(
            {
                "fixture": "deterministic_3000_utc_minutes",
                "train_rows": 1500,
                "validation_rows": 500,
                "test_rows": 1000,
                "elapsed_ms": elapsed_ms,
                "model_underperforms_naive": quality[
                    "model_underperforms_naive_on_test"
                ],
                "error_summary": quality["metrics"],
                "limitation": "Single runner, synthetic fixture, no latency SLA.",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
