"""Expanding-window chronological evaluation; no model promotion side effects."""

from collections.abc import Sequence
from typing import Any

from forecast_baselines import Point
from forecast_model import evaluate_model

MAX_FOLDS = 10


def rolling_evaluation(
    points: Sequence[Point], *, train_rows: int, validation_rows: int, test_rows: int
) -> dict[str, Any]:
    """Disjoint test blocks; earlier observations can enter later fold training."""
    if train_rows < 8 or validation_rows < 2 or test_rows < 2:
        raise ValueError("Insufficient fold size")
    first_test = train_rows + validation_rows
    count = (len(points) - first_test) // test_rows
    if not 1 <= count <= MAX_FOLDS:
        raise ValueError("Require 1 to 10 complete test folds")
    folds = []
    for i in range(count):
        start = first_test + i * test_rows
        report = evaluate_model(
            points[: start + test_rows],
            train_end=start - validation_rows,
            validation_end=start,
        )
        folds.append(report)
    return {
        "evaluation": "expanding_window_disjoint_test_v1",
        "fold_count": count,
        "unused_tail_rows": len(points) - first_test - count * test_rows,
        "folds": folds,
        "limitation": "Event-time rolling evaluation, not independent IID trials.",
    }
