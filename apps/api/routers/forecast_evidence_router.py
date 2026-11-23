"""Opt-in, read-only forecast evidence from one operator-pinned artifact."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from evidence_agent import run_evidence_agent

router = APIRouter(tags=["Forecast evidence"])


@router.get("/api/v1/forecast-evidence")
def forecast_evidence() -> dict:
    path = os.getenv("SENTINANCE_FORECAST_REPORT_PATH")
    digest = os.getenv("SENTINANCE_FORECAST_REPORT_SHA256")
    if not path or not digest:
        raise HTTPException(status_code=503, detail="Forecast evidence not configured")
    try:
        return run_evidence_agent("compare forecasts", Path(path), digest)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        raise HTTPException(
            status_code=503, detail="Pinned forecast evidence unavailable"
        ) from exc
