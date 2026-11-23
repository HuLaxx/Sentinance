"""Content-addressed local experiment registry, no executable model artifacts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forecast_model import report_bytes

MAX_REPORT_BYTES = 2_000_000


def read_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Trust only a separately pinned digest; never follow paths in a report."""
    with path.open("rb") as handle:
        content = handle.read(MAX_REPORT_BYTES + 1)
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("Forecast report exceeds byte budget")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Forecast report digest mismatch")
    report = json.loads(content)
    if not isinstance(report, dict) or report_bytes(report) != content:
        raise ValueError("Expected canonical forecast JSON object")
    return report


def register_report(root: Path, report: dict[str, Any]) -> str:
    """Write once by digest, using atomic publication on the local filesystem."""
    content = report_bytes(report)
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("Forecast report exceeds byte budget")
    digest = hashlib.sha256(content).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (digest + ".json")
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            try:
                os.link(temporary, destination)
            except FileExistsError:
                read_report(destination, digest)
        finally:
            temporary.unlink(missing_ok=True)
    return digest
