"""Offline, constrained LangGraph forecast-evidence workflow.

No LLM, external market network, retrieval of arbitrary files, or trade tools.
User and artifact text are not executable instructions.
"""

import argparse
import json
import math
import re
from pathlib import Path
from time import monotonic
from typing import Any, TypedDict

from experiment_registry import read_report

METHODS = ("last_close", "moving_average", "ridge_ar1")
SYMBOL = re.compile(r"[A-Z0-9]{2,20}")
DIGEST = re.compile(r"[0-9a-f]{64}")
MAX_ROWS = 10_000


class EvidenceState(TypedDict):
    query: str
    report: dict[str, Any]
    digest: str
    result: dict[str, Any]
    trace: list[dict[str, Any]]


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid numeric evidence")
    if not math.isfinite(value) or abs(value) >= 1e36:
        raise ValueError("Invalid numeric evidence")
    return float(value)


def _safe_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Derive metrics from rows rather than echoing arbitrary artifact prose."""
    if report.get("version") != "ridge_log_return_ar1_v1":
        raise ValueError("Unsupported forecast evidence version")
    symbol = report.get("symbol")
    manifest = report.get("manifest_sha256")
    if not isinstance(symbol, str) or SYMBOL.fullmatch(symbol) is None:
        raise ValueError("Invalid symbol")
    if not isinstance(manifest, str) or DIGEST.fullmatch(manifest) is None:
        raise ValueError("Invalid dataset manifest pin")
    rows = report.get("predictions")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("Forecast row budget exceeded")
    metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        errors = [_number(row[method]) - _number(row["actual"]) for row in rows]
        metrics[method] = {
            "mae": math.fsum(abs(x) for x in errors) / len(errors),
            "rmse": math.sqrt(math.fsum(x * x for x in errors) / len(errors)),
        }
    # Deliberately do not treat an untrusted report source label as live proof.
    return {"symbol": symbol, "manifest_sha256": manifest, "metrics": metrics}


def run_evidence_agent(query: str, path: Path, digest: str) -> dict[str, Any]:
    """Three fixed nodes, one bounded read, zero LLM tokens and provider calls."""
    from langgraph.graph import END, StateGraph

    if not isinstance(query, str) or len(query) > 256:
        raise ValueError("Query exceeds character budget")
    start = monotonic()

    def retrieve(state: EvidenceState) -> dict:
        t = monotonic()
        if state["query"] != "compare forecasts":
            return {
                "report": {},
                "trace": [{"step": "retrieve", "status": "not_requested"}],
            }
        report = read_report(path, state["digest"])
        return {
            "report": report,
            "trace": [
                {
                    "step": "retrieve",
                    "status": "verified_digest",
                    "latency_ms": round((monotonic() - t) * 1000, 3),
                }
            ],
        }

    def analyze(state: EvidenceState) -> dict:
        t = monotonic()
        if not state["report"]:
            result: dict[str, Any] = {
                "status": "unsupported_query",
                "message": "Only compare forecasts is supported; no live quotes.",
            }
        else:
            result = {
                "status": "historical_evaluation",
                "statistical_results": _safe_summary(state["report"]),
                "citation": "sha256:" + state["digest"],
                "interpretation_mode": "deterministic_no_llm",
                "live_market_verified": False,
            }
        return {
            "result": result,
            "trace": state["trace"]
            + [
                {
                    "step": "analyze",
                    "status": result["status"],
                    "latency_ms": round((monotonic() - t) * 1000, 3),
                }
            ],
        }

    def finalize(state: EvidenceState) -> dict:
        return {
            "result": {
                **state["result"],
                "limitation": "Historical fixture or provider-labeled evidence only.",
                "llm_tokens": 0,
                "external_calls": 0,
                "latency_ms": round((monotonic() - start) * 1000, 3),
            },
            "trace": state["trace"] + [{"step": "finalize", "status": "complete"}],
        }

    graph = StateGraph(EvidenceState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("analyze", analyze)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "finalize")
    graph.add_edge("finalize", END)
    result = graph.compile().invoke(
        {"query": query, "digest": digest, "report": {}, "result": {}, "trace": []},
        config={"recursion_limit": 8},
    )
    return {**result["result"], "trace": result["trace"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--query", default="compare forecasts")
    args = parser.parse_args()
    result = run_evidence_agent(args.query, args.report, args.sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
