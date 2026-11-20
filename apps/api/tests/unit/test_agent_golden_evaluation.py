"""Deterministic agent tool and unsupported-query acceptance."""

from agent_golden_evaluation import REFUSED_PROMPTS, evaluate_golden_prompts


def test_golden_refusals_and_integrity():
    report = evaluate_golden_prompts()
    assert report["passed"] == len(REFUSED_PROMPTS) + 5 == 17
    assert report["failed"] == 0
    assert report["scope"] == "synthetic_only_not_a_llm_evaluation"
    assert report["llm_tokens"] == report["external_provider_calls"] == 0
    assert all(x["observed_ms"] >= 0 for x in report["cases"])
    assert report["cases"][0]["status"] == "allowed"
    assert all(x["status"] != "allowed" for x in report["cases"][1:])
