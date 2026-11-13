"""Chat response truth boundary: evidence coverage is not probability.

All calls are offline stubs: no LLM, exchange, database or paid service.
"""

import httpx
import pytest

import enhanced_ai
import main


def request():
    return main.ChatRequest(message="BTC overview", history=[], use_agent=True)


@pytest.mark.asyncio
async def test_agent_evidence_coverage_is_not_presented_as_model_confidence(monkeypatch):
    async def agent_stub(_message):
        return {
            "analysis": "Only the available data can be discussed.",
            "confidence": 0.25,
            "plan": ["get_current_price"],
            "timestamp": "2026-11-14T00:00:00+00:00",
        }

    monkeypatch.setattr(main, "AGENT_AVAILABLE", True)
    monkeypatch.setattr(main, "run_agent", agent_stub)
    result = await main.chat(request())
    metadata = result["metadata"]
    assert metadata["confidence"] is None
    assert metadata["evidence_coverage"] == 0.25
    assert metadata["confidence_kind"] == "not_calibrated"
    assert metadata["source_type"] == "agent_interpretation"
    assert metadata["is_mock"] is False


@pytest.mark.asyncio
async def test_agent_programming_error_is_not_silently_replaced_with_mock(monkeypatch):
    async def broken(_message):
        raise RuntimeError("internal defect")

    monkeypatch.setattr(main, "AGENT_AVAILABLE", True)
    monkeypatch.setattr(main, "run_agent", broken)
    with pytest.raises(RuntimeError, match="internal defect"):
        await main.chat(request())


@pytest.mark.asyncio
async def test_agent_http_outage_can_use_an_explicitly_labelled_llm_fallback(
    monkeypatch,
):
    async def unavailable(_message):
        raise httpx.ConnectError("synthetic exchange outage")

    async def enhanced_stub(**_kwargs):
        return {
            "content": "A provider observation is unavailable.",
            "error": False,
            "model": "stubbed-model",
            "context_used": False,
        }

    monkeypatch.setattr(main, "AGENT_AVAILABLE", True)
    monkeypatch.setattr(main, "run_agent", unavailable)
    monkeypatch.setattr(enhanced_ai, "chat_with_ai", enhanced_stub)
    result = await main.chat(request())
    assert result["metadata"]["source_type"] == "llm_generated"
    assert result["metadata"]["confidence"] is None
    assert result["metadata"]["is_mock"] is False


@pytest.mark.asyncio
async def test_successful_llm_is_not_assigned_hardcoded_92_percent_confidence(
    monkeypatch,
):
    async def enhanced_stub(**_kwargs):
        return {
            "content": "Synthetic text from a test provider.",
            "error": False,
            "model": "synthetic-test",
        }

    monkeypatch.setattr(main, "AGENT_AVAILABLE", False)
    monkeypatch.setattr(enhanced_ai, "chat_with_ai", enhanced_stub)
    result = await main.chat(request())
    assert result["metadata"]["confidence"] is None
    assert result["metadata"]["confidence_kind"] == "not_calibrated"


@pytest.mark.asyncio
async def test_mock_response_overrides_any_fake_confidence_from_old_template(
    monkeypatch,
):
    async def unavailable(**_kwargs):
        return {"error": True}

    monkeypatch.setattr(main, "AGENT_AVAILABLE", False)
    monkeypatch.setattr(enhanced_ai, "chat_with_ai", unavailable)
    monkeypatch.setattr(
        main,
        "get_ai_response",
        lambda *_args: {
            "role": "assistant",
            "content": "illustrative text only",
            "metadata": {"model": "old-model", "confidence": 0.99},
        },
    )
    result = await main.chat(request())
    assert result["metadata"]["source_type"] == "synthetic"
    assert result["metadata"]["model"] == "mock"
    assert result["metadata"]["confidence"] is None
    assert result["metadata"]["is_mock"] is True


@pytest.mark.asyncio
async def test_unexpected_enhanced_llm_bug_propagates_instead_of_mocking(
    monkeypatch,
):
    async def buggy(**_kwargs):
        raise RuntimeError("unhandled response contract defect")

    monkeypatch.setattr(main, "AGENT_AVAILABLE", False)
    monkeypatch.setattr(enhanced_ai, "chat_with_ai", buggy)
    with pytest.raises(RuntimeError, match="response contract defect"):
        await main.chat(request())
