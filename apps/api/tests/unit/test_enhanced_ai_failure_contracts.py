"""Offline regression tests for enhanced AI source/fallback truth policies."""

from types import SimpleNamespace

import pytest
from google.api_core.exceptions import GoogleAPICallError
from qdrant_client.http.exceptions import ApiException

import anomaly_detection
import enhanced_ai
import indicators
import predictor
import rag_service


def gatherer():
    hist = [{"price": 100 + i, "volume": 20} for i in range(20)]
    return enhanced_ai.MarketDataGatherer(
        {"BTCUSDT": {"price": 120, "change_24h": 1}},
        {"BTCUSDT": hist},
    )


def model_bug(*_args, **_kwargs):
    raise RuntimeError("synthetic implementation defect")


def model_unavailable(*_args, **_kwargs):
    raise ValueError("insufficient model data")


def test_expected_model_input_failure_has_no_fake_forecast(monkeypatch):
    monkeypatch.setattr(predictor, "generate_prediction", model_unavailable)
    result = gatherer().get_predictions("BTCUSDT")
    assert result == "ML predictions unavailable."
    assert "$" not in result


def test_model_implementation_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(predictor, "generate_prediction", model_bug)
    with pytest.raises(RuntimeError, match="implementation defect"):
        gatherer().get_predictions("BTCUSDT")


def test_experimental_forecasts_are_labelled_not_calibrated_confidence(
    monkeypatch,
):
    def output(*_args, **_kwargs):
        return SimpleNamespace(
            direction="bullish",
            predicted_price=125.0,
            predicted_change_percent=4.2,
            confidence=0.99,
        )

    monkeypatch.setattr(predictor, "generate_prediction", output)
    result = gatherer().get_predictions("BTCUSDT")
    assert "experimental model output" in result
    assert "validated forecast accuracy" in result
    assert "Confidence: 99%" not in result


@pytest.mark.parametrize(
    ("module", "method", "function"),
    [
        (indicators, "get_indicators", "calculate_all_indicators"),
        (anomaly_detection, "get_anomaly_detection", "run_anomaly_detection"),
    ],
)
def test_expected_numeric_failure_returns_unavailable(
    monkeypatch, module, method, function
):
    monkeypatch.setattr(module, function, model_unavailable)
    result = getattr(gatherer(), method)("BTCUSDT")
    assert "unavailable" in result.lower()


@pytest.mark.parametrize(
    ("module", "method", "function"),
    [
        (indicators, "get_indicators", "calculate_all_indicators"),
        (anomaly_detection, "get_anomaly_detection", "run_anomaly_detection"),
    ],
)
def test_unexpected_numeric_bug_propagates(monkeypatch, module, method, function):
    monkeypatch.setattr(module, function, model_bug)
    with pytest.raises(RuntimeError, match="implementation defect"):
        getattr(gatherer(), method)("BTCUSDT")


@pytest.mark.asyncio
async def test_rag_provider_failure_does_not_fabricate_context(monkeypatch):
    async def unavailable(*_args, **_kwargs):
        raise ApiException("synthetic retrieval failure")

    monkeypatch.setattr(rag_service, "retrieve_context", unavailable)
    # Model data are intentionally absent; neither RAG nor news is invented.
    result = await enhanced_ai.MarketDataGatherer({}, {}).build_full_context("BTC")
    assert "no sample headlines are included" in result.lower()
    assert "synthetic retrieval failure" not in result


@pytest.mark.asyncio
async def test_unexpected_rag_bug_propagates(monkeypatch):
    async def broken(*_args, **_kwargs):
        raise RuntimeError("RAG code defect")

    monkeypatch.setattr(rag_service, "retrieve_context", broken)
    with pytest.raises(RuntimeError, match="RAG code defect"):
        await enhanced_ai.MarketDataGatherer({}, {}).build_full_context("BTC")


@pytest.mark.asyncio
async def test_gemini_provider_failure_can_use_groq_fallback(monkeypatch):
    async def minimal_context(self, _query):
        return "Only synthetic test context."

    monkeypatch.setattr(
        enhanced_ai.MarketDataGatherer, "build_full_context", minimal_context
    )
    monkeypatch.setattr(enhanced_ai, "GEMINI_AVAILABLE", False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = enhanced_ai.EnhancedAIChat()

    def gemini_fails(_prompt):
        raise GoogleAPICallError("synthetic provider 503")

    def groq_works(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Groq fallback"))]
        )

    client._gemini_initialized = True
    client.gemini_model = SimpleNamespace(generate_content=gemini_fails)
    client._groq_initialized = True
    client.groq_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=groq_works))
    )
    result = await client.chat("BTC?", {}, {})
    assert result["error"] is False
    assert result["provider"] == "groq"
    assert result["content"] == "Groq fallback"


@pytest.mark.asyncio
async def test_gemini_programming_error_does_not_trigger_groq_fallback(monkeypatch):
    async def minimal_context(self, _query):
        return "Only synthetic test context."

    monkeypatch.setattr(
        enhanced_ai.MarketDataGatherer, "build_full_context", minimal_context
    )
    monkeypatch.setattr(enhanced_ai, "GEMINI_AVAILABLE", False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = enhanced_ai.EnhancedAIChat()
    client._gemini_initialized = True
    client.gemini_model = SimpleNamespace(generate_content=model_bug)
    with pytest.raises(RuntimeError, match="implementation defect"):
        await client.chat("BTC?", {}, {})
