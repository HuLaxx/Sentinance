"""Offline evidence-only checks for historical LLM / Qdrant memory semantics."""

from types import SimpleNamespace

import pytest
from google.api_core.exceptions import GoogleAPICallError
from qdrant_client.http.exceptions import ApiException

from llm_wrapper import (
    EnhancedLLMWrapper,
    HistoricalPrediction,
    HistoricalRAGMemory,
    IndexContext,
)


def prediction():
    return HistoricalPrediction(
        id="c9dd5cce-d146-4e2d-a504-07deabf86da1",
        symbol="BTCUSDT",
        query="BTC",
        prediction="No validated forecast",
        predicted_price=110.0,
        predicted_direction="bullish",
        confidence=None,
        timestamp="2026-11-14T00:00:00+00:00",
    )


def memory_with_client(client):
    memory = HistoricalRAGMemory.__new__(HistoricalRAGMemory)
    memory.qdrant = client
    memory.gemini = None
    memory.index_contexts = {}
    return memory


def test_unverified_summary_does_not_fabricate_fifty_percent_accuracy():
    memory = memory_with_client(None)
    summary = memory.get_index_summary("BTCUSDT")
    assert summary["verified_predictions"] == 0
    assert summary["avg_accuracy"] is None


def test_missing_price_does_not_turn_into_accuracy_score():
    p = prediction()
    p.predicted_price = 0
    p.actual_price = 100
    assert p.calculate_accuracy() is None
    assert p.accuracy_score is None


@pytest.mark.asyncio
async def test_failed_durable_write_does_not_append_in_memory_prediction(monkeypatch):
    def failed_upsert(**_kwargs):
        raise ApiException("synthetic Qdrant outage")

    memory = memory_with_client(SimpleNamespace(upsert=failed_upsert))

    async def embedding(_text):
        return [0.1, 0.2]

    monkeypatch.setattr(memory, "_embed_text", embedding)
    with pytest.raises(ApiException, match="outage"):
        await memory.store_prediction(prediction())
    assert memory.index_contexts == {}


@pytest.mark.asyncio
async def test_failed_outcome_write_leaves_prediction_unverified():
    def failed_write(**_kwargs):
        raise ApiException("synthetic storage outage")

    memory = memory_with_client(SimpleNamespace(set_payload=failed_write))
    p = prediction()
    context = IndexContext(symbol="BTCUSDT", predictions=[p])
    memory.index_contexts = {"BTCUSDT": context}
    with pytest.raises(ApiException, match="outage"):
        await memory.update_with_actual(p.id, 120.0, "bullish")
    assert p.actual_price is None
    assert p.verified_at is None
    assert p.accuracy_score is None
    assert context.avg_accuracy is None


@pytest.mark.asyncio
async def test_successful_outcome_write_updates_memory_after_storage():
    writes = []

    def successful_write(**kwargs):
        writes.append(kwargs)

    memory = memory_with_client(SimpleNamespace(set_payload=successful_write))
    p = prediction()
    context = IndexContext(symbol="BTCUSDT", predictions=[p])
    memory.index_contexts = {"BTCUSDT": context}
    await memory.update_with_actual(p.id, 120.0, "bullish")
    assert len(writes) == 1
    assert writes[0]["payload"]["accuracy_score"] is not None
    assert p.actual_price == 120.0
    assert context.avg_accuracy is not None


def test_llm_confidence_without_explicit_evidence_is_unknown():
    wrapper = EnhancedLLMWrapper.__new__(EnhancedLLMWrapper)
    extracted = wrapper._extract_prediction("BTC is uncertain", "BTCUSDT")
    assert extracted["confidence"] is None
    assert wrapper._extract_prediction("200% confidence", "BTCUSDT")["confidence"] is None


@pytest.mark.asyncio
async def test_outcome_verification_requires_actual_direction():
    wrapper = EnhancedLLMWrapper.__new__(EnhancedLLMWrapper)
    with pytest.raises(ValueError, match="verified direction"):
        await wrapper.verify_prediction("test-id", 100.0)


@pytest.mark.asyncio
async def test_actual_provider_tracking_reports_successful_groq_fallback():
    wrapper = EnhancedLLMWrapper.__new__(EnhancedLLMWrapper)

    def gemini_outage(_prompt):
        raise GoogleAPICallError("synthetic Gemini 503")

    def groq_success(**_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback"))]
        )

    wrapper.gemini_model = SimpleNamespace(generate_content=gemini_outage)
    wrapper.groq_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=groq_success))
    )
    wrapper._last_provider = None
    result = await wrapper._call_llm("Explain BTC")
    assert result == "fallback"
    assert wrapper._last_provider == "groq"


@pytest.mark.asyncio
async def test_without_verified_history_prompt_says_unverified_not_50_percent(
    monkeypatch,
):
    prompts = []

    class FakeMemory:
        async def retrieve_similar(self, *_args, **_kwargs):
            return []

        def get_index_summary(self, _symbol):
            return {"avg_accuracy": None, "verified_predictions": 0}

        async def store_prediction(self, _prediction):
            return None

    wrapper = EnhancedLLMWrapper.__new__(EnhancedLLMWrapper)
    wrapper.memory = FakeMemory()
    wrapper.gemini_model = None
    wrapper.groq_client = None
    wrapper._last_provider = None

    async def generate(prompt):
        prompts.append(prompt)
        return "Observations unavailable"

    monkeypatch.setattr(wrapper, "_call_llm", generate)
    result = await wrapper.generate_with_history("BTCUSDT", "BTC", {})
    assert "Unverified" in prompts[0]
    assert "50% across 0" not in prompts[0]
    assert result["confidence"] is None
    assert result["historical_accuracy"] is None
