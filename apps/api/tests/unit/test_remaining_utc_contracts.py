"""Regression tests for UTC-aware metadata in previously naive code paths.

These tests use synthetic timestamps and mocks; they do not verify a live feed,
model accuracy, external vector storage, or end-to-end production service.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from enhanced_ai import MarketDataGatherer
from llm_wrapper import EnhancedLLMWrapper, IndexContext
from rag_service import RAGService
from streaming_llm import TokenUsage


def assert_utc(value: str) -> None:
    instant = datetime.fromisoformat(value)
    assert instant.tzinfo is not None
    assert instant.utcoffset() == timedelta(0)


def test_token_usage_records_explicit_utc():
    usage = TokenUsage(input_tokens=10, output_tokens=3)
    assert_utc(usage.timestamp)
    assert_utc(usage.to_dict()["timestamp"])


def test_historical_index_last_updated_is_explicit_utc():
    context = IndexContext(symbol="BTCUSDT")
    context.update_accuracy()
    assert context.last_updated is not None
    assert_utc(context.last_updated)


@pytest.mark.asyncio
async def test_context_gatherer_emits_utc_analysis_time(monkeypatch):
    import rag_service

    monkeypatch.setattr(
        rag_service, "retrieve_context", AsyncMock(return_value=""), raising=False
    )
    gatherer = MarketDataGatherer({}, {})
    for name in (
        "get_price_context",
        "get_predictions",
        "get_indicators",
        "get_anomaly_detection",
        "get_news_context",
    ):
        monkeypatch.setattr(gatherer, name, Mock(return_value="fixture"))

    result = await gatherer.build_full_context("Analyze BTC")
    stamp = next(
        line.removeprefix("ANALYSIS TIMESTAMP: ")
        for line in result.splitlines()
        if line.startswith("ANALYSIS TIMESTAMP: ")
    )
    assert_utc(stamp)


@pytest.mark.asyncio
async def test_rag_news_normalizes_naive_offset_and_bad_dates():
    service = RAGService.__new__(RAGService)
    service._connected = True
    service.gemini = SimpleNamespace(embed_text=AsyncMock(return_value=[0.1] * 768))
    upsert = Mock()
    service.client = SimpleNamespace(upsert=upsert)

    await service.add_news(
        [
            {
                "title": "naive",
                "summary": "synthetic",
                "published_at": "2026-11-13T18:00:00",
            },
            {
                "title": "offset",
                "summary": "synthetic",
                "published_at": "2026-11-13T13:00:00-05:00",
            },
            {"title": "invalid", "published_at": "not-a-time"},
        ]
    )

    points = upsert.call_args.kwargs["points"]
    assert len(points) == 3
    timestamps = {point.payload["title"]: point.payload["timestamp"] for point in points}
    for stamp in timestamps.values():
        assert_utc(stamp)
    expected = datetime(2026, 11, 13, 18, 0, tzinfo=UTC).isoformat()
    assert timestamps["naive"] == expected
    assert timestamps["offset"] == expected


@pytest.mark.asyncio
async def test_historical_prediction_id_uses_its_stored_utc_timestamp():
    wrapper = EnhancedLLMWrapper.__new__(EnhancedLLMWrapper)
    memory = SimpleNamespace(
        retrieve_similar=AsyncMock(return_value=[]),
        get_index_summary=Mock(return_value={}),
        store_prediction=AsyncMock(),
    )
    wrapper.memory = memory
    wrapper.gemini_model = True
    wrapper._call_llm = AsyncMock(return_value="fixture")
    wrapper._extract_prediction = Mock(return_value={"confidence": 0.0})
    wrapper._format_historical_context = Mock(return_value="none")

    result = await wrapper.generate_with_history("BTCUSDT", "fixture query", {})
    prediction = memory.store_prediction.await_args.args[0]
    assert_utc(prediction.timestamp)
    expected = hashlib.md5(
        f"BTCUSDT-fixture query-{prediction.timestamp}".encode()
    ).hexdigest()[:12]
    assert result["prediction_id"] == prediction.id == expected


@pytest.mark.asyncio
async def test_chat_agent_fallback_timestamp_is_explicit_utc(monkeypatch):
    import main

    monkeypatch.setattr(main, "AGENT_AVAILABLE", True)
    monkeypatch.setattr(
        main, "run_agent", AsyncMock(return_value={"analysis": "fixture"}), raising=False
    )
    result = await main.chat(main.ChatRequest(message="Explain BTC", use_agent=True))
    assert_utc(result["metadata"]["timestamp"])


@pytest.mark.asyncio
async def test_chat_enhanced_fallback_timestamp_is_explicit_utc(monkeypatch):
    import enhanced_ai
    import main

    monkeypatch.setattr(main, "AGENT_AVAILABLE", False)
    monkeypatch.setattr(
        enhanced_ai,
        "chat_with_ai",
        AsyncMock(return_value={"error": False, "content": "fixture"}),
        raising=False,
    )
    result = await main.chat(main.ChatRequest(message="Explain BTC", use_agent=False))
    assert_utc(result["metadata"]["timestamp"])
