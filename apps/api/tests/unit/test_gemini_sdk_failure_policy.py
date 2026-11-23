"""Offline regression tests for Gemini provider failure boundaries and AI context.

These tests exercise synthetic SDK faults, never real Gemini or market data.
"""

from types import SimpleNamespace

import pytest
from google.api_core.exceptions import GoogleAPICallError

import gemini_client
from enhanced_ai import MarketDataGatherer


def client_with_model(generate):
    client = gemini_client.GeminiClient(api_key=None)
    client.model = SimpleNamespace(generate_content=generate)
    client._initialized = True
    return client


def unavailable(*_args, **_kwargs):
    raise GoogleAPICallError("synthetic quota exhaustion")


def code_bug(*_args, **_kwargs):
    raise RuntimeError("synthetic implementation defect")


@pytest.mark.asyncio
async def test_chat_provider_error_is_safe_and_explicit():
    client = client_with_model(unavailable)
    answer = await client.chat("Describe BTC")
    assert answer["error"] is True
    assert "unavailable" in answer["content"]
    assert "quota" not in answer["content"]


@pytest.mark.asyncio
async def test_chat_does_not_disguise_programmer_faults():
    client = client_with_model(code_bug)
    with pytest.raises(RuntimeError, match="implementation defect"):
        await client.chat("BTC")


@pytest.mark.asyncio
async def test_chat_malformed_response_is_not_reported_as_provider_outage():
    client = client_with_model(lambda *_args: {})
    with pytest.raises(AttributeError):
        await client.chat("BTC")


@pytest.mark.asyncio
async def test_prompt_validation_faults_are_not_masked_as_provider_outage():
    client = client_with_model(lambda *_args: None)
    with pytest.raises(TypeError):
        await client.chat("BTC", context={"prices": {"BTCUSDT": {"price": None}}})


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["embed_text", "embed_query"])
async def test_embedding_provider_failure_returns_no_fake_vector(monkeypatch, method):
    monkeypatch.setattr(gemini_client.genai, "embed_content", unavailable)
    monkeypatch.setattr(gemini_client, "GEMINI_AVAILABLE", True)
    client = gemini_client.GeminiClient(api_key=None)
    client.api_key = "synthetic-test-key"
    assert await getattr(client, method)("BTC") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["embed_text", "embed_query"])
async def test_embedding_programmer_fault_propagates(monkeypatch, method):
    monkeypatch.setattr(gemini_client.genai, "embed_content", code_bug)
    monkeypatch.setattr(gemini_client, "GEMINI_AVAILABLE", True)
    client = gemini_client.GeminiClient(api_key=None)
    client.api_key = "synthetic-test-key"
    with pytest.raises(RuntimeError, match="implementation defect"):
        await getattr(client, method)("BTC")


@pytest.mark.asyncio
async def test_missing_embedding_payload_is_not_hidden(monkeypatch):
    monkeypatch.setattr(gemini_client.genai, "embed_content", lambda **_kwargs: {})
    monkeypatch.setattr(gemini_client, "GEMINI_AVAILABLE", True)
    client = gemini_client.GeminiClient(api_key=None)
    client.api_key = "synthetic-test-key"
    with pytest.raises(KeyError, match="embedding"):
        await client.embed_query("BTC")


def test_client_init_expected_invalid_configuration_does_not_mark_ready(monkeypatch):
    def invalid_config(*_args, **_kwargs):
        raise ValueError("invalid API key")

    monkeypatch.setattr(gemini_client.genai, "configure", invalid_config)
    monkeypatch.setattr(gemini_client, "GEMINI_AVAILABLE", True)
    client = gemini_client.GeminiClient(api_key="synthetic-key")
    assert client._initialized is False


def test_client_init_programming_fault_is_visible(monkeypatch):
    monkeypatch.setattr(gemini_client.genai, "configure", code_bug)
    monkeypatch.setattr(gemini_client, "GEMINI_AVAILABLE", True)
    with pytest.raises(RuntimeError, match="implementation defect"):
        gemini_client.GeminiClient(api_key="synthetic-key")


@pytest.mark.asyncio
async def test_ai_context_rejects_static_headlines_as_current_news():
    gatherer = MarketDataGatherer({"BTCUSDT": {"price": 100, "change_24h": 1}}, {})
    news = gatherer.get_news_context()
    assert "live news has not been verified" in news
    assert "Federal Reserve" not in news
    assert "record inflows" not in news
    assert "source and freshness unverified" in gatherer.get_price_context()
