"""Offline tests for typed LLM failures, stream completeness and retry semantics."""

import pytest
from google.api_core.exceptions import GoogleAPICallError

import streaming_llm
from structured_output import NewsDigest, StructuredOutputExtractor


@pytest.mark.asyncio
async def test_structured_provider_outage_retries_without_fabricated_result():
    extractor = StructuredOutputExtractor()
    extractor.retry_count = 2
    calls = []

    async def provider(prompt):
        calls.append(prompt)
        raise GoogleAPICallError("synthetic 503")

    assert await extractor.extract_with_retry(provider, NewsDigest, "summarise") is None
    assert len(calls) == 2
    assert all("Previous response was not valid JSON" not in p for p in calls)


@pytest.mark.asyncio
async def test_structured_programmer_fault_is_not_swallowed_or_retried():
    extractor = StructuredOutputExtractor()
    calls = []

    async def broken(_prompt):
        calls.append(1)
        raise RuntimeError("application defect")

    with pytest.raises(RuntimeError, match="application defect"):
        await extractor.extract_with_retry(broken, NewsDigest, "summarise")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_structured_parse_bug_propagates_instead_of_retrying(monkeypatch):
    extractor = StructuredOutputExtractor()

    async def text_response(_prompt):
        return "{}"

    def broken_parser(_response, _model):
        raise RuntimeError("broken schema handler")

    monkeypatch.setattr(extractor, "parse_response", broken_parser)
    with pytest.raises(RuntimeError, match="broken schema"):
        await extractor.extract_with_retry(text_response, NewsDigest, "summarise")


@pytest.mark.asyncio
async def test_bad_json_uses_correction_hint_only_after_receiving_a_response():
    extractor = StructuredOutputExtractor()
    extractor.retry_count = 2
    prompts = []

    async def provider(prompt):
        prompts.append(prompt)
        return "this is not JSON"

    assert await extractor.extract_with_retry(provider, NewsDigest, "summarise") is None
    assert "Previous response was not valid JSON" not in prompts[0]
    assert "Previous response was not valid JSON" in prompts[1]


def make_stream_client(monkeypatch):
    monkeypatch.setattr(streaming_llm, "GEMINI_API_KEY", None)
    monkeypatch.setattr(streaming_llm, "GROQ_API_KEY", None)
    return streaming_llm.StreamingLLMClient()


@pytest.mark.asyncio
async def test_no_streaming_provider_emits_error_not_success(monkeypatch):
    client = make_stream_client(monkeypatch)
    response = [item async for item in client.stream_response("BTC", use_cache=True)]
    assert [item["type"] for item in response] == ["error"]
    assert not client.cache


@pytest.mark.asyncio
async def test_interrupted_gemini_stream_is_not_cached_as_complete(monkeypatch):
    client = make_stream_client(monkeypatch)
    client.gemini_model = object()

    async def partial(*_args):
        yield {"type": "token", "content": "incomplete"}
        raise GoogleAPICallError("synthetic link drop")

    monkeypatch.setattr(client, "_stream_gemini", partial)
    response = [item async for item in client.stream_response("BTC", use_cache=True)]
    assert [item["type"] for item in response] == ["token", "error"]
    assert "incomplete" in response[-1]["content"]
    assert client.cache == {}


@pytest.mark.asyncio
async def test_unexpected_streaming_programmer_fault_propagates(monkeypatch):
    client = make_stream_client(monkeypatch)
    client.gemini_model = object()

    async def broken(*_args):
        if False:
            yield {}
        raise RuntimeError("stream parser defect")

    monkeypatch.setattr(client, "_stream_gemini", broken)
    with pytest.raises(RuntimeError, match="stream parser defect"):
        [item async for item in client.stream_response("BTC")]


@pytest.mark.asyncio
async def test_successful_stream_emits_done_and_caches_only_completed_text(monkeypatch):
    client = make_stream_client(monkeypatch)
    client.gemini_model = object()

    async def successful(*_args):
        yield {"type": "token", "content": "verified "}
        yield {"type": "token", "content": "stream"}

    monkeypatch.setattr(client, "_stream_gemini", successful)
    response = [item async for item in client.stream_response("BTC", use_cache=True)]
    assert [item["type"] for item in response] == ["token", "token", "done"]
    assert next(iter(client.cache.values())) == "verified stream"


@pytest.mark.asyncio
async def test_empty_provider_stream_is_not_mislabeled_completed(monkeypatch):
    client = make_stream_client(monkeypatch)
    client.gemini_model = object()

    async def empty(*_args):
        if False:
            yield {}

    monkeypatch.setattr(client, "_stream_gemini", empty)
    response = [item async for item in client.stream_response("BTC", use_cache=True)]
    assert [item["type"] for item in response] == ["error"]
    assert client.cache == {}
