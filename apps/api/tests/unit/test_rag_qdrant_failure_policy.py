"""Qdrant failures must not create a false connected state or successful writes.

All tests use a synthetic Qdrant client and never require a hosted vector DB.
"""

from types import SimpleNamespace

import pytest
from qdrant_client.http.exceptions import ApiException

import rag_service


class FakeEmbedding:
    async def embed_text(self, _content):
        return [0.1, 0.2]

    async def embed_query(self, _query):
        return [0.1, 0.2]


def disconnected_qdrant():
    raise ApiException("synthetic Qdrant outage")


def configured_service(client):
    rag = rag_service.RAGService.__new__(rag_service.RAGService)
    rag.client = client
    rag.gemini = FakeEmbedding()
    rag._connected = True
    return rag


def test_collection_readiness_does_not_report_success_after_provider_failure(
    monkeypatch,
):
    class BrokenClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            disconnected_qdrant()

    monkeypatch.setattr(rag_service, "QdrantClient", BrokenClient)
    monkeypatch.setattr(rag_service, "get_gemini_client", FakeEmbedding)
    result = rag_service.RAGService()
    assert result._connected is False


def test_collection_init_programming_error_propagates(monkeypatch):
    class BuggyClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            raise RuntimeError("broken collection logic")

    monkeypatch.setattr(rag_service, "QdrantClient", BuggyClient)
    monkeypatch.setattr(rag_service, "get_gemini_client", FakeEmbedding)
    with pytest.raises(RuntimeError, match="broken collection logic"):
        rag_service.RAGService()


def test_collection_creation_failure_does_not_claim_connected(monkeypatch):
    class CreateFailure:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            return SimpleNamespace(collections=[])

        def create_collection(self, **_kwargs):
            disconnected_qdrant()

    monkeypatch.setattr(rag_service, "QdrantClient", CreateFailure)
    monkeypatch.setattr(rag_service, "get_gemini_client", FakeEmbedding)
    result = rag_service.RAGService()
    assert result._connected is False


@pytest.mark.asyncio
async def test_qdrant_search_outage_returns_explicit_unavailability():
    client = SimpleNamespace(search=lambda **kwargs: disconnected_qdrant())
    rag = configured_service(client)
    assert await rag.retrieve("BTC") == "Knowledge base temporarily unavailable."


@pytest.mark.asyncio
async def test_qdrant_search_programming_error_is_not_hidden():
    def buggy_search(**_kwargs):
        raise RuntimeError("search code defect")

    rag = configured_service(SimpleNamespace(search=buggy_search))
    with pytest.raises(RuntimeError, match="search code defect"):
        await rag.retrieve("BTC")


@pytest.mark.asyncio
async def test_qdrant_upsert_failure_surfaces_to_caller():
    client = SimpleNamespace(upsert=lambda **kwargs: disconnected_qdrant())
    rag = configured_service(client)
    with pytest.raises(ApiException, match="outage"):
        await rag.add_news([{"title": "Example", "summary": "Synthetic headline"}])


@pytest.mark.asyncio
async def test_qdrant_upsert_programming_failure_is_not_disguised():
    def buggy_write(**_kwargs):
        raise RuntimeError("write implementation bug")

    rag = configured_service(SimpleNamespace(upsert=buggy_write))
    with pytest.raises(RuntimeError, match="implementation bug"):
        await rag.add_news([{"title": "Example", "summary": "Synthetic headline"}])
