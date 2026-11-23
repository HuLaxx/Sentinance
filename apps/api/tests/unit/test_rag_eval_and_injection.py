"""Offline tests for the RAG retrieval evaluation and the injectable RAGService.

No Qdrant server, embedding model or network is used here; the real-Qdrant
evaluation runs in CI (rag-retrieval-eval job).
"""

import json
from types import SimpleNamespace

import pytest

import rag_service
from evals.rag_retrieval_eval import BM25, CORPUS, score_rankings


def test_corpus_is_consistent():
    corpus = json.loads(CORPUS.read_text())
    ids = [d["id"] for d in corpus["documents"]]
    titles = [d["title"] for d in corpus["documents"]]
    assert len(ids) == len(set(ids)) and len(titles) == len(set(titles))
    for q in corpus["queries"]:
        assert q["relevant"] and set(q["relevant"]) <= set(ids)
    assert "SYNTHETIC" in corpus["scope"]


def test_metrics_on_a_hand_worked_example():
    scores = score_rankings({"a": ["x", "y"], "b": ["z", "w"]}, {"a": ["y"], "b": ["q"]})
    assert scores["recall_at_1"] == 0.0
    assert scores["recall_at_3"] == 0.5
    assert scores["mrr_at_10"] == 0.25


def test_bm25_ranks_the_matching_document_first():
    bm = BM25({"d1": "bitcoin etf inflows", "d2": "solana outage restart", "d3": "stablecoin reserves"})
    assert bm.rank("solana outage")[0] == "d2"


class _Collections(SimpleNamespace):
    pass


class RecordingClient:
    def __init__(self, url):
        self.url = url
        self.created = None

    def get_collections(self):
        return _Collections(collections=[])

    def create_collection(self, collection_name, vectors_config):
        self.created = (collection_name, vectors_config.size)

    def query_points(self, collection_name, query, limit):
        assert collection_name == "eval_collection" and limit == 3
        hit = SimpleNamespace(id="p1", score=0.9, payload={"external_id": "d7"})
        return SimpleNamespace(points=[hit])


class TinyEmbedder:
    async def embed_text(self, _text):
        return [1.0, 0.0]

    async def embed_query(self, _text):
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_injected_embedder_collection_and_query_points(monkeypatch):
    monkeypatch.setattr(rag_service, "QdrantClient", RecordingClient)
    rag = rag_service.RAGService(TinyEmbedder(), url="http://q:6333", collection="eval_collection", vector_size=2)
    assert rag._connected and rag.client.url == "http://q:6333"
    assert rag.client.created == ("eval_collection", 2)
    hits = await rag.search("anything", top_k=3)
    assert hits == [{"id": "p1", "score": 0.9, "payload": {"external_id": "d7"}}]


def test_production_defaults_unchanged():
    assert rag_service.RAGService.collection == "sentinance_knowledge"
    assert rag_service.RAGService.vector_size == 768
    assert rag_service.RAGService.min_relevance == 0.65
