"""Retrieval-quality evaluation for Sentinance RAG against a real Qdrant.

Runs the production RAGService code path (add_news -> Qdrant upsert -> search)
on a fixed synthetic corpus with labelled relevant documents, using an open
embedding model (BAAI/bge-small-en-v1.5 via FastEmbed, no API key) in place of
Gemini. Reports recall@1/3/5, MRR@10 and per-query latency, compared with a
BM25 keyword baseline on the same corpus, plus how many relevant hits the
service's min_relevance cut-off would discard.

Run (needs a Qdrant at --qdrant-url):
    PYTHONPATH=. python -m evals.rag_retrieval_eval --qdrant-url http://127.0.0.1:6333

SYNTHETIC ONLY: scores describe this corpus and model, not production retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

CORPUS = Path(__file__).parent / "data" / "rag_eval_corpus.json"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_DIM = 384
K_VALUES = (1, 3, 5)
TOP_N = 10


class FastEmbedEmbedder:
    """Open, local embedding model with the same async interface as the Gemini client."""

    def __init__(self, model_name: str = MODEL_NAME):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    async def embed_text(self, text: str) -> list[float]:
        return next(iter(self.model.passage_embed([text]))).tolist()

    async def embed_query(self, text: str) -> list[float]:
        return next(iter(self.model.query_embed(text))).tolist()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 keyword baseline (k1=1.5, b=0.75) over title + summary."""

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {doc_id: Counter(tokens(text)) for doc_id, text in docs.items()}
        self.length = {doc_id: sum(c.values()) for doc_id, c in self.tf.items()}
        self.avg = sum(self.length.values()) / len(self.length)
        df = Counter(term for c in self.tf.values() for term in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def rank(self, query: str, top_n: int = TOP_N) -> list[str]:
        q = tokens(query)
        scores = {}
        for doc_id, c in self.tf.items():
            s = 0.0
            for term in q:
                f = c.get(term, 0)
                if f:
                    s += (
                        self.idf[term]
                        * f
                        * (self.k1 + 1)
                        / (
                            f
                            + self.k1
                            * (1 - self.b + self.b * self.length[doc_id] / self.avg)
                        )
                    )
            scores[doc_id] = s
        return [
            d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        ]


def score_rankings(
    rankings: dict[str, list[str]], relevant: dict[str, list[str]]
) -> dict:
    out = {}
    for k in K_VALUES:
        out[f"recall_at_{k}"] = statistics.fmean(
            len(set(rankings[q][:k]) & set(rel)) / len(rel)
            for q, rel in relevant.items()
        )
    rr = []
    for q, rel in relevant.items():
        rank = next(
            (i + 1 for i, d in enumerate(rankings[q][:TOP_N]) if d in rel), None
        )
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr_at_10"] = statistics.fmean(rr)
    out["hit_at_1"] = statistics.fmean(
        1.0 if rankings[q][:1] and rankings[q][0] in rel else 0.0
        for q, rel in relevant.items()
    )
    return out


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]


async def evaluate(qdrant_url: str, corpus_path: Path = CORPUS) -> dict:
    from rag_service import RAGService

    corpus = json.loads(corpus_path.read_text())
    docs = corpus["documents"]
    queries = corpus["queries"]
    relevant = {q["id"]: q["relevant"] for q in queries}
    collection = f"rag_eval_{uuid.uuid4().hex[:8]}"

    embedder = FastEmbedEmbedder()
    rag = RAGService(
        embedder, url=qdrant_url, collection=collection, vector_size=MODEL_DIM
    )
    if not rag._connected:
        raise RuntimeError(f"Qdrant not reachable at {qdrant_url}")
    try:
        await rag.add_news(
            [
                {
                    "id": d["id"],
                    "title": d["title"],
                    "summary": d["summary"],
                    "published_at": corpus["published_at"],
                }
                for d in docs
            ]
        )
        dense, latencies_ms, below_cutoff, relevant_in_top5 = {}, [], 0, 0
        for q in queries:
            start = time.perf_counter()
            hits = await rag.search(q["query"], top_k=TOP_N)
            latencies_ms.append((time.perf_counter() - start) * 1000)
            dense[q["id"]] = [h["payload"].get("external_id", "") for h in hits]
            for h in hits[:5]:
                if h["payload"].get("external_id") in q["relevant"]:
                    relevant_in_top5 += 1
                    if h["score"] < rag.min_relevance:
                        below_cutoff += 1
    finally:
        rag.client.delete_collection(collection)

    bm25 = BM25({d["id"]: f"{d['title']} {d['summary']}" for d in docs})
    lexical = {q["id"]: bm25.rank(q["query"]) for q in queries}
    return {
        "scope": "SYNTHETIC corpus and open embedding model; not production retrieval quality",
        "corpus": {
            "documents": len(docs),
            "queries": len(queries),
            "relevant_pairs": sum(len(r) for r in relevant.values()),
        },
        "embedding_model": MODEL_NAME,
        "dense_qdrant": score_rankings(dense, relevant),
        "bm25_baseline": score_rankings(lexical, relevant),
        "latency_ms_embed_plus_search": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "max": max(latencies_ms),
            "runner": "GitHub Actions ubuntu-latest, CPU",
        },
        "min_relevance_cutoff": rag.min_relevance,
        "relevant_hits_in_top5": relevant_in_top5,
        "relevant_hits_below_cutoff": below_cutoff,
        "per_query_top3": {q: ids[:3] for q, ids in dense.items()},
    }


def markdown(report: dict, floors: dict) -> str:
    d, b = report["dense_qdrant"], report["bm25_baseline"]
    lat = report["latency_ms_embed_plus_search"]
    rows = "\n".join(
        f"| {m} | {d[m]:.3f} | {b[m]:.3f} |"
        for m in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr_at_10", "hit_at_1")
    )
    return (
        "### RAG retrieval evaluation (synthetic corpus)\n\n"
        f"{report['corpus']['documents']} documents, {report['corpus']['queries']} labelled queries, "
        f"model `{report['embedding_model']}` in a real Qdrant.\n\n"
        "| Metric | Dense (Qdrant) | BM25 baseline |\n| --- | --- | --- |\n"
        + rows
        + "\n\n"
        f"Latency (embed + search): p50 {lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms on {lat['runner']}.\n\n"
        f"Relevant hits in top 5 scored below the service's {report['min_relevance_cutoff']} cut-off: "
        f"{report['relevant_hits_below_cutoff']} of {report['relevant_hits_in_top5']}.\n\n"
        f"Floors: {json.dumps(floors)}\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sentinance RAG retrieval evaluation")
    parser.add_argument(
        "--qdrant-url", default=os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    )
    parser.add_argument("--output", type=Path, default=Path("rag_eval_report.json"))
    parser.add_argument("--min-recall-at-5", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument(
        "--require-beat-bm25",
        action="store_true",
        help="fail unless dense MRR@10 is above the BM25 baseline",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(evaluate(args.qdrant_url))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    floors = {
        "recall_at_5": args.min_recall_at_5,
        "mrr_at_10": args.min_mrr,
        "beat_bm25_mrr": args.require_beat_bm25,
    }
    text = markdown(report, floors)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    dense, bm25 = report["dense_qdrant"], report["bm25_baseline"]
    ok = (
        dense["recall_at_5"] >= args.min_recall_at_5
        and dense["mrr_at_10"] >= args.min_mrr
        and (not args.require_beat_bm25 or dense["mrr_at_10"] > bm25["mrr_at_10"])
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
