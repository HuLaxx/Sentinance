"""
RAG (Retrieval Augmented Generation) System v2

Uses Qdrant for vector storage and Gemini for text embeddings.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import structlog
from qdrant_client import QdrantClient, models as qdrant_models
from gemini_client import get_gemini_client

log = structlog.get_logger()

# ============================================
# CONFIGURATION
# ============================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "sentinance_knowledge"
VECTOR_SIZE = 768  # text-embedding-004 output dimension


@dataclass
class Document:
    """A document in the RAG system."""
    id: str
    content: str
    source: str  # 'news', 'analysis', 'price_event', 'research'
    timestamp: datetime
    metadata: Dict

    def to_payload(self) -> dict:
        return {
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            **self.metadata
        }


class RAGService:
    """Main RAG service using Qdrant and Gemini Embeddings."""

    def __init__(self):
        self.gemini = get_gemini_client()
        self.client = None
        self._connected = False
        
        try:
            # Initialize Qdrant Client
            self.client = QdrantClient(url=QDRANT_URL)
            self._ensure_collection()
            self._connected = True
            log.info("qdrant_connected", url=QDRANT_URL)
        except Exception as e:
            log.error("qdrant_connection_failed", error=str(e))
            self._connected = False

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            exists = any(c.name == COLLECTION_NAME for c in collections.collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                log.info("qdrant_collection_created", name=COLLECTION_NAME)
        except Exception as e:
            log.error("qdrant_collection_check_failed", error=str(e))

    async def add_news(self, news_items: List[Dict]) -> None:
        """Add news items to the knowledge base."""
        if not self._connected:
            return

        points = []
        for item in news_items:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            content = f"{title} {summary}".strip()
            
            # Parse timestamp
            published_at = datetime.utcnow()
            raw_date = item.get("published_at") or item.get("published")
            if raw_date:
                try:
                    if isinstance(raw_date, str):
                        published_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    elif isinstance(raw_date, datetime):
                        published_at = raw_date
                except ValueError:
                    pass

            # Create ID
            doc_id = hashlib.sha256(f"{title}|{published_at.isoformat()}".encode()).hexdigest()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

            # Generate Embedding
            embedding = await self.gemini.embed_text(content)
            if not embedding:
                continue

            # Add to batch
            points.append(qdrant_models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": content,
                    "source": "news",
                    "timestamp": published_at.isoformat(),
                    "title": title,
                    "url": item.get("url", "")
                }
            ))

        if points:
            try:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                log.info("rag_documents_added", count=len(points))
            except Exception as e:
                log.error("qdrant_upsert_failed", error=str(e))

    async def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context via semantic search."""
        if not self._connected:
            return "Knowledge base temporarily unavailable."

        # Generate query embedding
        query_vector = await self.gemini.embed_query(query)
        if not query_vector:
            return "Could not process query for retrieval."

        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k
            )
        except Exception as e:
            log.error("qdrant_search_failed", error=str(e))
            return "Error searching knowledge base."

        if not results:
            return "No relevant context found."

        context_parts = ["RELEVANT MARKET INTEL (RAG):"]
        for hit in results:
            score = hit.score
            if score < 0.65: # Filter low relevance
                continue
                
            payload = hit.payload
            source = payload.get("source", "UNKNOWN").upper()
            content = payload.get("content", "")
            params = payload.get("timestamp", "")[:10] # Date only
            
            context_parts.append(f"[{source} {params}] (Confidence: {score:.2f})")
            context_parts.append(f"{content[:400]}...")
            context_parts.append("---")

        return "\n".join(context_parts)


# ============================================
# SINGLETON
# ============================================

_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

async def retrieve_context(query: str, top_k: int = 5) -> str:
    """Main entry point for RAG retrieval."""
    rag = get_rag_service()
    return await rag.retrieve(query, top_k)
