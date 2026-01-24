"""
Enhanced LLM Wrapper with Historical RAG Learning

This module provides an intelligent LLM interface that:
1. Learns from past predictions and their actual outcomes
2. Stores historical context for each asset/index
3. Uses RAG to retrieve relevant past insights
4. Improves predictions based on historical accuracy

Key Features:
- Per-index historical memory
- Prediction accuracy tracking
- Context-aware response generation
- Multi-provider fallback (Gemini, Groq)
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import structlog

log = structlog.get_logger()

# ============================================
# CONFIGURATION
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from qdrant_client import QdrantClient, models as qdrant_models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ============================================
# DATA MODELS
# ============================================

@dataclass
class HistoricalPrediction:
    """A historical prediction with outcome tracking."""
    id: str
    symbol: str
    query: str
    prediction: str
    predicted_price: Optional[float]
    predicted_direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    timestamp: str
    actual_price: Optional[float] = None
    actual_direction: Optional[str] = None
    accuracy_score: Optional[float] = None  # 0-1, how accurate was this
    verified_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "query": self.query,
            "prediction": self.prediction[:200],  # Truncate for storage
            "predicted_price": self.predicted_price,
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "actual_price": self.actual_price,
            "actual_direction": self.actual_direction,
            "accuracy_score": self.accuracy_score,
            "verified_at": self.verified_at,
        }
    
    def calculate_accuracy(self) -> float:
        """Calculate accuracy score based on outcome."""
        if self.actual_price is None or self.predicted_price is None:
            return 0.5  # Unknown
        
        # Direction accuracy (0.5 weight)
        direction_score = 1.0 if self.actual_direction == self.predicted_direction else 0.0
        
        # Price accuracy (0.5 weight) - based on how close prediction was
        price_error = abs(self.actual_price - self.predicted_price) / self.predicted_price
        price_score = max(0, 1 - price_error * 5)  # 20% error = 0 score
        
        self.accuracy_score = (direction_score * 0.5) + (price_score * 0.5)
        return self.accuracy_score


@dataclass
class IndexContext:
    """Historical context for a specific index/symbol."""
    symbol: str
    predictions: List[HistoricalPrediction] = field(default_factory=list)
    avg_accuracy: float = 0.5
    successful_patterns: List[str] = field(default_factory=list)
    failed_patterns: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None
    
    def update_accuracy(self):
        """Recalculate average accuracy from verified predictions."""
        verified = [p for p in self.predictions if p.accuracy_score is not None]
        if verified:
            self.avg_accuracy = sum(p.accuracy_score for p in verified) / len(verified)
        self.last_updated = datetime.utcnow().isoformat()


# ============================================
# HISTORICAL RAG MEMORY
# ============================================

class HistoricalRAGMemory:
    """
    Vector-based historical memory for each index.
    
    Stores past predictions and their outcomes to:
    1. Retrieve similar past situations
    2. Learn from successful/failed predictions
    3. Provide context for new predictions
    """
    
    COLLECTION_NAME = "sentinance_history"
    VECTOR_SIZE = 768  # Gemini embedding dimension
    
    def __init__(self):
        self.qdrant = None
        self.gemini = None
        self.index_contexts: Dict[str, IndexContext] = {}
        self._initialized = False
        
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Qdrant and Gemini clients."""
        # Qdrant
        if QDRANT_AVAILABLE:
            try:
                qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
                self.qdrant = QdrantClient(url=qdrant_url)
                self._ensure_collection()
                log.info("historical_qdrant_connected")
            except Exception as e:
                log.warning("historical_qdrant_failed", error=str(e))
        
        # Gemini
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini = genai.GenerativeModel("gemini-2.5-flash")
                self._initialized = True
                log.info("historical_gemini_connected")
            except Exception as e:
                log.warning("historical_gemini_failed", error=str(e))
    
    def _ensure_collection(self):
        """Create Qdrant collection if not exists."""
        if not self.qdrant:
            return
        
        try:
            collections = self.qdrant.get_collections()
            exists = any(c.name == self.COLLECTION_NAME for c in collections.collections)
            
            if not exists:
                self.qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                log.info("historical_collection_created")
        except Exception as e:
            log.error("collection_creation_failed", error=str(e))
    
    async def _embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using Gemini."""
        if not GEMINI_AVAILABLE:
            return None
        
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            log.warning("embedding_failed", error=str(e))
            return None
    
    async def store_prediction(self, prediction: HistoricalPrediction):
        """Store a prediction in vector memory."""
        if not self.qdrant:
            log.warning("qdrant_not_available")
            return
        
        # Create embedding from prediction context
        text = f"{prediction.symbol} {prediction.query} {prediction.prediction}"
        embedding = await self._embed_text(text)
        
        if not embedding:
            return
        
        try:
            self.qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    qdrant_models.PointStruct(
                        id=prediction.id,
                        vector=embedding,
                        payload=prediction.to_dict()
                    )
                ]
            )
            
            # Update index context
            if prediction.symbol not in self.index_contexts:
                self.index_contexts[prediction.symbol] = IndexContext(symbol=prediction.symbol)
            
            self.index_contexts[prediction.symbol].predictions.append(prediction)
            log.info("prediction_stored", symbol=prediction.symbol, id=prediction.id)
            
        except Exception as e:
            log.error("prediction_store_failed", error=str(e))
    
    async def retrieve_similar(
        self,
        symbol: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve similar past predictions for context."""
        if not self.qdrant:
            return []
        
        # Create query embedding
        embedding = await self._embed_text(f"{symbol} {query}")
        if not embedding:
            return []
        
        try:
            results = self.qdrant.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=embedding,
                query_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="symbol",
                            match=qdrant_models.MatchValue(value=symbol)
                        )
                    ]
                ),
                limit=top_k
            )
            
            return [
                {
                    **hit.payload,
                    "similarity": hit.score
                }
                for hit in results
                if hit.score > 0.6  # Only return relevant matches
            ]
        except Exception as e:
            log.error("retrieval_failed", error=str(e))
            return []
    
    async def update_with_actual(
        self,
        prediction_id: str,
        actual_price: float,
        actual_direction: str
    ):
        """Update a prediction with actual outcome for learning."""
        if not self.qdrant:
            return
        
        try:
            # Find the prediction
            for symbol, context in self.index_contexts.items():
                for pred in context.predictions:
                    if pred.id == prediction_id:
                        pred.actual_price = actual_price
                        pred.actual_direction = actual_direction
                        pred.verified_at = datetime.utcnow().isoformat()
                        pred.calculate_accuracy()
                        
                        # Update in Qdrant
                        self.qdrant.set_payload(
                            collection_name=self.COLLECTION_NAME,
                            points=[prediction_id],
                            payload={
                                "actual_price": actual_price,
                                "actual_direction": actual_direction,
                                "accuracy_score": pred.accuracy_score,
                                "verified_at": pred.verified_at,
                            }
                        )
                        
                        # Update context accuracy
                        context.update_accuracy()
                        log.info("prediction_verified", id=prediction_id, accuracy=pred.accuracy_score)
                        return
        except Exception as e:
            log.error("update_actual_failed", error=str(e))
    
    def get_index_summary(self, symbol: str) -> Dict:
        """Get summary of historical performance for an index."""
        context = self.index_contexts.get(symbol)
        if not context:
            return {
                "symbol": symbol,
                "total_predictions": 0,
                "avg_accuracy": 0.5,
                "message": "No historical data yet"
            }
        
        verified = [p for p in context.predictions if p.accuracy_score is not None]
        
        return {
            "symbol": symbol,
            "total_predictions": len(context.predictions),
            "verified_predictions": len(verified),
            "avg_accuracy": round(context.avg_accuracy, 2),
            "last_updated": context.last_updated,
        }


# ============================================
# ENHANCED LLM WRAPPER
# ============================================

class EnhancedLLMWrapper:
    """
    LLM wrapper with RAG-based historical learning.
    
    Features:
    - Retrieves similar past predictions for context
    - Learns from prediction accuracy
    - Multi-provider fallback (Gemini → Groq)
    - Per-index historical memory
    """
    
    SYSTEM_PROMPT = """You are Sentinance AI, a crypto market intelligence system with HISTORICAL LEARNING capabilities.

You have access to:
1. Real-time prices from exchanges
2. Technical indicators (RSI, MACD, Bollinger)
3. ML model predictions (LSTM)
4. HISTORICAL PREDICTIONS - past predictions for this asset and their outcomes

IMPORTANT: Learn from past prediction accuracy!
- If past similar predictions were accurate, have higher confidence
- If past predictions failed, be more cautious and explain why this time is different
- Always reference relevant historical context in your analysis

When responding:
- Cite specific historical predictions that are relevant
- Adjust confidence based on historical accuracy  
- Explain your reasoning based on both current data and past patterns
"""

    def __init__(self):
        self.memory = HistoricalRAGMemory()
        self.gemini_model = None
        self.groq_client = None
        
        self._init_providers()
    
    def _init_providers(self):
        """Initialize LLM providers."""
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(
                    "gemini-2.5-flash",
                    system_instruction=self.SYSTEM_PROMPT
                )
            except Exception as e:
                log.warning("gemini_init_failed", error=str(e))
        
        if GROQ_AVAILABLE and GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                log.warning("groq_init_failed", error=str(e))
    
    async def generate_with_history(
        self,
        symbol: str,
        query: str,
        current_context: Dict,
    ) -> Dict:
        """
        Generate response with historical RAG context.
        
        Args:
            symbol: Asset symbol (e.g., BTCUSDT)
            query: User's question
            current_context: Current market data
            
        Returns:
            Response with prediction and metadata
        """
        # Step 1: Retrieve similar historical predictions
        historical = await self.memory.retrieve_similar(symbol, query, top_k=5)
        historical_context = self._format_historical_context(historical)
        
        # Step 2: Get index performance summary
        performance = self.memory.get_index_summary(symbol)
        
        # Step 3: Build full prompt
        full_prompt = f"""
CURRENT MARKET CONTEXT:
{json.dumps(current_context, indent=2)}

HISTORICAL PREDICTIONS for {symbol}:
{historical_context}

HISTORICAL ACCURACY: {performance.get('avg_accuracy', 0.5) * 100:.0f}% across {performance.get('verified_predictions', 0)} verified predictions

USER QUERY: {query}

Based on the current data AND historical patterns, provide your analysis.
Include your confidence level (0-100%) adjusted for historical accuracy.
"""
        
        # Step 4: Generate response
        response = await self._call_llm(full_prompt)
        
        # Step 5: Extract prediction details
        prediction_details = self._extract_prediction(response, symbol)
        
        # Step 6: Store for future learning
        prediction = HistoricalPrediction(
            id=hashlib.md5(f"{symbol}-{query}-{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12],
            symbol=symbol,
            query=query,
            prediction=response,
            predicted_price=prediction_details.get("price"),
            predicted_direction=prediction_details.get("direction", "neutral"),
            confidence=prediction_details.get("confidence", 0.5),
            timestamp=datetime.utcnow().isoformat(),
        )
        
        await self.memory.store_prediction(prediction)
        
        return {
            "response": response,
            "prediction_id": prediction.id,
            "historical_context_used": len(historical),
            "historical_accuracy": performance.get("avg_accuracy"),
            "confidence": prediction_details.get("confidence"),
            "model_used": "gemini" if self.gemini_model else "groq",
        }
    
    def _format_historical_context(self, historical: List[Dict]) -> str:
        """Format historical predictions for context."""
        if not historical:
            return "No similar historical predictions found."
        
        lines = []
        for h in historical:
            accuracy = h.get("accuracy_score")
            accuracy_str = f"Accuracy: {accuracy*100:.0f}%" if accuracy else "Not yet verified"
            
            lines.append(f"""
[{h.get('timestamp', '')[:10]}] Predicted: {h.get('predicted_direction', 'unknown').upper()}
  Query: {h.get('query', '')[:100]}
  Predicted Price: ${h.get('predicted_price', 0):,.2f}
  Actual: {h.get('actual_direction', 'pending').upper()} at ${h.get('actual_price', 0):,.2f}
  {accuracy_str} | Similarity: {h.get('similarity', 0)*100:.0f}%
""")
        
        return "\n".join(lines)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with fallback."""
        # Try Gemini first
        if self.gemini_model:
            try:
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content,
                    prompt
                )
                return response.text
            except Exception as e:
                log.warning("gemini_call_failed", error=str(e))
        
        # Fallback to Groq
        if self.groq_client:
            try:
                completion = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return completion.choices[0].message.content
            except Exception as e:
                log.error("groq_call_failed", error=str(e))
        
        return "AI service temporarily unavailable."
    
    def _extract_prediction(self, response: str, symbol: str) -> Dict:
        """Extract structured prediction from response."""
        response_lower = response.lower()
        
        # Extract direction
        if "bullish" in response_lower or "buy" in response_lower or "upward" in response_lower:
            direction = "bullish"
        elif "bearish" in response_lower or "sell" in response_lower or "downward" in response_lower:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Extract confidence (look for percentage)
        import re
        confidence_match = re.search(r'(\d{1,3})%\s*confiden', response_lower)
        confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.5
        
        # Extract price (look for dollar amounts)
        price_match = re.search(r'\$([0-9,]+(?:\.[0-9]{2})?)', response)
        price = float(price_match.group(1).replace(",", "")) if price_match else None
        
        return {
            "direction": direction,
            "confidence": confidence,
            "price": price,
        }
    
    async def verify_prediction(
        self,
        prediction_id: str,
        actual_price: float,
    ):
        """Verify a prediction with actual outcome."""
        # Determine actual direction based on context
        # This would normally compare to the predicted price
        actual_direction = "bullish"  # Placeholder
        
        await self.memory.update_with_actual(
            prediction_id,
            actual_price,
            actual_direction
        )


# ============================================
# SINGLETON
# ============================================

_llm_wrapper: Optional[EnhancedLLMWrapper] = None


def get_llm_wrapper() -> EnhancedLLMWrapper:
    """Get singleton LLM wrapper."""
    global _llm_wrapper
    if _llm_wrapper is None:
        _llm_wrapper = EnhancedLLMWrapper()
    return _llm_wrapper


async def chat_with_history(
    symbol: str,
    query: str,
    current_context: Dict,
) -> Dict:
    """Main entry point for historical RAG chat."""
    wrapper = get_llm_wrapper()
    return await wrapper.generate_with_history(symbol, query, current_context)


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    async def main():
        wrapper = get_llm_wrapper()
        
        # Example query
        response = await wrapper.generate_with_history(
            symbol="BTCUSDT",
            query="What's your prediction for BTC in the next 24 hours?",
            current_context={
                "price": 95000,
                "change_24h": 2.5,
                "rsi_14": 62,
                "macd": {"value": 150, "signal": 145},
            }
        )
        
        print(f"Response: {response['response'][:500]}...")
        print(f"Prediction ID: {response['prediction_id']}")
        print(f"Historical context used: {response['historical_context_used']}")
        print(f"Confidence: {response['confidence']}")
    
    asyncio.run(main())
