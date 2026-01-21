"""
Gemini AI Client

Uses Google's Gemini API for market intelligence.
"""

import os
import asyncio
from typing import Optional
import structlog

log = structlog.get_logger()

# Check if google-generativeai is available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-generativeai not installed. Run: pip install google-generativeai")


# ============================================
# CONFIGURATION
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    log.warning("gemini_api_key_missing")

# System prompt for market analysis
SYSTEM_PROMPT = """You are Sentinance AI, an expert crypto market analyst. You provide:

1. **Market Analysis**: Technical and fundamental analysis of cryptocurrencies
2. **Manipulation Detection**: Identify potential wash trading, pump & dump schemes
3. **Price Predictions**: Data-driven price forecasts with confidence levels
4. **Whale Tracking**: Monitor large wallet movements and their impact
5. **News Sentiment**: Analyze how news affects market sentiment

Always be:
- Data-driven and objective
- Clear about confidence levels
- Honest about uncertainty
- Actionable in your insights

Current context: You have access to real-time prices from Binance, Coinbase, and Kraken.
"""


# ============================================
# GEMINI CLIENT
# ============================================

class GeminiClient:
    """Client for Gemini AI interactions."""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.model = None
        self._initialized = False
        
        if GEMINI_AVAILABLE and api_key:
            self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
            )
            self._initialized = True
            log.info("gemini_initialized", model=MODEL_NAME)
        except Exception as e:
            log.error("gemini_init_failed", error=str(e))
            self._initialized = False
    
    async def chat(
        self, 
        message: str, 
        context: Optional[dict] = None
    ) -> dict:
        """
        Send a message to Gemini and get a response.
        
        Args:
            message: User's message
            context: Optional context (prices, alerts, etc.)
        
        Returns:
            Response dict with content and metadata
        """
        if not self._initialized or not self.model:
            return {
                "content": "AI service temporarily unavailable. Please try again later.",
                "error": True,
                "model": None,
            }
        
        try:
            # Build context-aware prompt
            prompt = self._build_prompt(message, context)
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            return {
                "content": response.text,
                "error": False,
                "model": MODEL_NAME,
                "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
            }
        
        except Exception as e:
            log.error("gemini_chat_failed", error=str(e))
            return {
                "content": f"Error generating response: {str(e)}",
                "error": True,
                "model": MODEL_NAME,
            }
    
    def _build_prompt(self, message: str, context: Optional[dict] = None) -> str:
        """Build a context-aware prompt."""
        parts = []
        
        # Add current market data if available
        if context:
            if "prices" in context:
                parts.append("CURRENT PRICES:")
                for symbol, data in context["prices"].items():
                    price = data.get("price", 0)
                    change = data.get("change_24h", 0)
                    parts.append(f"  {symbol}: ${price:,.2f} ({change:+.2f}%)")
                parts.append("")
            
            if "alerts" in context:
                parts.append(f"USER HAS {len(context['alerts'])} ACTIVE ALERTS")
                parts.append("")
        
        # Add user message
        parts.append(f"USER QUERY: {message}")
        
        return "\n".join(parts)
    
    async def analyze_symbol(self, symbol: str, price_data: dict) -> dict:
        """Get AI analysis for a specific symbol."""
        prompt = f"""Analyze {symbol}:
        
Current Price: ${price_data.get('price', 0):,.2f}
24h Change: {price_data.get('change_24h', 0):+.2f}%
24h Volume: ${price_data.get('volume', 0):,.0f}

Provide:
1. Technical outlook (bullish/bearish/neutral)
2. Key support/resistance levels
3. Risk assessment
4. Short-term prediction (24h)

Be concise and actionable."""

        return await self.chat(prompt)
    
    async def embed_text(self, text: str) -> list:
        """
        Generate embeddings for text using Gemini's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector (768 dimensions)
        """
        if not GEMINI_AVAILABLE or not self.api_key:
            log.warning("gemini_embed_unavailable")
            return []
        
        try:
            # Use text-embedding-004 model
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result["embedding"]
        except Exception as e:
            log.error("gemini_embed_failed", error=str(e))
            return []

    async def embed_query(self, query: str) -> list:
        """
        Generate embeddings for a search query.
        Uses task_type="retrieval_query" for better search performance.
        """
        if not GEMINI_AVAILABLE or not self.api_key:
            return []
        
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return result["embedding"]
        except Exception as e:
            log.error("gemini_embed_query_failed", error=str(e))
            return []

    async def detect_manipulation(self, symbol: str, data: dict) -> dict:
        """Ask AI to analyze potential manipulation."""
        prompt = f"""Analyze {symbol} for potential market manipulation:

Price: ${data.get('price', 0):,.2f}
Volume: ${data.get('volume', 0):,.0f}
24h Change: {data.get('change_24h', 0):+.2f}%

Look for signs of:
- Wash trading
- Pump and dump patterns
- Unusual volume spikes
- Spoofing indicators

Provide confidence level (low/medium/high) for any findings."""

        return await self.chat(prompt)


# ============================================
# SINGLETON INSTANCE
# ============================================

_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client singleton."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


async def chat_with_gemini(message: str, context: Optional[dict] = None) -> dict:
    """Convenience function for chatting with Gemini."""
    client = get_gemini_client()
    return await client.chat(message, context)
