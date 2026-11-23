"""
Enhanced AI Chat Service

Integrates:
1. Real-time prices from exchanges
2. ML model predictions (LSTM, ensemble)
3. Technical indicators
4. News context (simple RAG)
5. Gemini AI for reasoning and explanation
"""

import asyncio
import os
from datetime import UTC, datetime

import httpx
import structlog
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.auth.exceptions import GoogleAuthError
from qdrant_client.http.exceptions import ApiException

# Optional Groq SDK: when absent, its error set is empty and Groq is not run.
GROQ_ERRORS: tuple[type[Exception], ...] = ()
try:
    from groq import GroqError
except ImportError:
    pass
else:
    GROQ_ERRORS = (GroqError,)
GROQ_INIT_ERRORS: tuple[type[Exception], ...] = (ValueError,) + GROQ_ERRORS

log = structlog.get_logger()

# ============================================
# IMPORTS
# ============================================

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    log.warning("google-generativeai not installed")

# ============================================
# CONFIGURATION
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    log.warning("gemini_api_key_missing")

# Enhanced system prompt with context awareness
SYSTEM_PROMPT = """You are Sentinance AI, an advanced crypto market intelligence system.

YOU MAY RECEIVE (not guaranteed to be current, accurate, or available):
1. **Provider-reported price snapshots**, only when supplied with an actual source and timestamp
2. **ML Predictions** from our LSTM neural network and ensemble models
3. **Technical Indicators** (RSI, MACD, Bollinger Bands, SMA/EMA)
4. **News Sentiment** from crypto news sources
5. **Anomaly Detection** for manipulation signals

WHEN RESPONDING:
- Use the PROVIDED DATA to make your analysis
- Cite specific numbers from the context
- Explain your reasoning based on the indicators
- Give confidence levels based on how much data supports your view
- Be specific about timeframes (1h, 24h, 7d)

FORMAT:
- Use bullet points for key insights
- Bold important numbers and signals
- Include relevant emojis for visual clarity
- Keep responses focused and actionable

You are NOT just a chatbot - you are a data-driven market analyst that uses real ML models and indicators."""


# ============================================
# DATA GATHERER
# ============================================


class MarketDataGatherer:
    """Gathers all relevant data for AI context."""

    def __init__(self, current_prices: dict, price_history: dict):
        self.prices = current_prices
        self.history = price_history

    def get_price_context(self) -> str:
        """Format current prices for AI."""
        if not self.prices:
            return "No price data available."

        lines = ["📊 **PROVIDED PRICE SNAPSHOT** (source and freshness unverified):"]
        for symbol, data in self.prices.items():
            price = data.get("price", 0)
            change = data.get("change_24h", 0)
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            lines.append(f"  {emoji} {symbol}: ${price:,.2f} ({change:+.2f}% 24h)")
        return "\n".join(lines)

    def get_predictions(self, symbol: str = "BTCUSDT") -> str:
        """Get ML model predictions."""
        try:
            from predictor import generate_prediction

            # Get price history for symbol
            history = self.history.get(symbol, [])
            prices = [h["price"] for h in history] if history else []
            current_price = self.prices.get(symbol, {}).get("price", 0)

            if not prices or current_price == 0:
                return f"Insufficient data for {symbol} predictions."

            # Generate predictions for different horizons
            lines = [f"🤖 **ML PREDICTIONS** for {symbol}:"]

            for horizon in ["1h", "24h", "7d"]:
                pred = generate_prediction(
                    symbol, prices, current_price, horizon, "ensemble"
                )
                emoji = (
                    "📈"
                    if pred.direction == "bullish"
                    else "📉"
                    if pred.direction == "bearish"
                    else "➡️"
                )
                lines.append(
                    f"  {emoji} {horizon}: ${pred.predicted_price:,.2f} "
                    f"({pred.predicted_change_percent:+.1f}%) - {pred.direction.upper()} "
                    "(experimental model output; no validated forecast accuracy)"
                )

            return "\n".join(lines)
        except (OSError, ValueError) as e:
            log.warning("prediction_unavailable", error=str(e))
            return "ML predictions unavailable."

    def get_indicators(self, symbol: str = "BTCUSDT") -> str:
        """Get technical indicators."""
        try:
            from indicators import calculate_all_indicators

            history = self.history.get(symbol, [])
            prices = [h["price"] for h in history] if history else []

            if len(prices) < 14:
                return f"Need more data for {symbol} indicators (have {len(prices)} points, need 14+)."

            indicators_obj = calculate_all_indicators(symbol, prices)
            indicators = indicators_obj.to_dict()

            lines = [f"TECHNICAL INDICATORS for {symbol}:"]

            rsi = indicators.get("rsi_14")
            if rsi is not None:
                rsi_signal = (
                    "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
                )
                lines.append(f"  - RSI(14): {rsi:.1f} - {rsi_signal}")

            macd = indicators.get("macd") or {}
            macd_line = macd.get("value")
            signal = macd.get("signal")
            if macd_line is not None and signal is not None:
                macd_signal = "BULLISH" if macd_line > signal else "BEARISH"
                lines.append(
                    f"  - MACD: {macd_line:.2f} (Signal: {signal:.2f}) - {macd_signal}"
                )

            bb = indicators.get("bollinger_bands") or {}
            if bb:
                current = prices[-1]
                upper = bb.get("upper")
                lower = bb.get("lower")
                if upper is not None and current > upper:
                    bb_signal = "ABOVE UPPER BAND (overbought)"
                elif lower is not None and current < lower:
                    bb_signal = "BELOW LOWER BAND (oversold)"
                else:
                    bb_signal = "WITHIN BANDS (normal)"
                lines.append(f"  - Bollinger: {bb_signal}")

            moving_avgs = indicators.get("moving_averages") or {}
            sma = moving_avgs.get("sma_20")
            if sma is not None:
                current = prices[-1]
                sma_signal = (
                    "ABOVE SMA (bullish)" if current > sma else "BELOW SMA (bearish)"
                )
                lines.append(f"  - SMA(20): ${sma:,.2f} - Price {sma_signal}")

            return "\n".join(lines)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            log.warning("indicator_unavailable", error=str(e))
            return "Technical indicators unavailable."

    def get_news_context(self) -> str:
        """Do not pass hard-coded headlines off as current market observations."""
        return (
            "News context unavailable: live news has not been verified. "
            "No sample headlines are included in market analysis."
        )

    def get_anomaly_detection(self, symbol: str = "BTCUSDT") -> str:
        """Check for market anomalies."""
        try:
            from anomaly_detection import run_anomaly_detection

            history = self.history.get(symbol, [])
            if len(history) < 10:
                return "Insufficient data for anomaly detection."

            prices = [h["price"] for h in history]
            volumes = [h.get("volume", 0) for h in history]

            anomalies = run_anomaly_detection(symbol, prices, volumes)

            if not anomalies:
                return f"ANOMALY CHECK for {symbol}: No manipulation signals detected."

            lines = [f"ANOMALY ALERTS for {symbol}:"]
            for anomaly in anomalies:
                info = anomaly.to_dict()
                lines.append(f"  - {info['type']}: {info['description']}")

            return "\n".join(lines)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            log.warning("anomaly_unavailable", error=str(e))
            return "Anomaly detection unavailable."

    async def build_full_context(self, user_query: str) -> str:
        """Build complete context for AI."""
        # Detect which symbol user is asking about
        query_lower = user_query.lower()
        symbol = "BTCUSDT"  # default

        if "eth" in query_lower or "ethereum" in query_lower:
            symbol = "ETHUSDT"
        elif "sol" in query_lower or "solana" in query_lower:
            symbol = "SOLUSDT"
        elif "bnb" in query_lower or "binance" in query_lower:
            symbol = "BNBUSDT"
        elif "xrp" in query_lower or "ripple" in query_lower:
            symbol = "XRPUSDT"
        elif "doge" in query_lower:
            symbol = "DOGEUSDT"

        # Get RAG knowledge context
        try:
            from rag_service import retrieve_context

            rag_context = await retrieve_context(user_query, top_k=3)
        except (ApiException, httpx.HTTPError, GoogleAPICallError) as e:
            log.warning("rag_provider_unavailable", error=str(e))
            rag_context = ""

        context_parts = [
            f"USER QUERY: {user_query}",
            f"ANALYSIS TIMESTAMP: {datetime.now(UTC).isoformat()}",
            "",
            self.get_price_context(),
            "",
            self.get_predictions(symbol),
            "",
            self.get_indicators(symbol),
            "",
            self.get_anomaly_detection(symbol),
            "",
            self.get_news_context(),
            "",
            rag_context,  # Add RAG context
        ]
        return "\n".join(context_parts)


# ============================================
# ENHANCED AI CLIENT (With Multi-Provider Fallback)
# ============================================


class EnhancedAIChat:
    """AI Chat with full ML integration and multi-provider fallback."""

    def __init__(self):
        self.gemini_model = None
        self.groq_client = None
        self._gemini_initialized = False
        self._groq_initialized = False

        # Initialize Gemini (Primary)
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            self._init_gemini()

        # Initialize Groq (Fallback)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if self.groq_api_key:
            self._init_groq()

    def _init_gemini(self):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
            )
            self._gemini_initialized = True
            log.info("gemini_initialized", model=MODEL_NAME)
        except (GoogleAPICallError, RetryError, GoogleAuthError, ValueError) as e:
            log.error("gemini_init_failed", error=str(e))

    def _init_groq(self):
        try:
            from groq import Groq

            self.groq_client = Groq(api_key=self.groq_api_key)
            self._groq_initialized = True
            log.info("groq_initialized", model=self.groq_model)
        except GROQ_INIT_ERRORS as e:
            log.error("groq_init_failed", error=str(e))

    async def chat(
        self, message: str, current_prices: dict, price_history: dict
    ) -> dict:
        """Generate AI response with full context and fallback."""

        # Build comprehensive context
        gatherer = MarketDataGatherer(current_prices, price_history)
        full_context = await gatherer.build_full_context(message)

        prompt_content = f"""CONTEXT DATA (use this for your analysis):
{full_context}

---

Based only on available context, please answer the user's question. Distinguish provider observations from estimated, cached or sample data.
Remember to cite the specific numbers from the context in your response."""

        # 1. Try Gemini (Primary)
        if self._gemini_initialized and self.gemini_model:
            try:
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content, prompt_content
                )
                return {
                    "content": response.text,
                    "error": False,
                    "model": MODEL_NAME,
                    "provider": "google",
                    "context_used": True,
                }
            except (GoogleAPICallError, RetryError, GoogleAuthError) as e:
                log.warning("gemini_failed_switching_to_fallback", error=str(e))

        # 2. Try Groq (Fallback)
        if self._groq_initialized and self.groq_client:
            try:
                log.info("using_groq_fallback", model=self.groq_model)
                completion = await asyncio.to_thread(
                    self.groq_client.chat.completions.create,
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_content},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                return {
                    "content": completion.choices[0].message.content,
                    "error": False,
                    "model": self.groq_model,
                    "provider": "groq",
                    "context_used": True,
                }
            except GROQ_ERRORS as e:
                log.error("groq_fallback_failed", error=str(e))

        # 3. Both Failed
        return {
            "content": "AI service currently unavailable. Please check API keys or try again later.",
            "error": True,
            "model": None,
            "context_used": False,
        }


_enhanced_chat: EnhancedAIChat | None = None


def get_enhanced_chat() -> EnhancedAIChat:
    global _enhanced_chat
    if _enhanced_chat is None:
        _enhanced_chat = EnhancedAIChat()
    return _enhanced_chat


async def chat_with_ai(message: str, current_prices: dict, price_history: dict) -> dict:
    """Main entry point for enhanced AI chat."""
    chat = get_enhanced_chat()
    return await chat.chat(message, current_prices, price_history)
