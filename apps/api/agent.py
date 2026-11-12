"""
LangGraph AI Agent System

Multi-agent system for market intelligence:
- Planner: Breaks down queries into sub-tasks
- Researcher: Gathers market data
- Analyst: Synthesizes insights
"""

from datetime import datetime, timezone
from math import isfinite
from typing import Annotated, Any, TypedDict

import structlog

log = structlog.get_logger()

# Check if langgraph is available
try:
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # The fallback reducer is only used when LangGraph is unavailable.
    add_messages = lambda left, right: left + right
    log.warning(
        "langgraph_not_installed", message="Install with: pip install langgraph"
    )


# ============================================
# STATE DEFINITION
# ============================================


class AgentState(TypedDict):
    """State shared between all agent nodes."""

    query: str
    messages: Annotated[list, add_messages]
    plan: list[str] | None
    research_data: dict[str, Any] | None
    analysis: str | None
    confidence: float | None
    timestamp: str


# ============================================
# TOOL DEFINITIONS (MCP-style)
# ============================================

AVAILABLE_TOOLS = {
    "get_current_price": {
        "description": "Get current price for a cryptocurrency symbol",
        "parameters": {"symbol": "string"},
    },
    "get_price_history": {
        "description": "Get historical prices for a symbol",
        "parameters": {"symbol": "string", "limit": "integer"},
    },
    "get_order_book_depth": {
        "description": "Get order book bid/ask depth",
        "parameters": {"symbol": "string"},
    },
    "analyze_whale_movements": {
        "description": "Check recent whale wallet movements",
        "parameters": {"symbol": "string"},
    },
    "get_news_sentiment": {
        "description": "Get recent news and sentiment analysis",
        "parameters": {"symbol": "string"},
    },
    "calculate_technical_indicators": {
        "description": "Calculate RSI, MACD, and other indicators",
        "parameters": {"symbol": "string"},
    },
}


# ============================================
# AGENT NODES
# ============================================


async def planner_node(state: AgentState) -> AgentState:
    """
    Planner Agent: Breaks down the query into actionable sub-tasks.
    In production, this would call an LLM.
    """
    query = state["query"].lower()
    plan = []

    # Simple rule-based planning (replace with LLM in production)
    if "price" in query or "value" in query:
        plan.append("get_current_price")

    if "manipulat" in query or "whale" in query or "suspicious" in query:
        plan.append("analyze_whale_movements")
        plan.append("get_order_book_depth")

    if (
        "technical" in query
        or "indicator" in query
        or "rsi" in query
        or "macd" in query
    ):
        plan.append("calculate_technical_indicators")

    if "news" in query or "sentiment" in query:
        plan.append("get_news_sentiment")

    if "predict" in query or "forecast" in query:
        plan.append("get_price_history")
        plan.append("calculate_technical_indicators")

    # Default actions
    if not plan:
        plan = ["get_current_price", "get_news_sentiment"]

    log.info("planner_completed", plan=plan)

    return {
        **state,
        "plan": plan,
        "messages": state.get("messages", [])
        + [{"role": "system", "content": f"Plan created: {plan}"}],
    }


async def researcher_node(state: AgentState) -> AgentState:
    """
    Researcher Agent: Executes the plan by calling real APIs.
    Fetches actual market data from exchanges and calculates indicators.
    """
    plan = state.get("plan") or []
    research_data: dict[str, Any] = {}

    # Extract symbol from query (simple extraction)
    query = state["query"].upper()
    symbol = "BTCUSDT"  # Default
    for s in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE"]:
        if s in query:
            symbol = f"{s}USDT"
            break

    # Execute each tool in the plan with REAL data
    for tool in plan:
        try:
            if tool == "get_current_price":
                # REAL: Fetch from Binance
                try:
                    import httpx
                    from multi_exchange import fetch_binance_prices

                    async with httpx.AsyncClient() as client:
                        prices = await fetch_binance_prices(client)
                        if symbol in prices:
                            p = prices[symbol]
                            research_data["current_price"] = {
                                "symbol": symbol,
                                "price": p.get("price", 0),
                                "change_24h": p.get("change_24h", 0),
                                "volume": p.get("volume", 0),
                                "high": p.get("high", 0),
                                "low": p.get("low", 0),
                            }
                        else:
                            research_data["current_price"] = {
                                "symbol": symbol,
                                "error": "Symbol not found",
                            }
                except Exception as e:
                    log.warning("get_current_price_failed", error=str(e))
                    research_data["current_price"] = {"symbol": symbol, "error": str(e)}

            elif tool == "analyze_whale_movements":
                # Whale data requires specialized APIs (mock for now, but structured for real integration)
                research_data["whale_data"] = {
                    "large_transfers_24h": "Data requires on-chain API integration",
                    "net_exchange_flow": "Recommend: Glassnode, CryptoQuant API",
                    "note": "On-chain analytics pending API key configuration",
                }

            elif tool == "get_order_book_depth":
                # REAL: Fetch order book from Binance
                try:
                    import httpx

                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=100",
                            timeout=5.0,
                        )
                        if resp.status_code == 200:
                            book = resp.json()
                            bids = sum(float(b[1]) for b in book.get("bids", [])[:20])
                            asks = sum(float(a[1]) for a in book.get("asks", [])[:20])
                            best_bid = (
                                float(book["bids"][0][0]) if book.get("bids") else 0
                            )
                            best_ask = (
                                float(book["asks"][0][0]) if book.get("asks") else 0
                            )
                            spread = (
                                ((best_ask - best_bid) / best_bid * 100)
                                if best_bid > 0
                                else 0
                            )
                            research_data["order_book"] = {
                                "bid_depth_top20": round(bids, 2),
                                "ask_depth_top20": round(asks, 2),
                                "spread_percent": round(spread, 4),
                                "bid_ask_ratio": round(bids / asks, 2)
                                if asks > 0
                                else 0,
                            }
                except Exception as e:
                    log.warning("get_order_book_failed", error=str(e))
                    research_data["order_book"] = {"error": str(e)}

            elif tool == "calculate_technical_indicators":
                # REAL: Calculate using our indicators module
                try:
                    # We need price history - fetch recent candles
                    import httpx
                    from indicators import calculate_all_indicators

                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100",
                            timeout=5.0,
                        )
                        if resp.status_code == 200:
                            klines = resp.json()
                            close_prices = [float(k[4]) for k in klines]  # Close prices
                            indicators_obj = calculate_all_indicators(symbol, close_prices)
                            ind = indicators_obj.to_dict()
                            research_data["indicators"] = {
                                "rsi_14": ind.get("rsi_14"),
                                "macd": ind.get("macd"),
                                "sma_20": ind.get("moving_averages", {}).get("sma_20"),
                                "sma_50": ind.get("moving_averages", {}).get("sma_50"),
                                "bollinger": ind.get("bollinger_bands"),
                            }
                except Exception as e:
                    log.warning("calculate_indicators_failed", error=str(e))
                    research_data["indicators"] = {"error": str(e)}

            elif tool == "get_news_sentiment":
                # REAL: Fetch news using news_scraper
                try:
                    from news_scraper import get_latest_news

                    news = await get_latest_news(limit=10)
                    if news:
                        # Simple sentiment approximation based on keywords
                        positive_words = [
                            "bull",
                            "surge",
                            "gain",
                            "up",
                            "high",
                            "record",
                            "etf",
                            "adoption",
                        ]
                        negative_words = [
                            "bear",
                            "crash",
                            "drop",
                            "down",
                            "low",
                            "fear",
                            "sell",
                            "dump",
                        ]

                        pos_count = sum(
                            1
                            for n in news
                            for w in positive_words
                            if w in n.get("title", "").lower()
                        )
                        neg_count = sum(
                            1
                            for n in news
                            for w in negative_words
                            if w in n.get("title", "").lower()
                        )

                        total = pos_count + neg_count
                        sentiment = (pos_count - neg_count) / total if total > 0 else 0

                        research_data["sentiment"] = {
                            "overall": round(sentiment, 2),
                            "news_count": len(news),
                            "recent_headlines": [
                                n.get("title", "")[:80] for n in news[:5]
                            ],
                        }
                    else:
                        research_data["sentiment"] = {
                            "overall": 0,
                            "news_count": 0,
                            "note": "No recent news found",
                        }
                except Exception as e:
                    log.warning("get_news_failed", error=str(e))
                    research_data["sentiment"] = {"error": str(e)}

            elif tool == "get_price_history":
                # REAL: Fetch historical data from Binance
                try:
                    import httpx

                    async with httpx.AsyncClient() as client:
                        resp = await client.get(
                            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=7",
                            timeout=5.0,
                        )
                        if resp.status_code == 200:
                            klines = resp.json()
                            highs = [float(k[2]) for k in klines]
                            lows = [float(k[3]) for k in klines]
                            closes = [float(k[4]) for k in klines]
                            research_data["price_history"] = {
                                "high_24h": highs[-1] if highs else 0,
                                "low_24h": lows[-1] if lows else 0,
                                "high_7d": max(highs) if highs else 0,
                                "low_7d": min(lows) if lows else 0,
                                "close_7d_ago": closes[0] if closes else 0,
                                "close_now": closes[-1] if closes else 0,
                                "change_7d_pct": round(
                                    ((closes[-1] - closes[0]) / closes[0] * 100), 2
                                )
                                if closes and closes[0] > 0
                                else 0,
                            }
                except Exception as e:
                    log.warning("get_price_history_failed", error=str(e))
                    research_data["price_history"] = {"error": str(e)}

        except Exception as e:
            log.error("researcher_tool_error", tool=tool, error=str(e))
            research_data[tool] = {"error": str(e)}

    log.info("researcher_completed", data_keys=list(research_data.keys()))

    return {
        **state,
        "research_data": research_data,
        "messages": state.get("messages", [])
        + [
            {
                "role": "system",
                "content": f"Research completed: {list(research_data.keys())}",
            }
        ],
    }


def _valid_metric(value: object) -> bool:
    """Accept finite numeric provider values, never placeholders or bools."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


async def analyst_node(state: AgentState) -> AgentState:
    """Summarise verified fields only; never invent trends or prediction confidence.

    `confidence` is an evidence-coverage proxy, not a probability that a
    forecast, trade, or claim is correct. Mock and unavailable sources contribute
    no evidence.
    """
    research = state.get("research_data") or {}
    analysis_parts: list[str] = []

    price_data = research.get("current_price")
    if isinstance(price_data, dict) and not price_data.get("error"):
        symbol = price_data.get("symbol")
        price = price_data.get("price")
        daily_change = price_data.get("change_24h")
        if isinstance(symbol, str) and _valid_metric(price) and price > 0:
            snapshot = f"Provider-reported {symbol} price snapshot: ${price:,.2f}"
            if _valid_metric(daily_change):
                snapshot += f" (reported 24-hour change {daily_change:+.2f}%)"
            analysis_parts.append(snapshot + ". Snapshot freshness has not been independently verified.")

    indicators = research.get("indicators")
    if isinstance(indicators, dict) and not indicators.get("error"):
        readings: list[str] = []
        rsi = indicators.get("rsi_14")
        if _valid_metric(rsi) and 0 <= rsi <= 100:
            label = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
            readings.append(f"RSI(14) {rsi:.1f} ({label})")
        macd = indicators.get("macd")
        if isinstance(macd, dict):
            histogram = macd.get("histogram")
            if _valid_metric(histogram):
                label = "positive" if histogram > 0 else "negative" if histogram < 0 else "flat"
                readings.append(f"MACD histogram {histogram:.2f} ({label})")
        for period in ("sma_20", "sma_50"):
            sma = indicators.get(period)
            if _valid_metric(sma) and sma > 0:
                readings.append(f"{period.upper()} {sma:,.2f}")
        if readings:
            analysis_parts.append("Calculated technical indicators: " + "; ".join(readings) + ".")

    # The on-chain whale integration is currently a string-valued placeholder.
    # Do not turn placeholder data into fabricated transfer counts or flow claims.
    whale = research.get("whale_data")
    if isinstance(whale, dict):
        flows = whale.get("net_exchange_flow")
        count = whale.get("large_transfers_24h")
        if _valid_metric(flows) and _valid_metric(count) and count >= 0:
            analysis_parts.append(
                f"Provided on-chain transfer observations: {int(count)} large transfers "
                f"and net exchange flow {flows:+.2f} (units/source not independently verified)."
            )

    sentiment = research.get("sentiment")
    if isinstance(sentiment, dict) and not sentiment.get("error"):
        sentiment_value = sentiment.get("overall")
        news_count = sentiment.get("news_count")
        if _valid_metric(sentiment_value) and _valid_metric(news_count) and news_count > 0:
            label = (
                "positive" if sentiment_value > 0.3 else
                "negative" if sentiment_value < -0.3 else "mixed/neutral"
            )
            analysis_parts.append(
                f"A simple headline-keyword sentiment heuristic is {label} "
                f"({sentiment_value:+.2f}; {int(news_count)} headlines); "
                "this is not a validated market prediction."
            )

    final_analysis = (
        " ".join(analysis_parts)
        if analysis_parts
        else "Unable to generate a supported market analysis from available provider data."
    )
    # Percentage of the four source categories supplying usable measurements,
    # NOT a model-prediction or investment confidence score.
    evidence_coverage = round(len(analysis_parts) / 4.0, 2)
    log.info("analyst_completed", evidence_coverage=evidence_coverage)

    return {
        **state,
        "analysis": final_analysis,
        "confidence": evidence_coverage,
        "messages": (state.get("messages") or [])
        + [{"role": "assistant", "content": final_analysis}],
    }


# ============================================
# GRAPH BUILDER
# ============================================


def create_agent_graph():
    """Create the LangGraph agent workflow."""
    if not LANGGRAPH_AVAILABLE:
        log.error("langgraph_not_available")
        return None

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", END)

    # Compile the graph
    return workflow.compile()


# ============================================
# MAIN ENTRY POINT
# ============================================


async def run_agent(query: str) -> dict:
    """
    Run the multi-agent system on a user query.

    Args:
        query: User's market intelligence question

    Returns:
        Analysis result with confidence score
    """
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "plan": None,
        "research_data": None,
        "analysis": None,
        "confidence": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if LANGGRAPH_AVAILABLE:
        graph = create_agent_graph()
        if graph:
            result = await graph.ainvoke(initial_state)
            return {
                "analysis": result.get("analysis", ""),
                "confidence": result.get("confidence", 0),
                "plan": result.get("plan", []),
                "timestamp": result.get("timestamp"),
            }

    # Fallback if LangGraph not available
    state = await planner_node(initial_state)
    state = await researcher_node(state)
    state = await analyst_node(state)

    return {
        "analysis": state.get("analysis", ""),
        "confidence": state.get("confidence", 0),
        "plan": state.get("plan", []),
        "timestamp": state.get("timestamp"),
    }
