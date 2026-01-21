"""
LangGraph AI Agent System

Multi-agent system for market intelligence:
- Planner: Breaks down queries into sub-tasks
- Researcher: Gathers market data
- Analyst: Synthesizes insights
"""

from typing import TypedDict, Annotated, Sequence, Literal, Optional
from datetime import datetime
import structlog
import asyncio

log = structlog.get_logger()

# Check if langgraph is available
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    log.warning("langgraph_not_installed", message="Install with: pip install langgraph")


# ============================================
# STATE DEFINITION
# ============================================

class AgentState(TypedDict):
    """State shared between all agent nodes."""
    query: str
    messages: Annotated[list, add_messages] if LANGGRAPH_AVAILABLE else list
    plan: Optional[list[str]]
    research_data: Optional[dict]
    analysis: Optional[str]
    confidence: Optional[float]
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
    
    if "technical" in query or "indicator" in query or "rsi" in query or "macd" in query:
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
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Plan created: {plan}"}
        ],
    }


async def researcher_node(state: AgentState) -> AgentState:
    """
    Researcher Agent: Executes the plan by calling tools.
    In production, this would call actual APIs.
    """
    plan = state.get("plan", [])
    research_data = {}
    
    # Extract symbol from query (simple extraction)
    query = state["query"].upper()
    symbol = "BTCUSDT"  # Default
    for s in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA"]:
        if s in query:
            symbol = f"{s}USDT"
            break
    
    # Execute each tool in the plan
    for tool in plan:
        if tool == "get_current_price":
            # Mock data (replace with actual API call)
            research_data["current_price"] = {
                "symbol": symbol,
                "price": 95234.56,
                "change_24h": 2.45,
                "volume": 142500000000,
            }
        
        elif tool == "analyze_whale_movements":
            research_data["whale_data"] = {
                "large_transfers_24h": 12,
                "net_exchange_flow": -450.5,  # Negative = moving off exchanges
                "top_holder_change": 0.02,
            }
        
        elif tool == "get_order_book_depth":
            research_data["order_book"] = {
                "bid_depth_1pct": 15000000,
                "ask_depth_1pct": 12000000,
                "spread": 0.01,
            }
        
        elif tool == "calculate_technical_indicators":
            research_data["indicators"] = {
                "rsi_14": 62.5,
                "macd": {"value": 150.2, "signal": 145.8, "histogram": 4.4},
                "ma_50": 92500,
                "ma_200": 88000,
            }
        
        elif tool == "get_news_sentiment":
            research_data["sentiment"] = {
                "overall": 0.72,  # -1 to 1
                "news_count_24h": 47,
                "top_keywords": ["ETF", "institutional", "adoption"],
            }
        
        elif tool == "get_price_history":
            research_data["price_history"] = {
                "high_24h": 96500,
                "low_24h": 93200,
                "high_7d": 98000,
                "low_7d": 91500,
            }
    
    log.info("researcher_completed", data_keys=list(research_data.keys()))
    
    return {
        **state,
        "research_data": research_data,
        "messages": state.get("messages", []) + [
            {"role": "system", "content": f"Research completed: {list(research_data.keys())}"}
        ],
    }


async def analyst_node(state: AgentState) -> AgentState:
    """
    Analyst Agent: Synthesizes research into actionable insights.
    In production, this would use an LLM for natural language generation.
    """
    research = state.get("research_data", {})
    query = state["query"]
    
    # Build analysis (replace with LLM in production)
    analysis_parts = []
    confidence = 0.0
    confidence_factors = 0
    
    if "current_price" in research:
        price_data = research["current_price"]
        direction = "up" if price_data["change_24h"] > 0 else "down"
        analysis_parts.append(
            f"{price_data['symbol'].replace('USDT', '')} is currently trading at "
            f"${price_data['price']:,.2f}, {direction} {abs(price_data['change_24h']):.2f}% "
            f"in the last 24 hours."
        )
        confidence += 0.9
        confidence_factors += 1
    
    if "indicators" in research:
        indicators = research["indicators"]
        rsi = indicators["rsi_14"]
        rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        macd_signal = "bullish" if indicators["macd"]["histogram"] > 0 else "bearish"
        
        analysis_parts.append(
            f"Technical analysis shows RSI at {rsi:.1f} ({rsi_signal}) with a "
            f"{macd_signal} MACD crossover. Price is trading above both 50-day and 200-day MAs, "
            f"indicating a strong uptrend."
        )
        confidence += 0.8
        confidence_factors += 1
    
    if "whale_data" in research:
        whale = research["whale_data"]
        flow_direction = "outflow" if whale["net_exchange_flow"] < 0 else "inflow"
        analysis_parts.append(
            f"Whale activity shows {whale['large_transfers_24h']} large transfers in 24h. "
            f"Net exchange {flow_direction} of {abs(whale['net_exchange_flow']):.1f} BTC "
            f"suggests {'accumulation' if flow_direction == 'outflow' else 'potential selling pressure'}."
        )
        confidence += 0.7
        confidence_factors += 1
    
    if "sentiment" in research:
        sentiment = research["sentiment"]
        sentiment_label = "bullish" if sentiment["overall"] > 0.3 else "bearish" if sentiment["overall"] < -0.3 else "neutral"
        analysis_parts.append(
            f"Market sentiment is {sentiment_label} ({sentiment['overall']:.2f}). "
            f"Key themes in {sentiment['news_count_24h']} news articles: {', '.join(sentiment['top_keywords'])}."
        )
        confidence += 0.75
        confidence_factors += 1
    
    final_analysis = " ".join(analysis_parts) if analysis_parts else "Unable to generate analysis with available data."
    final_confidence = (confidence / confidence_factors) if confidence_factors > 0 else 0.5
    
    log.info("analyst_completed", confidence=final_confidence)
    
    return {
        **state,
        "analysis": final_analysis,
        "confidence": final_confidence,
        "messages": state.get("messages", []) + [
            {"role": "assistant", "content": final_analysis}
        ],
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
    initial_state = {
        "query": query,
        "messages": [],
        "plan": None,
        "research_data": None,
        "analysis": None,
        "confidence": None,
        "timestamp": datetime.utcnow().isoformat(),
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
