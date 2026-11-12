"""Evidence-only LangGraph analyst assertions (no provider calls required)."""
from datetime import datetime, timezone

import pytest

from agent import analyst_node


def state_with(research_data):
    return {
        "query": "What is happening in the market?",
        "messages": [],
        "plan": None,
        "research_data": research_data,
        "analysis": None,
        "confidence": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.mark.asyncio
async def test_unavailable_sources_do_not_generate_fake_confidence_or_values():
    result = await analyst_node(state_with({
        "current_price": {"symbol": "BTCUSDT", "error": "Provider down"},
        "indicators": {"error": "No data"},
        "whale_data": {
            "large_transfers_24h": "requires on-chain integration",
            "net_exchange_flow": "not available",
        },
        "sentiment": {"overall": 0, "news_count": 0},
    }))
    assert result["confidence"] == 0.0
    assert "Unable to generate a supported market analysis" in result["analysis"]
    assert "Whale activity shows" not in result["analysis"]


@pytest.mark.asyncio
async def test_numeric_indicators_do_not_imply_unmeasured_trends():
    result = await analyst_node(state_with({
        "indicators": {
            "rsi_14": 72.5,
            "macd": {"histogram": -0.4},
            "sma_20": 43000.0,
            "sma_50": None,
        }
    }))
    assert "RSI(14) 72.5" in result["analysis"]
    assert "MACD histogram -0.40" in result["analysis"]
    assert "200-day" not in result["analysis"]
    assert "crossover" not in result["analysis"]
    assert result["confidence"] == 0.25


@pytest.mark.asyncio
async def test_keyword_sentiment_uses_actual_research_schema():
    result = await analyst_node(state_with({
        "sentiment": {
            "overall": 0.6,
            "news_count": 5,
            "recent_headlines": ["Example"],
        }
    }))
    assert "5 headlines" in result["analysis"]
    assert "keyword" in result["analysis"]
    assert "validated market prediction" in result["analysis"]


@pytest.mark.asyncio
async def test_price_can_be_reported_without_inventing_missing_change():
    result = await analyst_node(state_with({
        "current_price": {"symbol": "ETHUSDT", "price": 2500.0},
    }))
    assert "ETHUSDT" in result["analysis"]
    assert "2500" in result["analysis"].replace(",", "")
    assert "24-hour change" not in result["analysis"]
    assert "freshness has not been independently verified" in result["analysis"]


@pytest.mark.asyncio
async def test_non_finite_market_metrics_are_rejected():
    result = await analyst_node(state_with({
        "current_price": {"symbol": "BTCUSDT", "price": float("nan")},
        "indicators": {"rsi_14": float("inf")},
    }))
    assert result["confidence"] == 0.0
