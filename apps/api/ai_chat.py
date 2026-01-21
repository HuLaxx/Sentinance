"""
AI Chat Service

Provides market intelligence through conversational AI.
Uses mock responses for now - will be replaced with LangGraph agent.
"""

import random
from datetime import datetime
from typing import List, Dict

# Mock AI responses based on keywords
ANALYSIS_TEMPLATES = {
    "bitcoin": [
        "Bitcoin is showing strong accumulation patterns. Whale wallets have increased holdings by 2.3% this week.",
        "BTC/USDT displays a bullish divergence on the 4H chart. RSI is at 45, suggesting room for upside.",
        "On-chain metrics indicate long-term holders are not selling. This is typically bullish for price."
    ],
    "ethereum": [
        "Ethereum gas fees have dropped 40% this week, indicating reduced network congestion.",
        "ETH staking deposits continue to rise. Currently 28M ETH staked (23% of supply).",
        "The ETH/BTC ratio is at 0.055, near yearly lows. Potential mean reversion opportunity."
    ],
    "manipulation": [
        "⚠️ ALERT: Unusual order book activity detected. Large sell walls at $45,500 suggest potential manipulation.",
        "Wash trading indicators are elevated. Volume-weighted analysis shows 15% suspicious transactions.",
        "No clear manipulation signals detected at current price levels. Market appears organic."
    ],
    "whale": [
        "🐋 Whale Alert: 3 wallets moved 12,500 BTC to exchanges in the last 24 hours. This could signal selling pressure.",
        "Whale accumulation continues. Top 100 wallets have increased holdings by 1.8% this month.",
        "Large OTC desk activity detected. Institutional buyers appear active in this range."
    ],
    "solana": [
        "Solana network TPS remains stable at 3,000. No congestion issues detected.",
        "SOL staking yield is at 7.2% APY. Validator count: 1,900 active.",
        "DeFi TVL on Solana has increased 15% this week. Growing ecosystem activity."
    ],
    "default": [
        "Based on current market conditions, I recommend monitoring key support and resistance levels.",
        "Market sentiment is neutral. Watch for volume confirmation before making decisions.",
        "Technical indicators are mixed. Consider waiting for a clearer setup."
    ]
}


def get_ai_response(message: str, prices: Dict[str, dict]) -> dict:
    """
    Generate an AI response based on the user's question.
    
    This is a mock implementation. In production, this would:
    1. Call LangGraph agent
    2. Use tool calls to fetch real data
    3. Generate response with Mistral/GPT-4
    """
    message_lower = message.lower()
    
    # Determine which template to use
    template_key = "default"
    for key in ANALYSIS_TEMPLATES:
        if key in message_lower:
            template_key = key
            break
    
    # Get random response from template
    responses = ANALYSIS_TEMPLATES[template_key]
    response_text = random.choice(responses)
    
    # Add current price context
    price_context = ""
    if "btc" in message_lower or "bitcoin" in message_lower:
        btc = prices.get("BTCUSDT", {})
        price_context = f"\n\n📊 Current BTC Price: ${btc.get('price', 0):,.2f} ({btc.get('change_24h', 0):+.2f}% 24h)"
    elif "eth" in message_lower or "ethereum" in message_lower:
        eth = prices.get("ETHUSDT", {})
        price_context = f"\n\n📊 Current ETH Price: ${eth.get('price', 0):,.2f} ({eth.get('change_24h', 0):+.2f}% 24h)"
    elif "sol" in message_lower or "solana" in message_lower:
        sol = prices.get("SOLUSDT", {})
        price_context = f"\n\n📊 Current SOL Price: ${sol.get('price', 0):,.2f} ({sol.get('change_24h', 0):+.2f}% 24h)"
    
    # Build response
    return {
        "role": "assistant",
        "content": response_text + price_context,
        "metadata": {
            "model": "sentinance-mock-v1",
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "tools_used": ["price_analysis", "on_chain_metrics"],
            "timestamp": datetime.now().isoformat()
        }
    }


def get_suggested_questions() -> List[str]:
    """Return suggested questions for the user."""
    return [
        "Is Bitcoin showing manipulation signals?",
        "What are whales doing with ETH?",
        "Give me a Solana market analysis",
        "Are there any unusual patterns today?",
        "What's the market sentiment for BTC?"
    ]
