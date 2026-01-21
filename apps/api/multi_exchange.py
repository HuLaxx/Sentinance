"""
Multi-Exchange Connector

Fetches real-time prices from multiple exchanges:
- Binance
- Coinbase
- Kraken (public API)
"""

import asyncio
import httpx
import structlog
from typing import Optional
from datetime import datetime

log = structlog.get_logger()

# ============================================
# SUPPORTED EXCHANGES
# ============================================

EXCHANGES = {
    "binance": {
        "name": "Binance",
        "endpoint": "https://api.binance.com/api/v3/ticker/24hr",
        "enabled": True,
    },
    "coinbase": {
        "name": "Coinbase",
        "endpoint": "https://api.exchange.coinbase.com/products",
        "enabled": True,
    },
    "kraken": {
        "name": "Kraken",
        "endpoint": "https://api.kraken.com/0/public/Ticker",
        "enabled": True,
    },
}

# Symbol mapping across exchanges
SYMBOL_MAP = {
    "BTCUSDT": {
        "binance": "BTCUSDT",
        "coinbase": "BTC-USD",
        "kraken": "XXBTZUSD",
    },
    "ETHUSDT": {
        "binance": "ETHUSDT",
        "coinbase": "ETH-USD",
        "kraken": "XETHZUSD",
    },
    "SOLUSDT": {
        "binance": "SOLUSDT",
        "coinbase": "SOL-USD",
        "kraken": "SOLUSD",
    },
}

CRYPTO_NAMES = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "SOLUSDT": "Solana",
    "BNBUSDT": "BNB",
    "XRPUSDT": "XRP",
    "ADAUSDT": "Cardano",
    "DOGEUSDT": "Dogecoin",
    "MATICUSDT": "Polygon",
    "DOTUSDT": "Polkadot",
    "AVAXUSDT": "Avalanche",
}


# ============================================
# EXCHANGE FETCHERS
# ============================================

async def fetch_binance_prices(client: httpx.AsyncClient) -> dict[str, dict]:
    """Fetch prices from Binance"""
    try:
        # Fetch 24hr ticker for all symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", 
                   "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT"]
        
        prices = {}
        for symbol in symbols:
            try:
                response = await client.get(
                    f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    prices[symbol] = {
                        "exchange": "binance",
                        "symbol": symbol,
                        "name": CRYPTO_NAMES.get(symbol, symbol),
                        "price": float(data["lastPrice"]),
                        "change_24h": float(data["priceChangePercent"]),
                        "volume": float(data["volume"]),
                        "high": float(data["highPrice"]),
                        "low": float(data["lowPrice"]),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception as e:
                log.warning("binance_symbol_fetch_failed", symbol=symbol, error=str(e))
        
        return prices
    except Exception as e:
        log.error("binance_fetch_failed", error=str(e))
        return {}


async def fetch_coinbase_prices(client: httpx.AsyncClient) -> dict[str, dict]:
    """Fetch prices from Coinbase"""
    try:
        symbols = {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT", 
            "SOL-USD": "SOLUSDT",
        }
        
        prices = {}
        for cb_symbol, unified_symbol in symbols.items():
            try:
                # Get ticker
                response = await client.get(
                    f"https://api.exchange.coinbase.com/products/{cb_symbol}/ticker",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    
                    # Get 24hr stats
                    stats_response = await client.get(
                        f"https://api.exchange.coinbase.com/products/{cb_symbol}/stats",
                        timeout=5.0
                    )
                    stats = stats_response.json() if stats_response.status_code == 200 else {}
                    
                    price = float(data.get("price", 0))
                    open_price = float(stats.get("open", price))
                    change_24h = ((price - open_price) / open_price * 100) if open_price > 0 else 0
                    
                    prices[unified_symbol] = {
                        "exchange": "coinbase",
                        "symbol": unified_symbol,
                        "name": CRYPTO_NAMES.get(unified_symbol, unified_symbol),
                        "price": price,
                        "change_24h": round(change_24h, 2),
                        "volume": float(stats.get("volume", 0)),
                        "high": float(stats.get("high", price)),
                        "low": float(stats.get("low", price)),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
            except Exception as e:
                log.warning("coinbase_symbol_fetch_failed", symbol=cb_symbol, error=str(e))
        
        return prices
    except Exception as e:
        log.error("coinbase_fetch_failed", error=str(e))
        return {}


async def fetch_kraken_prices(client: httpx.AsyncClient) -> dict[str, dict]:
    """Fetch prices from Kraken"""
    try:
        kraken_symbols = {
            "XXBTZUSD": "BTCUSDT",
            "XETHZUSD": "ETHUSDT",
        }
        
        pairs = ",".join(kraken_symbols.keys())
        response = await client.get(
            f"https://api.kraken.com/0/public/Ticker?pair={pairs}",
            timeout=5.0
        )
        
        prices = {}
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                for kraken_symbol, unified_symbol in kraken_symbols.items():
                    if kraken_symbol in data["result"]:
                        ticker = data["result"][kraken_symbol]
                        price = float(ticker["c"][0])  # Last trade closed price
                        open_price = float(ticker["o"])
                        change_24h = ((price - open_price) / open_price * 100) if open_price > 0 else 0
                        
                        prices[unified_symbol] = {
                            "exchange": "kraken",
                            "symbol": unified_symbol,
                            "name": CRYPTO_NAMES.get(unified_symbol, unified_symbol),
                            "price": price,
                            "change_24h": round(change_24h, 2),
                            "volume": float(ticker["v"][1]),  # 24h volume
                            "high": float(ticker["h"][1]),  # 24h high
                            "low": float(ticker["l"][1]),  # 24h low
                            "timestamp": datetime.utcnow().isoformat(),
                        }
        
        return prices
    except Exception as e:
        log.error("kraken_fetch_failed", error=str(e))
        return {}


# ============================================
# AGGREGATED FETCH
# ============================================

async def fetch_all_exchanges() -> tuple[dict[str, dict], dict[str, list]]:
    """
    Fetch prices from all enabled exchanges.
    Returns:
        - aggregated_prices: Best price per symbol
        - exchange_data: All prices per exchange for comparison
    """
    async with httpx.AsyncClient() as client:
        # Fetch from all exchanges concurrently
        results = await asyncio.gather(
            fetch_binance_prices(client),
            fetch_coinbase_prices(client),
            fetch_kraken_prices(client),
            return_exceptions=True
        )
        
        binance_prices = results[0] if isinstance(results[0], dict) else {}
        coinbase_prices = results[1] if isinstance(results[1], dict) else {}
        kraken_prices = results[2] if isinstance(results[2], dict) else {}
        
        # Aggregate - use Binance as primary (most liquid)
        aggregated = {}
        exchange_data = {
            "binance": [],
            "coinbase": [],
            "kraken": [],
        }
        
        # Merge all symbols
        all_symbols = set(binance_prices.keys()) | set(coinbase_prices.keys()) | set(kraken_prices.keys())
        
        for symbol in all_symbols:
            # Prefer Binance, fallback to others
            if symbol in binance_prices:
                aggregated[symbol] = binance_prices[symbol]
                exchange_data["binance"].append(binance_prices[symbol])
            elif symbol in coinbase_prices:
                aggregated[symbol] = coinbase_prices[symbol]
                exchange_data["coinbase"].append(coinbase_prices[symbol])
            elif symbol in kraken_prices:
                aggregated[symbol] = kraken_prices[symbol]
                exchange_data["kraken"].append(kraken_prices[symbol])
            
            # Track all exchange prices for the symbol
            if symbol in coinbase_prices:
                exchange_data["coinbase"].append(coinbase_prices[symbol])
            if symbol in kraken_prices:
                exchange_data["kraken"].append(kraken_prices[symbol])
        
        log.info("multi_exchange_fetch_complete", 
                 binance=len(binance_prices),
                 coinbase=len(coinbase_prices),
                 kraken=len(kraken_prices))
        
        return aggregated, exchange_data


# ============================================
# VWAP CALCULATION (Volume-Weighted Average Price)
# ============================================

def calculate_vwap(prices: list[dict]) -> Optional[float]:
    """
    Calculate Volume-Weighted Average Price across exchanges.
    VWAP = Σ(Price × Volume) / Σ(Volume)
    """
    if not prices:
        return None
    
    total_volume = sum(p.get("volume", 0) for p in prices)
    if total_volume == 0:
        return prices[0]["price"]
    
    weighted_sum = sum(p.get("price", 0) * p.get("volume", 0) for p in prices)
    return weighted_sum / total_volume
