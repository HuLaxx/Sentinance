"""
Exchange & Index Connector - Multi-Asset Price Fetcher

Fetches real-time prices from:
- Binance API (crypto)
- yFinance (global indices)

Supports:
- Crypto: BTC, ETH, SOL, XRP
- Indices: S&P 500 (US), Nifty 50 (India), FTSE 100 (UK), Nikkei 225 (Japan)
"""

import asyncio
import httpx
from typing import Dict, List, Optional
from datetime import datetime
import structlog

log = structlog.get_logger()

# ============================================
# BINANCE API (CRYPTO)
# ============================================
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TICKER_24H = "/api/v3/ticker/24hr"

# Crypto symbols (as specified by user)
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

CRYPTO_NAMES = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "SOLUSDT": "Solana",
    "XRPUSDT": "XRP",
}

# ============================================
# YAHOO FINANCE (GLOBAL INDICES)
# ============================================
# yFinance symbols for major indices
INDEX_SYMBOLS = {
    "^GSPC": {"name": "S&P 500", "region": "USA", "currency": "USD"},
    "^NSEI": {"name": "Nifty 50", "region": "India", "currency": "INR"},
    "^FTSE": {"name": "FTSE 100", "region": "UK", "currency": "GBP"},
    "^N225": {"name": "Nikkei 225", "region": "Japan", "currency": "JPY"},
}


class BinanceConnector:
    """Fetches crypto prices from Binance public API."""
    
    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or CRYPTO_SYMBOLS
        self.client: Optional[httpx.AsyncClient] = None
        self._last_prices: Dict[str, dict] = {}
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=BINANCE_BASE_URL,
                timeout=10.0
            )
        return self.client
    
    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def get_24h_ticker(self, symbol: str) -> dict:
        """Get 24-hour price statistics for a crypto symbol."""
        try:
            client = await self._get_client()
            response = await client.get(BINANCE_TICKER_24H, params={"symbol": symbol})
            response.raise_for_status()
            
            data = response.json()
            return {
                "symbol": data["symbol"],
                "price": float(data["lastPrice"]),
                "priceChangePercent": float(data["priceChangePercent"]),
                "volume": float(data["volume"]),
                "high": float(data["highPrice"]),
                "low": float(data["lowPrice"]),
                "name": CRYPTO_NAMES.get(data["symbol"], data["symbol"]),
                "asset_type": "crypto",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            log.error("binance_fetch_error", symbol=symbol, error=str(e))
            raise
    
    async def get_prices(self) -> List[dict]:
        """Fetch current prices for all crypto symbols."""
        tasks = [self.get_24h_ticker(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = []
        for result in results:
            if isinstance(result, Exception):
                log.warning("crypto_price_fetch_failed", error=str(result))
                continue
            prices.append(result)
            self._last_prices[result["symbol"]] = result
        
        return prices


class IndicesConnector:
    """Fetches global index prices via yFinance."""
    
    def __init__(self):
        self._last_prices: Dict[str, dict] = {}
    
    async def get_index_price(self, symbol: str) -> dict:
        """Fetch current price for a stock index using yFinance with timeout."""
        try:
            import yfinance as yf
            
            # yFinance is synchronous, so we run in thread pool
            def fetch():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if hist.empty:
                    raise ValueError(f"No data for {symbol}")
                
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change_pct = ((current - prev) / prev) * 100 if prev else 0
                
                return {
                    "symbol": symbol,
                    "price": round(current, 2),
                    "priceChangePercent": round(change_pct, 2),
                    "high": round(hist['High'].iloc[-1], 2),
                    "low": round(hist['Low'].iloc[-1], 2),
                    "volume": int(hist['Volume'].iloc[-1]) if hist['Volume'].iloc[-1] else 0,
                    "name": INDEX_SYMBOLS[symbol]["name"],
                    "region": INDEX_SYMBOLS[symbol]["region"],
                    "currency": INDEX_SYMBOLS[symbol]["currency"],
                    "asset_type": "index",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Add 5-second timeout to prevent slow API calls
            result = await asyncio.wait_for(
                asyncio.to_thread(fetch),
                timeout=5.0
            )
            return result
        
        except asyncio.TimeoutError:
            log.warning("index_fetch_timeout", symbol=symbol)
            # Return cached data if available
            if symbol in self._last_prices:
                return self._last_prices[symbol]
            raise ValueError(f"Timeout fetching {symbol}")
            
        except Exception as e:
            log.error("index_fetch_error", symbol=symbol, error=str(e))
            raise
    
    async def get_prices(self) -> List[dict]:
        """Fetch current prices for all configured indices."""
        tasks = [self.get_index_price(symbol) for symbol in INDEX_SYMBOLS.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = []
        for result in results:
            if isinstance(result, Exception):
                log.warning("index_price_fetch_failed", error=str(result))
                continue
            prices.append(result)
            self._last_prices[result["symbol"]] = result
        
        return prices


class UnifiedConnector:
    """
    Unified connector that fetches both crypto and global indices.
    
    Usage:
        connector = UnifiedConnector()
        prices = await connector.get_all_prices()
    """
    
    def __init__(self):
        self.binance = BinanceConnector()
        self.indices = IndicesConnector()
    
    async def close(self):
        await self.binance.close()
    
    async def get_crypto_prices(self) -> List[dict]:
        """Get only crypto prices."""
        return await self.binance.get_prices()
    
    async def get_index_prices(self) -> List[dict]:
        """Get only index prices."""
        return await self.indices.get_prices()
    
    async def get_all_prices(self) -> List[dict]:
        """Get all prices (crypto + indices)."""
        crypto_task = self.binance.get_prices()
        index_task = self.indices.get_prices()
        
        crypto_prices, index_prices = await asyncio.gather(
            crypto_task, index_task, return_exceptions=True
        )
        
        all_prices = []
        
        if not isinstance(crypto_prices, Exception):
            all_prices.extend(crypto_prices)
        else:
            log.error("crypto_fetch_error", error=str(crypto_prices))
        
        if not isinstance(index_prices, Exception):
            all_prices.extend(index_prices)
        else:
            log.error("index_fetch_error", error=str(index_prices))
        
        return all_prices
    
    async def get_prices(self) -> List[dict]:
        """Alias for get_all_prices() - for backward compatibility."""
        return await self.get_all_prices()


# ============================================
# GLOBAL INSTANCES
# ============================================
_connector: Optional[UnifiedConnector] = None


def get_connector() -> UnifiedConnector:
    """Get or create the global unified connector."""
    global _connector
    if _connector is None:
        _connector = UnifiedConnector()
    return _connector


async def fetch_live_prices() -> List[dict]:
    """Convenience function to fetch all live prices."""
    connector = get_connector()
    return await connector.get_all_prices()


# For testing
if __name__ == "__main__":
    async def main():
        connector = UnifiedConnector()
        try:
            print("=== CRYPTO ===")
            crypto = await connector.get_crypto_prices()
            for p in crypto:
                print(f"{p['name']}: ${p['price']:,.2f} ({p['priceChangePercent']:+.2f}%)")
            
            print("\n=== GLOBAL INDICES ===")
            indices = await connector.get_index_prices()
            for p in indices:
                print(f"{p['name']} ({p['region']}): {p['price']:,.2f} {p['currency']} ({p['priceChangePercent']:+.2f}%)")
        finally:
            await connector.close()
    
    asyncio.run(main())
