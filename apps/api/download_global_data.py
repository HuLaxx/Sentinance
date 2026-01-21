"""
Download Global Market Data

Downloads historical data for:
- Crypto: BTC, ETH, SOL, BNB
- USA: S&P 500, NASDAQ, DOW JONES
- India: NIFTY 50, SENSEX
- Japan: NIKKEI 225
- UK: FTSE 100
- Europe: DAX (Germany), CAC 40 (France)

Uses Yahoo Finance API for stocks/indices
Uses Binance API for crypto

Run: python download_global_data.py
"""

import os
import json
from datetime import datetime
from typing import List, Dict

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

# Yahoo Finance symbols for global indices
# https://finance.yahoo.com/
STOCK_SYMBOLS = {
    # USA
    "^GSPC": {"name": "S&P 500", "country": "USA", "type": "index"},
    "^IXIC": {"name": "NASDAQ Composite", "country": "USA", "type": "index"},
    "^DJI": {"name": "Dow Jones", "country": "USA", "type": "index"},
    
    # India
    "^NSEI": {"name": "NIFTY 50", "country": "India", "type": "index"},
    "^BSESN": {"name": "SENSEX", "country": "India", "type": "index"},
    
    # Japan
    "^N225": {"name": "NIKKEI 225", "country": "Japan", "type": "index"},
    
    # UK
    "^FTSE": {"name": "FTSE 100", "country": "UK", "type": "index"},
    
    # Europe
    "^GDAXI": {"name": "DAX", "country": "Germany", "type": "index"},
    "^FCHI": {"name": "CAC 40", "country": "France", "type": "index"},
    
    # China
    "000001.SS": {"name": "Shanghai Composite", "country": "China", "type": "index"},
    
    # Major ETFs (for comparison)
    "SPY": {"name": "S&P 500 ETF", "country": "USA", "type": "etf"},
    "QQQ": {"name": "NASDAQ 100 ETF", "country": "USA", "type": "etf"},
}

# Crypto from Binance (already downloaded)
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


# ============================================
# DOWNLOAD FUNCTIONS
# ============================================

def download_stock_data(symbol: str, info: dict, years: int = 15) -> dict:
    """Download stock/index data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return None
    
    print(f"\n📥 Downloading {info['name']} ({symbol})...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Download max history
        df = ticker.history(period="max")
        
        if df.empty:
            print(f"   ⚠️ No data available for {symbol}")
            return None
        
        # Convert to list of dicts
        data = []
        for idx, row in df.iterrows():
            data.append({
                "timestamp": int(idx.timestamp() * 1000),
                "datetime": idx.isoformat(),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]) if row["Volume"] > 0 else 0,
            })
        
        # Calculate summary
        if data:
            start_date = datetime.fromisoformat(data[0]["datetime"].replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(data[-1]["datetime"].replace("Z", "+00:00"))
            prices = [d["close"] for d in data]
            
            summary = {
                "symbol": symbol,
                "name": info["name"],
                "country": info["country"],
                "type": info["type"],
                "start_date": data[0]["datetime"],
                "end_date": data[-1]["datetime"],
                "total_candles": len(data),
                "total_years": round((end_date - start_date).days / 365, 1),
                "price_min": min(prices),
                "price_max": max(prices),
                "price_current": prices[-1],
            }
            
            print(f"   ✅ {len(data):,} days ({summary['total_years']} years)")
            print(f"   📅 {data[0]['datetime'][:10]} to {data[-1]['datetime'][:10]}")
            print(f"   💰 ${summary['price_min']:,.2f} - ${summary['price_max']:,.2f}")
            
            return {"data": data, "summary": summary}
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None
    
    return None


def save_data(symbol: str, data: list, output_dir: str):
    """Save data to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean symbol for filename
    clean_symbol = symbol.replace("^", "").replace(".", "_").lower()
    filepath = os.path.join(output_dir, f"{clean_symbol}_historical.csv")
    
    with open(filepath, "w") as f:
        f.write("timestamp,datetime,open,high,low,close,volume\n")
        for d in data:
            f.write(f"{d['timestamp']},{d['datetime']},{d['open']},{d['high']},{d['low']},{d['close']},{d['volume']}\n")
    
    size_kb = os.path.getsize(filepath) / 1024
    print(f"   💾 Saved: {filepath} ({size_kb:.1f} KB)")
    
    return filepath


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("🌍 SENTINANCE GLOBAL MARKET DATA DOWNLOADER")
    print("=" * 70)
    
    print("\n📊 Assets to download:")
    print("   • Crypto: BTC, ETH, SOL, BNB (already have from Binance)")
    print("   • USA: S&P 500, NASDAQ, Dow Jones")
    print("   • India: NIFTY 50, SENSEX")
    print("   • Japan: NIKKEI 225")
    print("   • UK: FTSE 100")
    print("   • Europe: DAX, CAC 40")
    print("   • China: Shanghai Composite")
    
    all_summaries = []
    
    # Download stock indices
    print("\n" + "=" * 70)
    print("📈 DOWNLOADING GLOBAL INDICES")
    print("=" * 70)
    
    for symbol, info in STOCK_SYMBOLS.items():
        result = download_stock_data(symbol, info)
        
        if result:
            save_data(symbol, result["data"], OUTPUT_DIR)
            all_summaries.append(result["summary"])
    
    # Check existing crypto data
    print("\n" + "=" * 70)
    print("₿ CRYPTO DATA (already downloaded)")
    print("=" * 70)
    
    for symbol in CRYPTO_SYMBOLS:
        filepath = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_historical.csv")
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   ✅ {symbol}: {size_mb:.2f} MB")
        else:
            print(f"   ⚠️ {symbol}: Not found (run download_data.py first)")
    
    # Save all summaries
    all_summary_path = os.path.join(OUTPUT_DIR, "global_data_summary.json")
    with open(all_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\n📈 INDICES DOWNLOADED:")
    for s in sorted(all_summaries, key=lambda x: x["country"]):
        print(f"   {s['country']:12} | {s['name']:25} | {s['total_years']:5.1f} years | {s['total_candles']:,} days")
    
    print("\n" + "=" * 70)
    print(f"✅ Data saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 70)
    
    # Calculate totals
    total_candles = sum(s["total_candles"] for s in all_summaries)
    print(f"\n📊 TOTAL: {len(all_summaries)} indices + 4 crypto = {total_candles:,} data points")


if __name__ == "__main__":
    main()
