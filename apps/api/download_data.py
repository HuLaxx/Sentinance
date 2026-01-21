"""
Download Full Historical Crypto Data

Downloads ALL available historical data from Binance API.
BTC/USDT available since August 2017 (~7 years of data).

Run: python download_data.py
"""

import os
import json
import httpx
from datetime import datetime, timedelta, timezone
import time

# ============================================
# CONFIGURATION
# ============================================

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
INTERVAL = "1h"  # Hourly data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

# Binance listing dates (approximate)
LISTING_DATES = {
    "BTCUSDT": "2017-08-17",  # BTC on Binance
    "ETHUSDT": "2017-08-17",  # ETH on Binance
    "BNBUSDT": "2017-11-06",  # BNB launched
    "SOLUSDT": "2020-08-11",  # SOL on Binance
}

# ============================================
# DOWNLOAD FUNCTION
# ============================================

def download_all_candles(symbol: str, start_date: str) -> list:
    """
    Download ALL candles from start_date to now.
    Handles pagination automatically.
    """
    url = "https://api.binance.com/api/v3/klines"
    
    # Convert start date to timestamp
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    all_candles = []
    current_start = start_ts
    
    print(f"\n📥 Downloading {symbol} from {start_date} to today...")
    print(f"   This may take a few minutes...")
    
    batch_count = 0
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": current_start,
            "limit": 1000,  # Max per request
        }
        
        try:
            response = httpx.get(url, params=params, timeout=30)
            response.raise_for_status()
            candles = response.json()
            
            if not candles:
                break
            
            all_candles.extend(candles)
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                current_date = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
                print(f"   Progress: {len(all_candles):,} candles (up to {current_date.date()})")
            
            # Move start to after last candle
            current_start = candles[-1][0] + 1
            
            # Rate limiting (respect Binance limits)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   Error at batch {batch_count}: {e}")
            time.sleep(1)
            continue
    
    print(f"   ✅ Downloaded {len(all_candles):,} candles for {symbol}")
    return all_candles


def save_to_csv(candles: list, symbol: str, output_dir: str):
    """Save candles to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{symbol.lower()}_historical.csv")
    
    with open(filepath, "w") as f:
        f.write("timestamp,datetime,open,high,low,close,volume\n")
        for c in candles:
            ts = c[0]
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            f.write(f"{ts},{dt},{c[1]},{c[2]},{c[3]},{c[4]},{c[5]}\n")
    
    # Get file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"   💾 Saved to {filepath} ({size_mb:.2f} MB)")
    
    return filepath


def save_to_json(candles: list, symbol: str, output_dir: str):
    """Save candles to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{symbol.lower()}_historical.json")
    
    data = []
    for c in candles:
        data.append({
            "timestamp": c[0],
            "datetime": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).isoformat(),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })
    
    with open(filepath, "w") as f:
        json.dump(data, f)
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"   💾 Saved to {filepath} ({size_mb:.2f} MB)")
    
    return filepath


def get_data_summary(candles: list, symbol: str) -> dict:
    """Get summary statistics for downloaded data."""
    if not candles:
        return {}
    
    prices = [float(c[4]) for c in candles]  # Close prices
    volumes = [float(c[5]) for c in candles]
    
    start_date = datetime.fromtimestamp(candles[0][0] / 1000, tz=timezone.utc)
    end_date = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
    
    return {
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_candles": len(candles),
        "total_days": (end_date - start_date).days,
        "total_years": round((end_date - start_date).days / 365, 1),
        "price_min": min(prices),
        "price_max": max(prices),
        "price_current": prices[-1],
        "total_volume": sum(volumes),
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("🗄️  SENTINANCE HISTORICAL DATA DOWNLOADER")
    print("=" * 60)
    print(f"\nDownloading data for: {', '.join(SYMBOLS)}")
    print(f"Interval: {INTERVAL}")
    print(f"Output directory: {OUTPUT_DIR}/")
    
    all_summaries = []
    
    for symbol in SYMBOLS:
        start_date = LISTING_DATES.get(symbol, "2020-01-01")
        
        # Download
        candles = download_all_candles(symbol, start_date)
        
        if candles:
            # Save
            save_to_csv(candles, symbol, OUTPUT_DIR)
            
            # Summary
            summary = get_data_summary(candles, symbol)
            all_summaries.append(summary)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    
    for s in all_summaries:
        print(f"\n{s['symbol']}:")
        print(f"  📅 Date Range: {s['start_date'][:10]} to {s['end_date'][:10]}")
        print(f"  📊 Total Candles: {s['total_candles']:,}")
        print(f"  📆 Total Years: {s['total_years']}")
        print(f"  💰 Price Range: ${s['price_min']:,.2f} - ${s['price_max']:,.2f}")
        print(f"  💵 Current Price: ${s['price_current']:,.2f}")
    
    # Save summary
    with open(os.path.join(OUTPUT_DIR, "data_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()
