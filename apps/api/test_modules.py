import asyncio
from exchange_connector import BinanceConnector, IndicesConnector

async def test_data_ingestion():
    print("=" * 60)
    print("DATA INGESTION TESTS")
    print("=" * 60)
    
    # Test 1: Binance Connector
    print("\n[TEST 1] Binance Connector (Crypto Prices)")
    bc = BinanceConnector()
    try:
        prices = await bc.get_prices()
        print(f"  ✅ PASS: {len(prices)} crypto prices fetched")
        for p in prices:
            symbol = p["symbol"]
            price = p["price"]
            change = p["priceChangePercent"]
            print(f"     {symbol}: ${price:,.2f} ({change:+.2f}%)")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
    finally:
        await bc.close()
    
    # Test 2: yFinance Connector
    print("\n[TEST 2] yFinance Connector (Market Indices)")
    ic = IndicesConnector()
    try:
        prices = await ic.get_prices()
        print(f"  ✅ PASS: {len(prices)} index prices fetched")
        for p in prices:
            name = p["name"]
            price = p["price"]
            currency = p["currency"]
            change = p["priceChangePercent"]
            print(f"     {name}: {price:,.2f} {currency} ({change:+.2f}%)")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_ingestion())
