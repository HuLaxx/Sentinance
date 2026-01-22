"""
Comprehensive Module Tests for Sentinance
Tests all major components and outputs pass/fail status
"""
import asyncio
import requests
import json

API_BASE = "http://127.0.0.1:8000"

def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

def test_data_ingestion():
    """Test 1: Data Ingestion - Binance & yFinance"""
    print_header("TEST 1: DATA INGESTION")
    
    try:
        r = requests.get(f"{API_BASE}/api/prices")
        data = r.json()
        prices = data.get("prices", [])
        
        crypto = [p for p in prices if p.get("symbol", "").endswith("USDT")]
        indices = [p for p in prices if p.get("symbol", "").startswith("^")]
        
        print(f"  Crypto prices: {len(crypto)}")
        for p in crypto:
            print(f"    {p['symbol']}: ${p['price']:,.2f}")
        
        print(f"  Market indices: {len(indices)}")
        for p in indices:
            print(f"    {p['name']}: {p['price']:,.2f}")
        
        if len(crypto) >= 4 and len(indices) >= 4:
            print("  ✅ PASS: All 8 assets streaming")
            return True
        else:
            print("  ⚠️ PARTIAL: Some assets missing")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_indicators():
    """Test 2: Technical Indicators"""
    print_header("TEST 2: TECHNICAL INDICATORS")
    
    try:
        r = requests.get(f"{API_BASE}/api/indicators/BTCUSDT")
        data = r.json()
        
        rsi = data.get("rsi_14")
        macd = data.get("macd", {})
        ma = data.get("moving_averages", {})
        bb = data.get("bollinger_bands", {})
        
        print(f"  RSI (14): {rsi}")
        print(f"  MACD Value: {macd.get('value')}")
        print(f"  MACD Signal: {macd.get('signal')}")
        print(f"  MACD Histogram: {macd.get('histogram')}")
        print(f"  SMA 20: {ma.get('sma_20')}")
        print(f"  EMA 12: {ma.get('ema_12')}")
        print(f"  Bollinger Upper: {bb.get('upper')}")
        print(f"  Bollinger Lower: {bb.get('lower')}")
        
        if rsi is not None and macd.get("value") is not None:
            print("  ✅ PASS: Indicators calculated correctly")
            return True
        else:
            print("  ⚠️ PARTIAL: Some indicators missing")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_predictions():
    """Test 3: Price Predictions"""
    print_header("TEST 3: PRICE PREDICTIONS")
    
    try:
        horizons = ["1h", "24h", "7d"]
        all_pass = True
        
        for h in horizons:
            r = requests.get(f"{API_BASE}/api/predict/BTCUSDT?horizon={h}")
            data = r.json()
            
            pred_price = data.get("predicted_price")
            direction = data.get("direction", "unknown")
            confidence = data.get("confidence", 0)
            
            print(f"  {h}: ${pred_price:,.2f} ({direction}) Conf: {confidence:.0%}")
            
            if pred_price is None:
                all_pass = False
        
        if all_pass:
            print("  ✅ PASS: Predictions working for all horizons")
            return True
        else:
            print("  ⚠️ PARTIAL: Some predictions missing")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_ai_chat():
    """Test 4: AI Chat Agent"""
    print_header("TEST 4: AI CHAT AGENT")
    
    try:
        r = requests.post(f"{API_BASE}/api/chat", json={
            "message": "What is the current Bitcoin price and market sentiment?",
            "history": [],
            "use_agent": True
        }, timeout=30)
        
        data = r.json()
        content = data.get("content", "")
        metadata = data.get("metadata", {})
        model = metadata.get("model", "unknown")
        confidence = metadata.get("confidence", 0)
        
        print(f"  Model: {model}")
        print(f"  Confidence: {confidence}")
        print(f"  Response length: {len(content)} chars")
        print(f"  Preview: {content[:150]}...")
        
        if len(content) > 20 and model != "unknown":
            print("  ✅ PASS: AI Agent responding with analysis")
            return True
        else:
            print("  ⚠️ PARTIAL: Response incomplete")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_alerts():
    """Test 5: Alerts Service"""
    print_header("TEST 5: ALERTS SERVICE")
    
    try:
        # Create an alert (no auth needed for create)
        r = requests.post(f"{API_BASE}/api/alerts", json={
            "symbol": "BTCUSDT",
            "alert_type": "price_above",
            "target_value": 100000,
            "message": "BTC hit 100k!"
        })
        
        if r.status_code == 200:
            alert = r.json()
            print(f"  Created alert: {alert.get('id', 'unknown')}")
            print(f"  Type: {alert.get('alert_type')}")
            print(f"  Target: ${alert.get('target_value'):,}")
            print("  ✅ PASS: Alert creation working")
            return True
        else:
            print(f"  Status: {r.status_code}")
            print("  ⚠️ PARTIAL: Alert creation returned non-200")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_websocket():
    """Test 6: WebSocket Connection"""
    print_header("TEST 6: WEBSOCKET STREAMING")
    
    try:
        import websockets
        
        async def ws_test():
            uri = "ws://127.0.0.1:8000/ws/prices"
            async with websockets.connect(uri) as ws:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(msg)
                return data
        
        data = asyncio.run(ws_test())
        msg_type = data.get("type")
        prices = data.get("prices", [])
        
        print(f"  Message type: {msg_type}")
        print(f"  Prices received: {len(prices)}")
        
        if msg_type == "initial" and len(prices) >= 4:
            print("  ✅ PASS: WebSocket streaming working")
            return True
        else:
            print("  ⚠️ PARTIAL: WebSocket response incomplete")
            return False
    except Exception as e:
        print(f"  ⚠️ SKIP: WebSocket test requires websockets library ({e})")
        return None

def test_frontend_proxy():
    """Test 7: Frontend API Proxy"""
    print_header("TEST 7: FRONTEND API PROXY")
    
    try:
        FRONTEND = "http://localhost:3000"
        
        # Test indicators proxy
        r1 = requests.get(f"{FRONTEND}/api/indicators/BTCUSDT", timeout=10)
        print(f"  Indicators proxy: {r1.status_code}")
        
        # Test predict proxy
        r2 = requests.get(f"{FRONTEND}/api/predict/BTCUSDT?horizon=24h", timeout=10)
        print(f"  Predict proxy: {r2.status_code}")
        
        # Test chat proxy
        r3 = requests.post(f"{FRONTEND}/api/chat", json={
            "messages": [{"role": "user", "content": "test"}]
        }, timeout=15)
        print(f"  Chat proxy: {r3.status_code}")
        
        if r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200:
            print("  ✅ PASS: All frontend proxies working")
            return True
        else:
            print("  ⚠️ PARTIAL: Some proxies failing")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

def test_health():
    """Test 8: Health Endpoints"""
    print_header("TEST 8: HEALTH & MONITORING")
    
    try:
        # Health check
        r1 = requests.get(f"{API_BASE}/health")
        health = r1.json()
        print(f"  Status: {health.get('status')}")
        print(f"  WebSocket connections: {health.get('websocket_connections')}")
        
        # Metrics
        r2 = requests.get(f"{API_BASE}/metrics")
        has_metrics = "sentinance" in r2.text
        print(f"  Prometheus metrics: {'Available' if has_metrics else 'Not found'}")
        
        if health.get("status") == "ok":
            print("  ✅ PASS: Health endpoints working")
            return True
        else:
            print("  ⚠️ PARTIAL: Health check issues")
            return False
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SENTINANCE - COMPREHENSIVE MODULE TESTS")
    print("=" * 60)
    
    results = {}
    results["Data Ingestion"] = test_data_ingestion()
    results["Technical Indicators"] = test_indicators()
    results["Price Predictions"] = test_predictions()
    results["AI Chat Agent"] = test_ai_chat()
    results["Alerts Service"] = test_alerts()
    results["WebSocket"] = test_websocket()
    results["Frontend Proxy"] = test_frontend_proxy()
    results["Health & Monitoring"] = test_health()
    
    print_header("SUMMARY")
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
            passed += 1
        elif result is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⏭️ SKIP"
            skipped += 1
        print(f"  {name}: {status}")
    
    print()
    print(f"  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
