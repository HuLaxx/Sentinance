import asyncio
import websockets
import json

async def test():
    uri = "ws://127.0.0.1:8000/ws/prices"
    async with websockets.connect(uri) as ws:
        msg = await ws.recv()
        data = json.loads(msg)
        print("Message type:", data.get("type"))
        print("Number of prices:", len(data.get("prices", [])))
        for p in data.get("prices", [])[:3]:
            print(f"  {p.get('symbol')}: price={p.get('price')}, priceChangePercent={p.get('priceChangePercent')}")

asyncio.run(test())
