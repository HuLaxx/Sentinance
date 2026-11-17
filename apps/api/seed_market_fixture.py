"""Explicit, offline market fixture publisher for the P1 review journey.

The request uses httpx.MockTransport and never reaches Binance's servers.
Fixture event IDs are deliberately isolated from real-provider source IDs.
"""

import asyncio
import json
import os
from pathlib import Path

import httpx

from binance_aggtrades_v1 import BinanceAggregateTrades
from kafka_producer import PriceProducer

FIXTURE = (
    Path(__file__).parent / "data" / "fixtures" / "synthetic_binance_aggtrades.json"
)


async def seed() -> int:
    if os.getenv("P1_FIXTURE_SEED_ENABLED") != "1":
        raise RuntimeError(
            "Fixture publisher is disabled; explicitly set "
            "P1_FIXTURE_SEED_ENABLED=1 for isolated test data"
        )
    source_rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("Synthetic trade fixture is empty or malformed")

    def respond(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path != "/api/v3/aggTrades"
            or request.url.params.get("symbol") != "BTCUSDT"
            or request.url.params.get("fromId") != "900001"
        ):
            raise ValueError("Unexpected synthetic provider request")
        return httpx.Response(200, json=source_rows)

    # MockTransport blocks all external network connections regardless of
    # whether the operator has live provider credentials or access.
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter = BinanceAggregateTrades(client, mode="fixture")
        producer = PriceProducer()
        await producer.start()
        try:
            return await adapter.publish_page(
                producer, "BTCUSDT", from_id=900001, limit=2
            )
        finally:
            await producer.stop()


if __name__ == "__main__":
    count = asyncio.run(seed())
    print(f"Broker-acknowledged synthetic fixture records: {count}")
