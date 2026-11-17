"""Explicit one-page live aggregate-trade ingestion, never started implicitly.

Requires a conscious operator opt-in and source trade ID. No API keys are
needed for this public endpoint, but regional access and terms still apply.
"""

import argparse
import asyncio
import os

import httpx

from binance_aggtrades_v1 import ALLOWED_SYMBOLS, BinanceAggregateTrades
from kafka_producer import PriceProducer


async def run(symbol: str, from_id: int, limit: int) -> int:
    if os.getenv("SENTINANCE_BINANCE_LIVE_ENABLED") != "1":
        raise RuntimeError(
            "Live Binance requests are disabled. Review provider terms, "
            "geography and rate limits, then explicitly set "
            "SENTINANCE_BINANCE_LIVE_ENABLED=1"
        )

    async with httpx.AsyncClient(timeout=8.0) as client:
        adapter = BinanceAggregateTrades(client, mode="provider")
        producer = PriceProducer()
        await producer.start()
        try:
            return await adapter.publish_page(
                producer, symbol, from_id=from_id, limit=limit
            )
        finally:
            await producer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opt-in one-page Binance aggregate trade Kafka ingestion"
    )
    parser.add_argument("--symbol", required=True, choices=sorted(ALLOWED_SYMBOLS))
    parser.add_argument("--from-id", type=int, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    total = asyncio.run(run(args.symbol, args.from_id, args.limit))
    print(f"Broker-acknowledged aggregate trade events: {total}")


if __name__ == "__main__":
    main()
