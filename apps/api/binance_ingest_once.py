"""Explicit bounded live aggregate-trade ingestion, never started implicitly.

Requires a conscious operator opt-in and source trade ID. No API keys are
needed for this public endpoint, but regional access and terms still apply.
"""

import argparse
import asyncio
import os
from pathlib import Path

import httpx

from binance_aggtrades_v1 import ALLOWED_SYMBOLS, BinanceAggregateTrades
from kafka_producer import PriceProducer
from source_checkpoint import LocalCheckpoint, ingest_checkpointed


async def run(
    symbol: str, from_id: int, limit: int, checkpoint_path: Path, max_pages: int
) -> int:
    if os.getenv("SENTINANCE_BINANCE_LIVE_ENABLED") != "1":
        raise RuntimeError(
            "Live Binance requests are disabled. Review provider terms, "
            "geography and rate limits, then explicitly set "
            "SENTINANCE_BINANCE_LIVE_ENABLED=1"
        )

    if os.getenv("SENTINANCE_MARKET_DATA_RIGHTS_CONFIRMED") != "1":
        raise RuntimeError("Verify your market-data permissions before live use")
    checkpoint = LocalCheckpoint(checkpoint_path)
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
        adapter = BinanceAggregateTrades(client, mode="provider")
        producer = PriceProducer()
        await producer.start()
        try:
            return await ingest_checkpointed(
                source=adapter,
                producer=producer,
                checkpoint=checkpoint,
                symbol=symbol,
                initial_id=from_id,
                limit=limit,
                max_pages=max_pages,
            )
        finally:
            await producer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opt-in checkpointed Binance aggregate trade Kafka ingestion"
    )
    parser.add_argument("--symbol", required=True, choices=sorted(ALLOWED_SYMBOLS))
    parser.add_argument("--from-id", type=int, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=1)
    args = parser.parse_args()
    cursor = asyncio.run(
        run(args.symbol, args.from_id, args.limit, args.checkpoint, args.max_pages)
    )
    print(f"Next broker-acknowledged source cursor: {cursor}")


if __name__ == "__main__":
    main()
