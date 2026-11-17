"""Explicit, bounded Binance aggregate-trade ingest command.

No network call or producer starts on import. Live exchange use is off by
default, requires operator-supplied cursor and two explicit authorization
switches. This is a *single bounded batch*, not an always-on daemon.
"""

import argparse
import asyncio
import os

import httpx

from kafka_producer import PriceProducer
from market_event_sources import BinanceAggregateTradeSource, publish_bounded_pages


def configuration() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--from-id", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1)
    return parser.parse_args()


async def run(
    *,
    symbol: str,
    from_id: int,
    page_size: int,
    max_pages: int,
) -> int:
    if os.getenv("SENTINANCE_BINANCE_LIVE_INGEST_ENABLED") != "1":
        raise RuntimeError("Live Binance adapter is disabled by default")
    if os.getenv("SENTINANCE_MARKET_DATA_RIGHTS_CONFIRMED") != "1":
        raise RuntimeError("Verify your market-data permissions before live use")

    # No Kafka connection, credentials or HTTP request until validation passes.
    # Operator-supplied source IDs are mandatory: never infer a cursor from time.
    producer = PriceProducer()
    async with httpx.AsyncClient(follow_redirects=False) as client:
        source = BinanceAggregateTradeSource(client)
        await producer.start()
        try:
            return await publish_bounded_pages(
                source=source,
                producer=producer,
                symbol=symbol,
                from_id=from_id,
                page_size=page_size,
                max_pages=max_pages,
            )
        finally:
            await producer.stop()


if __name__ == "__main__":
    options = configuration()
    next_id = asyncio.run(
        run(
            symbol=options.symbol,
            from_id=options.from_id,
            page_size=options.page_size,
            max_pages=options.max_pages,
        )
    )
    print(f"Next source cursor (not persisted): {next_id}")
