"""Bounded, source-identifiable aggregate-trade ingestion (no synthetic IDs).

Binance Spot aggTrades has aggregate IDs ('a'), price ('p'), quantity ('q')
and trade time milliseconds ('T'). This is *trade* ingestion, not a 24h
price snapshot. Live access requires operator opt-in; CI uses fixtures only.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from kafka_producer import PriceProducer
from price_event_contract import make_price_v1

SOURCE = "binance_aggtrades"
BASE_URL = "https://data-api.binance.vision/api/v3/aggTrades"
MAX_PAGE_SIZE = 1000
MAX_BATCH_PAGES = 10


class SourceGapError(ValueError):
    """An expected upstream aggregate-trade ID was absent in a response."""


@dataclass(frozen=True)
class TradePage:
    records: tuple[dict[str, Any], ...]
    next_from_id: int
    complete: bool


def parse_trade(
    raw: Any, *, symbol: str, expected_id: int
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("Aggregate trade must be an object")
    trade_id = raw.get("a")
    milliseconds = raw.get("T")
    if type(trade_id) is not int or trade_id < 0:
        raise ValueError("Aggregate trade ID must be a nonnegative integer")
    if trade_id != expected_id:
        raise SourceGapError(
            f"Aggregate trade ID gap: expected {expected_id}, got {trade_id}"
        )
    if type(milliseconds) is not int or milliseconds < 0:
        raise ValueError("Exchange trade time must be nonnegative milliseconds")
    try:
        observed_at = datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("Exchange trade time is out of range") from exc
    return make_price_v1(
        source=SOURCE,
        event_id=str(trade_id),
        symbol=symbol,
        observed_at=observed_at,
        price=raw.get("p"),
        volume=raw.get("q"),
    )


def validate_page(
    payload: Any, *, symbol: str, from_id: int, page_size: int
) -> TradePage:
    if type(from_id) is not int or from_id < 0:
        raise ValueError("from_id must be a nonnegative integer")
    if type(page_size) is not int or not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError("page_size must be between 1 and 1000")
    if not isinstance(payload, list):
        raise TypeError("Exchange response must be a JSON array")
    if len(payload) > page_size:
        raise ValueError("Provider returned more records than requested")
    result: list[dict[str, Any]] = []
    next_id = from_id
    for raw in payload:
        envelope = parse_trade(raw, symbol=symbol, expected_id=next_id)
        result.append(envelope)
        next_id += 1
    # This is a bounded cursor over *received contiguous* provider IDs,
    # not a durable watermark or a claim that future IDs are already known.
    return TradePage(tuple(result), next_id, len(result) < page_size)


class BinanceAggregateTradeSource:
    """Read-only public endpoint, bounded pages, no credential transmission."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def fetch_page(
        self, *, symbol: str, from_id: int, page_size: int = 100
    ) -> TradePage:
        # Validate inputs before any HTTP request to prevent URL injection
        # and accidentally expensive unbounded API calls.
        validate_page([], symbol=symbol, from_id=from_id, page_size=page_size)
        if not symbol.isascii() or not symbol.isalnum() or not symbol.isupper():
            raise ValueError("Market symbol must be uppercase alphanumeric")
        if not 2 <= len(symbol) <= 20:
            raise ValueError("Market symbol must have 2-20 characters")
        response = await self.client.get(
            BASE_URL,
            params={"symbol": symbol, "fromId": from_id, "limit": page_size},
            timeout=8.0,
        )
        response.raise_for_status()  # rate limits/403/5xx fail visibly
        return validate_page(
            response.json(), symbol=symbol, from_id=from_id, page_size=page_size
        )


async def publish_bounded_pages(
    *,
    source: BinanceAggregateTradeSource,
    producer: PriceProducer,
    symbol: str,
    from_id: int,
    page_size: int = 100,
    max_pages: int = 1,
) -> int:
    """Publish each contiguous record before advancing an in-memory cursor.

    On Kafka failure, propagate immediately; callers must retry from the
    *original* persisted/explicit cursor, not from an inferred checkpoint.
    Same source-symbol event ID makes retries idempotent at PostgreSQL sink.
    """
    if type(max_pages) is not int or not 1 <= max_pages <= MAX_BATCH_PAGES:
        raise ValueError("max_pages must be between 1 and 10")
    cursor = from_id
    for _ in range(max_pages):
        page = await source.fetch_page(
            symbol=symbol, from_id=cursor, page_size=page_size
        )
        for event in page.records:
            await producer.send_price_v1(
                source=event["source"],
                event_id=event["event_id"],
                symbol=event["symbol"],
                observed_at=datetime.fromisoformat(event["timestamp"]),
                price=event["data"]["price"],
                volume=event["data"]["volume"],
            )
        cursor = page.next_from_id
        if page.complete:
            break
    return cursor
