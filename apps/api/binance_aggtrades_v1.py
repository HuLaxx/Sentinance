"""Opt-in Binance aggregate-trade source for the versioned Kafka price ledger.

Uses Binance's aggregate trade ID and trade time from the provider response,
never a locally invented event ID. No live network request runs in CI.
A full page is parsed and gap-checked before publishing any event.
"""

import asyncio
import random
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx

from kafka_producer import PriceProducer
from price_event_contract import make_price_v1

BINANCE_AGGTRADES_URL = "https://api.binance.com/api/v3/aggTrades"
ALLOWED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"})
SourceMode = Literal["fixture", "provider"]


class MarketSourceGap(ValueError):
    """Provider trade IDs have an unexpected gap, duplication or ordering."""


class ProviderRateLimit(RuntimeError):
    """Provider blocked or rate-limited; do not automatically hammer it."""


class BinanceAggregateTrades:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        mode: SourceMode = "fixture",
        max_attempts: int = 3,
        retry_base_seconds: float = 0.2,
    ):
        if mode not in ("fixture", "provider"):
            raise ValueError("Unrecognized market-data provenance mode")
        if not 1 <= max_attempts <= 4:
            raise ValueError("Maximum source fetch attempts must be between 1 and 4")
        if not 0 < retry_base_seconds <= 3:
            raise ValueError("Invalid source retry backoff")
        self.client = client
        self.mode = mode
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds

    async def _pause(self, attempt: int) -> None:
        delay = min(
            self.retry_base_seconds * (2**attempt) + random.uniform(0, 0.1),
            3.0,
        )
        await asyncio.sleep(delay)

    async def fetch_page(
        self, symbol: str, *, from_id: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetch one bounded provider page; fail closed on gaps/HTTP outages."""
        if symbol not in ALLOWED_SYMBOLS:
            raise ValueError("Unapproved Binance market symbol")
        if isinstance(from_id, bool) or not isinstance(from_id, int) or from_id < 0:
            raise ValueError("from_id must be a nonnegative aggregate-trade ID")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError("limit must be between 1 and 1000")

        for attempt in range(self.max_attempts):
            try:
                response = await self.client.get(
                    BINANCE_AGGTRADES_URL,
                    params={"symbol": symbol, "fromId": from_id, "limit": limit},
                    timeout=8.0,
                )
            except httpx.TransportError:
                if attempt + 1 == self.max_attempts:
                    raise
                await self._pause(attempt)
                continue
            if response.status_code in (418, 429):
                raise ProviderRateLimit(
                    f"Binance returned HTTP {response.status_code}; stop ingestion"
                )
            if response.status_code >= 500:
                if attempt + 1 == self.max_attempts:
                    response.raise_for_status()
                await self._pause(attempt)
                continue
            # Other 4xx failures are permanent for this request, not retried.
            response.raise_for_status()
            return self._parse_page(response.json(), symbol, from_id, limit)
        raise RuntimeError("Unreachable source retry state")

    def _parse_page(
        self, response: Any, symbol: str, from_id: int, limit: int
    ) -> list[dict[str, Any]]:
        if not isinstance(response, list) or len(response) > limit:
            raise ValueError("Malformed or oversized aggregate-trade response")
        results: list[dict[str, Any]] = []
        expected_id = from_id
        for raw in response:
            if not isinstance(raw, dict):
                raise TypeError("Aggregate trade must be an object")
            agg_id = raw.get("a")
            timestamp_ms = raw.get("T")
            if isinstance(agg_id, bool) or not isinstance(agg_id, int) or agg_id < 0:
                raise ValueError("Aggregate trade ID must be a nonnegative integer")
            if (
                isinstance(timestamp_ms, bool)
                or not isinstance(timestamp_ms, int)
                or timestamp_ms < 0
            ):
                raise ValueError("Provider trade time must be epoch milliseconds")
            if agg_id != expected_id:
                raise MarketSourceGap(
                    f"{symbol} expected aggregate trade {expected_id}, received {agg_id}"
                )
            expected_id += 1
            try:
                source_time = datetime.fromtimestamp(timestamp_ms // 1000, UTC)
                source_time += timedelta(milliseconds=timestamp_ms % 1000)
            except (OverflowError, OSError, ValueError) as exc:
                raise ValueError("Provider trade time outside supported range") from exc
            price = raw.get("p")
            quantity = raw.get("q")
            if not isinstance(price, str) or not isinstance(quantity, str):
                raise TypeError("Provider price and quantity must be decimal strings")

            event = make_price_v1(
                source="binance_aggtrade",
                event_id=str(agg_id),
                symbol=symbol,
                observed_at=source_time,
                price=price,
                volume=quantity,
            )
            # Preserve the source-level representation and truthfully identify
            # fixtures used for CI versus an explicitly enabled API request.
            event["provenance"] = {
                "mode": self.mode,
                "provider": "binance",
                "endpoint": "/api/v3/aggTrades",
                "aggregate_trade_id": agg_id,
                "first_trade_id": raw.get("f"),
                "last_trade_id": raw.get("l"),
                "buyer_is_maker": raw.get("m"),
                "timestamp_unit": "milliseconds",
            }
            results.append(event)
        return results

    async def publish_page(
        self,
        producer: PriceProducer,
        symbol: str,
        *,
        from_id: int,
        limit: int = 100,
    ) -> int:
        """Publish validated page; return only after broker acknowledges each.

        A mid-page broker outage can leave a published prefix. A subsequent
        retry with the SAME provider IDs is safe at the PostgreSQL sink.
        """
        events = await self.fetch_page(symbol, from_id=from_id, limit=limit)
        for event in events:
            await producer.send_price_event_v1(event)
        return len(events)
