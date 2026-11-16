"""Versioned Kafka price-event envelope with explicit, finite market values.

This contract does not infer source timestamps or event identities.
Callers must supply a stable source-specific ID for application-level replay.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

PRICE_V1_TOPIC = "sentinance.prices.v1"
VALID_SYMBOL = re.compile(r"[A-Z0-9]{2,20}")
VALID_SOURCE = re.compile(r"[a-z][a-z0-9_-]{1,31}")


@dataclass(frozen=True)
class PriceEventV1:
    symbol: str
    source: str
    source_event_id: str
    observed_at: datetime
    price: Decimal
    volume: Decimal
    raw: dict[str, Any]


def _market_decimal(raw: Any, field: str, *, positive: bool) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float, Decimal)):
        raise TypeError(f"{field} must be a decimal string or number")
    try:
        number = Decimal(str(raw))
    except InvalidOperation as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not number.is_finite() or (number <= 0 if positive else number < 0):
        raise ValueError(f"{field} is outside the allowed nonnegative range")
    # Database NUMERIC(30,12) permits 18 integer digits and 12 decimals.
    exponent = number.as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError(f"{field} must be a finite decimal")
    if number.adjusted() >= 18 or exponent < -12:
        raise ValueError(f"{field} exceeds database precision NUMERIC(30,12)")
    return number


def validate_price_v1(raw: dict[str, Any]) -> PriceEventV1:
    if not isinstance(raw, dict) or raw.get("type") != "price_update":
        raise ValueError("Expected a price_update object")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Expected schema_version 1; no implicit schema migration")

    symbol = raw.get("symbol")
    source = raw.get("source")
    source_id = raw.get("event_id")
    if not isinstance(symbol, str) or VALID_SYMBOL.fullmatch(symbol) is None:
        raise ValueError("Invalid uppercase market symbol")
    if not isinstance(source, str) or VALID_SOURCE.fullmatch(source) is None:
        raise ValueError("Invalid price source identifier")
    if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 128:
        raise ValueError("A stable source_event_id of at most 128 chars is required")

    observed = raw.get("timestamp")
    if not isinstance(observed, str):
        raise TypeError("An explicit source event UTC timestamp is required")
    try:
        observed_at = datetime.fromisoformat(observed)
    except ValueError as exc:
        raise ValueError("Invalid ISO-8601 source event timestamp") from exc
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("Naive source event timestamps are forbidden")
    observed_at = observed_at.astimezone(UTC)

    data = raw.get("data")
    if not isinstance(data, dict):
        raise TypeError("Price event data must be an object")
    price = _market_decimal(data.get("price"), "price", positive=True)
    volume = _market_decimal(data.get("volume", "0"), "volume", positive=False)
    return PriceEventV1(
        symbol=symbol,
        source=source,
        source_event_id=source_id,
        observed_at=observed_at,
        price=price,
        volume=volume,
        raw=raw,
    )


def make_price_v1(
    *,
    source: str,
    event_id: str,
    symbol: str,
    observed_at: datetime,
    price: Decimal | str,
    volume: Decimal | str = "0",
) -> dict[str, Any]:
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("Observed price timestamp must be timezone aware")
    result: dict[str, Any] = {
        "type": "price_update",
        "schema_version": 1,
        "source": source,
        "event_id": event_id,
        "symbol": symbol,
        "timestamp": observed_at.astimezone(UTC).isoformat(),
        "data": {"price": str(price), "volume": str(volume)},
    }
    validate_price_v1(result)
    return result
