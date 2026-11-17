"""Read-only provenance-first view of durably persisted market-price events.

This serves the PostgreSQL raw event ledger, not Redis's in-memory prices.
No event is inferred to be 'live' just because a provider label is present.
"""

from datetime import UTC, datetime, timedelta
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from db_time import db_utc_to_aware
from kafka_price_models import RawKafkaPriceEvent

router = APIRouter(tags=["Market data evidence"])
log = structlog.get_logger()


def format_market_event(row: RawKafkaPriceEvent, now: datetime) -> dict:
    metadata = row.raw_event.get("provenance", {})
    mode = metadata.get("mode") if isinstance(metadata, dict) else None
    source_mode = mode if mode in ("fixture", "provider") else "unverified"

    observed = db_utc_to_aware(row.observed_at)
    receipt = db_utc_to_aware(row.received_at)
    age = now - observed
    if age < timedelta(seconds=-60):
        freshness = "future_timestamp"
    elif age > timedelta(minutes=5):
        freshness = "stale"
    else:
        freshness = "recent"

    return {
        "source": row.source,
        "source_mode": source_mode,
        "symbol": row.symbol,
        "source_event_id": row.source_event_id,
        "schema_version": row.schema_version,
        "observed_at": observed.isoformat(),
        "received_at": receipt.isoformat(),
        "price": str(row.price),
        "volume": str(row.volume),
        "freshness": freshness,
        "provenance_status": "verified_schema_not_provider_uptime",
    }


@router.get("/api/v1/market-events")
async def market_events(
    session: AsyncSession = Depends(get_db),
    symbol: str = Query(
        "BTCUSDT", min_length=2, max_length=20, regex=r"^[A-Z0-9]+$"
    ),
    limit: int = Query(20, ge=1, le=50),
) -> dict:
    """Return a bounded latest-event slice with explicit origin and staleness."""
    try:
        rows = (
            await session.scalars(
                select(RawKafkaPriceEvent)
                .where(RawKafkaPriceEvent.symbol == symbol)
                .order_by(
                    RawKafkaPriceEvent.observed_at.desc(),
                    RawKafkaPriceEvent.offset.desc(),
                )
                .limit(limit)
            )
        ).all()
    except SQLAlchemyError as exc:
        log.error("market_event_ledger_read_failed", error=type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Durable market-event ledger unavailable"
        ) from exc

    now = datetime.now(UTC)
    result = [format_market_event(row, now) for row in rows]
    return {
        "schema_version": 1,
        "status": "stored_events" if result else "no_data",
        "query": {"symbol": symbol, "limit": limit},
        "as_of": now.isoformat(),
        "events": result,
        "live_feed_verified": False,
        "warning": (
            "Stored events only; fixture/provider origin is explicit. "
            "Recent timestamps do not prove active exchange connectivity."
        ),
    }
