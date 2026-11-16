"""PostgreSQL-backed idempotence and failed-validation contracts.

Ordinary CI supplies a disposable PostgreSQL service. Local runs without a
database may skip; GitHub CI must fail rather than silently skip the test.
"""

import os
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError

from database import async_session, engine
from durable_price_ingestion import DurablePriceSink
from kafka_price_models import RawKafkaPriceEvent
from price_event_contract import PRICE_V1_TOPIC, make_price_v1


async def require_db() -> None:
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SELECT 1"))
            # For local development we can bootstrap this isolated table;
            # the real broker + PostgreSQL CI separately verifies Alembic.
            await connection.run_sync(RawKafkaPriceEvent.__table__.create, checkfirst=True)
    except (SQLAlchemyError, OSError, TimeoutError) as exc:
        if os.getenv("CI"):
            raise AssertionError("Disposable CI PostgreSQL must be available") from exc
        pytest.skip(f"PostgreSQL not configured for local test: {exc}")


def record(event: dict, offset: int):
    return SimpleNamespace(
        topic=PRICE_V1_TOPIC,
        partition=0,
        offset=offset,
        value=event,
    )


async def count_source(source_id: str) -> int:
    async with async_session() as session:
        result = await session.scalar(
            select(func.count())
            .select_from(RawKafkaPriceEvent)
            .where(RawKafkaPriceEvent.source_event_id == source_id)
        )
        return int(result or 0)


@pytest.mark.asyncio
async def test_postgres_retry_is_idempotent_and_detects_identity_collision():
    await require_db()
    source_id = "fixture-" + uuid4().hex
    price_event = make_price_v1(
        source="fixture_exchange",
        event_id=source_id,
        symbol="BTCUSDT",
        observed_at=datetime(2026, 11, 16, tzinfo=UTC),
        price="101.125",
        volume="2",
    )
    sink = DurablePriceSink()

    await sink.persist_record(record(price_event, 400))
    await sink.persist_record(record(price_event, 400))
    await sink.persist_record(record(price_event, 401))
    assert await count_source(source_id) == 1

    async with async_session() as session:
        saved = await session.scalar(
            select(RawKafkaPriceEvent).where(
                RawKafkaPriceEvent.source_event_id == source_id
            )
        )
        assert saved is not None
        assert saved.price == Decimal("101.125")
        assert saved.volume == Decimal("2")
        assert saved.observed_at.utcoffset() is not None

    invalid_retry = {
        **price_event,
        "data": {"price": "9000.0", "volume": "2"},
    }
    with pytest.raises(ValueError, match="different payload"):
        await sink.persist_record(record(invalid_retry, 402))
    assert await count_source(source_id) == 1


@pytest.mark.asyncio
async def test_postgres_does_not_persist_invalid_events():
    await require_db()
    source_id = "invalid-" + uuid4().hex
    event = make_price_v1(
        source="fixture_exchange",
        event_id=source_id,
        symbol="BTCUSDT",
        observed_at=datetime(2026, 11, 16, tzinfo=UTC),
        price="12",
    )
    event["data"]["price"] = "NaN"
    with pytest.raises(ValueError):
        await DurablePriceSink().persist_record(record(event, 1000))
    assert await count_source(source_id) == 0
