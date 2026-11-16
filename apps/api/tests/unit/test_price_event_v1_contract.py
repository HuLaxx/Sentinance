"""Offline schema contract: market payload quality and source event identity."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from kafka_producer import PriceProducer
from price_event_contract import (
    PRICE_V1_TOPIC,
    make_price_v1,
    validate_price_v1,
)


def fixture():
    return make_price_v1(
        source="fixture_exchange",
        event_id="stable-tick-0001",
        symbol="BTCUSDT",
        observed_at=datetime(2026, 11, 16, 10, 0, tzinfo=UTC),
        price="102.123456789012",
        volume="0",
    )


def test_v1_preserves_exact_decimal_and_explicit_utc_source_time():
    event = fixture()
    decoded = validate_price_v1(event)
    assert decoded.price == Decimal("102.123456789012")
    assert decoded.volume == Decimal("0")
    assert decoded.observed_at.utcoffset() == timedelta(0)
    assert event["event_id"] == "stable-tick-0001"
    assert event["schema_version"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"timestamp": "2026-11-16T10:00:00"},
        {"event_id": ""},
        {"source": "SOURCE WITH SPACES"},
        {"symbol": "btc/usdt"},
        {"data": {"price": "NaN", "volume": "1"}},
        {"data": {"price": "-1", "volume": "1"}},
        {"data": {"price": "1e100", "volume": "1"}},
        {"data": {"price": "1.0000000000001", "volume": "1"}},
        {"data": {"price": "1", "volume": "-1"}},
        {"data": {"price": True, "volume": "0"}},
    ],
)
def test_v1_rejects_malformed_event_or_unsafe_values(change):
    event = fixture()
    event.update(change)
    with pytest.raises((ValueError, TypeError)):
        validate_price_v1(event)


def test_v1_accepts_offset_time_but_normalizes_to_utc():
    event = fixture()
    event["timestamp"] = "2026-11-16T15:30:00+05:30"
    assert validate_price_v1(event).observed_at == datetime(
        2026, 11, 16, 10, 0, tzinfo=UTC
    )


@pytest.mark.asyncio
async def test_v1_broker_ack_and_explicit_source_identity():
    producer = PriceProducer()
    broker = AsyncMock()
    producer.producer = broker
    producer._started = True
    await producer.send_price_v1(
        source="fixture_exchange",
        event_id="origin-tick-123",
        symbol="BTCUSDT",
        observed_at=datetime(2026, 11, 16, tzinfo=UTC),
        price="101.25",
    )
    broker.send_and_wait.assert_awaited_once()
    args = broker.send_and_wait.await_args
    assert args.args == (PRICE_V1_TOPIC,)
    assert args.kwargs["key"] == "fixture_exchange:BTCUSDT"
    assert args.kwargs["value"]["event_id"] == "origin-tick-123"


@pytest.mark.asyncio
async def test_invalid_event_is_rejected_before_any_broker_send():
    producer = PriceProducer()
    broker = AsyncMock()
    producer.producer = broker
    producer._started = True
    with pytest.raises(ValueError):
        await producer.send_price_v1(
            source="fixture_exchange",
            event_id="tick-invalid",
            symbol="BTCUSDT",
            observed_at=datetime(2026, 11, 16, tzinfo=UTC),
            price="Infinity",
        )
    broker.send_and_wait.assert_not_awaited()
