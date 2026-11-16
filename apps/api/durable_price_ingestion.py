"""PostgreSQL-first Kafka price ingestion with application-level idempotence.

A Kafka position is acknowledged only after PostgreSQL has committed. Events
replayed at another offset with the same source ID do not create duplicate rows.
This is not a two-phase transaction, Kafka exactly-once or outage backfill.
"""

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import async_session
from kafka_consumer import PriceConsumer
from kafka_price_models import RawKafkaPriceEvent
from price_event_contract import PRICE_V1_TOPIC, validate_price_v1

log = structlog.get_logger()


class DurablePriceSink:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession] = async_session
    ):
        self.session_factory = session_factory

    async def persist_record(self, record: Any) -> None:
        if record.topic != PRICE_V1_TOPIC:
            raise ValueError(f"Unexpected durable topic: {record.topic!r}")
        if (
            type(record.partition) is not int
            or type(record.offset) is not int
            or record.partition < 0
            or record.offset < 0
        ):
            raise ValueError("Kafka partition and offset must be nonnegative integers")
        event = validate_price_v1(record.value)

        async with self.session_factory() as session, session.begin():
            stmt = (
                insert(RawKafkaPriceEvent)
                .values(
                    topic=record.topic,
                    partition=record.partition,
                    offset=record.offset,
                    schema_version=1,
                    source=event.source,
                    source_event_id=event.source_event_id,
                    symbol=event.symbol,
                    observed_at=event.observed_at,
                    price=event.price,
                    volume=event.volume,
                    raw_event=event.raw,
                )
                .on_conflict_do_nothing()
                .returning(RawKafkaPriceEvent.offset)
            )
            inserted_offset = await session.scalar(stmt)
            if inserted_offset is None:
                # A conflict is safe only if the original event content
                # matches. Reusing IDs for different prices is a hard
                # data-contract violation, not an accepted duplicate.
                existing = await session.scalar(
                    select(RawKafkaPriceEvent).where(
                        RawKafkaPriceEvent.source == event.source,
                        RawKafkaPriceEvent.symbol == event.symbol,
                        RawKafkaPriceEvent.source_event_id == event.source_event_id,
                    )
                )
                if existing is None:
                    existing = await session.get(
                        RawKafkaPriceEvent,
                        (record.topic, record.partition, record.offset),
                    )
                if existing is None or existing.raw_event != event.raw:
                    raise ValueError(
                        "Kafka position or source event ID reused with a "
                        "different payload"
                    )
        # Transaction is committed successfully before emitting success logs
        # or allowing the caller to acknowledge the Kafka position.
        if inserted_offset is None:
            log.info(
                "price_event_duplicate_ignored",
                source=event.source,
                source_event_id=event.source_event_id,
            )
        else:
            log.info(
                "price_event_durably_stored",
                source=event.source,
                symbol=event.symbol,
                kafka_offset=record.offset,
            )


def create_durable_price_consumer(
    *,
    bootstrap_servers: str | None = None,
    group_id: str = "sentinance-durable-prices-v1",
    session_factory: async_sessionmaker[AsyncSession] = async_session,
) -> PriceConsumer:
    kwargs: dict[str, Any] = {
        "topics": [PRICE_V1_TOPIC],
        "group_id": group_id,
        "auto_offset_reset": "earliest",
    }
    if bootstrap_servers is not None:
        kwargs["bootstrap_servers"] = bootstrap_servers
    consumer = PriceConsumer(**kwargs)
    consumer.record_handler = DurablePriceSink(session_factory).persist_record
    return consumer
