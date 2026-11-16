"""Durable raw Kafka price-event ledger for at-least-once consumers.

Uniqueness by Kafka position protects broker redelivery. Source event identity
also deduplicates retries published at different Kafka offsets.
"""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import JSON, BigInteger, DateTime, Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.schema import UniqueConstraint

from database import Base


class RawKafkaPriceEvent(Base):
    __tablename__ = "raw_kafka_price_events"

    topic: Mapped[str] = mapped_column(String(200), primary_key=True)
    partition: Mapped[int] = mapped_column(Integer, primary_key=True)
    offset: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    price: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    raw_event: Mapped[dict] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        UniqueConstraint(
            "source", "source_event_id", name="uq_kafka_price_source_event"
        ),
        Index("ix_raw_kafka_price_symbol_time", "symbol", "observed_at"),
    )
