"""Explicit timestamp contract for legacy SQLAlchemy timestamp-without-time-zone columns.

The current database models use SQLAlchemy DateTime(timezone=False), mapped
to PostgreSQL TIMESTAMP WITHOUT TIME ZONE. Their historical values are UTC
wall-clock timestamps without tzinfo. Do not silently switch these columns
to aware datetimes: that needs an audited data migration and round-trip
verification against PostgreSQL.
"""

from datetime import UTC, datetime


def utc_now_db() -> datetime:
    """Return naive UTC *only* for the existing naive-UTC database schema."""
    return datetime.now(UTC).replace(tzinfo=None)


def db_utc_to_aware(value: datetime) -> datetime:
    """Interpret a DB UTC-naive timestamp as UTC for API/presentation use.

    Supports already-aware datetime objects without losing their offset.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
