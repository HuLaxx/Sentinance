"""UTC database boundary and legacy-schema regression tests (local SQLite)."""

from datetime import UTC, datetime, timedelta, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from database import Base, normalize_async_database_url
from db_models import Alert, AuditLog, Prediction, Price, User
from db_time import db_utc_to_aware, utc_now_db


def test_database_now_is_explicitly_naive_utc():
    before = datetime.now(UTC).replace(tzinfo=None)
    created = utc_now_db()
    after = datetime.now(UTC).replace(tzinfo=None)
    assert created.tzinfo is None
    assert before <= created <= after


def test_utc_timestamp_converts_to_explicit_api_utc_without_local_timezone():
    source = datetime(2026, 11, 13, 12, 30)
    assert db_utc_to_aware(source) == source.replace(tzinfo=UTC)

    offset_time = datetime(2026, 11, 13, 7, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert db_utc_to_aware(offset_time).utcoffset() == timedelta(0)
    assert db_utc_to_aware(offset_time).hour == 12


def test_existing_db_columns_are_explicitly_timestamp_without_timezone():
    columns = [
        User.__table__.c.created_at,
        User.__table__.c.updated_at,
        Alert.__table__.c.created_at,
        Alert.__table__.c.triggered_at,
        Price.__table__.c.timestamp,
        Prediction.__table__.c.created_at,
        AuditLog.__table__.c.created_at,
    ]
    assert all(col.type.timezone is False for col in columns)
    # SQLAlchemy adapts Python defaults to a context-taking wrapper.
    # Inspect its wrapped function rather than assuming identity of the wrapper.
    assert all(
        col.default is None
        or getattr(col.default.arg, "__wrapped__", col.default.arg) is utc_now_db
        for col in columns
    )
    updater = User.__table__.c.updated_at.onupdate.arg
    assert getattr(updater, "__wrapped__", updater) is utc_now_db


def test_legacy_naive_utc_database_round_trip_without_migration():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            user = User(
                id="u-utc-1",
                email="utc-contract@example.test",
                hashed_password="not-a-real-password-hash",
            )
            price = Price(symbol="BTCUSDT", price=125.0, volume=3.0)
            session.add_all([user, price])
            session.commit()

        with Session(engine) as session:
            stored_user = session.scalars(select(User)).one()
            stored_price = session.scalars(select(Price)).one()
            assert stored_user.created_at.tzinfo is None
            assert stored_price.timestamp.tzinfo is None
            assert db_utc_to_aware(stored_price.timestamp).utcoffset() == timedelta(0)
    finally:
        engine.dispose()


def test_database_url_normalizes_legacy_postgres_schemes():
    source = "postgresql://service:example@localhost:5432/sentinance_test"
    expected = "postgresql+asyncpg://service:example@localhost:5432/sentinance_test"
    assert normalize_async_database_url(source) == expected
    assert (
        normalize_async_database_url(source.replace("postgresql://", "postgres://"))
        == expected
    )
    assert normalize_async_database_url(expected) == expected


def test_database_url_rejects_sync_or_unknown_drivers():
    import pytest

    with pytest.raises(ValueError, match="requires a PostgreSQL URL"):
        normalize_async_database_url("postgresql+psycopg2://localhost/sentinance")
