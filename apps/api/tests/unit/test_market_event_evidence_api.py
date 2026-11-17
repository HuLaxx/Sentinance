"""API evidence routes with synthetic sessions; no external provider calls."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from database import get_db
from kafka_price_models import RawKafkaPriceEvent
from routers.market_event_router import router


class FakeSession:
    def __init__(self, events=(), error=None):
        self.events = list(events)
        self.error = error
        self.calls = 0

    async def scalars(self, _statement):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(all=lambda: self.events)


def client_with_session(session):
    test_app = FastAPI()
    test_app.include_router(router)

    async def override_db():
        yield session

    test_app.dependency_overrides[get_db] = override_db
    return TestClient(test_app)


def sample_event(mode="fixture", observed=None):
    now = datetime.now(UTC)
    return RawKafkaPriceEvent(
        topic="sentinance.prices.v1",
        partition=0,
        offset=12,
        schema_version=1,
        source="binance_aggtrade_fixture",
        symbol="BTCUSDT",
        source_event_id="500",
        observed_at=observed or now - timedelta(minutes=10),
        received_at=now,
        price=Decimal("100.123456789012"),
        volume=Decimal("0.25"),
        raw_event={"provenance": {"mode": mode}},
    )


def test_query_returns_stale_fixture_evidence_not_unverified_live():
    response = client_with_session(FakeSession([sample_event()])).get(
        "/api/v1/market-events?symbol=BTCUSDT&limit=5"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stored_events"
    assert body["live_feed_verified"] is False
    assert "Stored events only" in body["warning"]
    assert body["events"][0]["source_mode"] == "fixture"
    assert body["events"][0]["freshness"] == "stale"
    assert body["events"][0]["price"] == "100.123456789012"
    assert body["events"][0]["source_event_id"] == "500"


def test_empty_query_is_no_data_not_a_fake_price():
    response = client_with_session(FakeSession()).get(
        "/api/v1/market-events?symbol=BTCUSDT"
    )
    assert response.status_code == 200
    assert response.json()["events"] == []
    assert response.json()["status"] == "no_data"


def test_provider_record_is_not_misrepresented_as_verified_live():
    response = client_with_session(
        FakeSession([sample_event(mode="provider", observed=datetime.now(UTC))])
    ).get("/api/v1/market-events?symbol=BTCUSDT")
    assert response.status_code == 200
    event = response.json()["events"][0]
    assert event["source_mode"] == "provider"
    assert event["freshness"] == "recent"
    assert response.json()["live_feed_verified"] is False


def test_unlabeled_old_record_is_unverified():
    response = client_with_session(
        FakeSession([sample_event(mode="synthetic_unknown")])
    ).get("/api/v1/market-events?symbol=BTCUSDT")
    assert response.json()["events"][0]["source_mode"] == "unverified"


def test_future_provider_timestamp_is_visible_not_classed_recent():
    event = sample_event(observed=datetime.now(UTC) + timedelta(minutes=10))
    response = client_with_session(FakeSession([event])).get(
        "/api/v1/market-events?symbol=BTCUSDT"
    )
    assert response.json()["events"][0]["freshness"] == "future_timestamp"


def test_bad_symbol_and_limit_are_rejected_before_db():
    session = FakeSession()
    client = client_with_session(session)
    for query in ("symbol=../secrets", "symbol=BTCUSDT&limit=1000"):
        result = client.get(f"/api/v1/market-events?{query}")
        assert result.status_code == 422
    assert session.calls == 0


def test_database_outage_returns_explicit_503_not_false_empty_result():
    error = OperationalError("SELECT", {}, RuntimeError("fixture database offline"))
    response = client_with_session(FakeSession(error=error)).get(
        "/api/v1/market-events?symbol=BTCUSDT"
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Durable market-event ledger unavailable"
