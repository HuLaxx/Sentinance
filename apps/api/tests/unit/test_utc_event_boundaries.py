"""Contract tests for UTC-aware timestamps emitted by API-side subsystems.

Tests validate serialization, not actual Kafka delivery, production exchange
ingestion, an ML backtest or profitability.
"""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from ab_testing import ABTestingService
from kafka_producer import ALERT_TOPIC, NEWS_TOPIC, PRICE_TOPIC, PriceProducer
from market_stats import MarketStats
from news_scraper import NewsArticle
from predictor import generate_prediction
from trading_signals import SignalGenerator


def assert_utc_iso(value: str) -> None:
    timestamp = datetime.fromisoformat(value)
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_experiment_lifecycle_has_utc_offset_on_all_written_dates():
    service = ABTestingService()
    experiment = service.create_experiment(
        "utc-test",
        "UTC emission regression",
        "baseline",
        "candidate",
    )
    assert_utc_iso(experiment.created_at)
    assert service.start_experiment(experiment.id)
    assert_utc_iso(experiment.started_at)
    assert service.complete_experiment(experiment.id, winner="control")
    assert_utc_iso(experiment.completed_at)


@pytest.mark.asyncio
async def test_kafka_price_alert_and_news_envelopes_include_utc_offset():
    producer = PriceProducer()
    sender = AsyncMock()
    producer.producer = sender
    producer._started = True

    await producer.send_price("BTCUSDT", {"price": 100.0})
    await producer.send_alert("alert-1", {"status": "triggered"})
    await producer.send_news({"title": "Fixture event"})

    sent = sender.send.await_args_list
    assert len(sent) == 3
    assert [call.args[0] for call in sent] == [
        PRICE_TOPIC,
        ALERT_TOPIC,
        NEWS_TOPIC,
    ]
    for call in sent:
        assert_utc_iso(call.kwargs["value"]["timestamp"])


def test_market_statistics_timestamp_is_explicit_utc():
    result = MarketStats().to_dict()
    assert_utc_iso(result["timestamp"])


def test_mock_prediction_metadata_timestamp_is_explicit_utc():
    prediction = generate_prediction(
        symbol="BTCUSDT",
        prices=[100.0, 101.0, 102.0, 103.0, 104.0],
        current_price=104.0,
        model="momentum",
    )
    assert_utc_iso(prediction.timestamp)


@pytest.mark.asyncio
async def test_price_history_records_utc_offset_without_external_feed(
    monkeypatch,
):
    from services import price_stream

    class FakeConnector:
        async def get_prices(self):
            return [
                {
                    "symbol": "BTCUSDT",
                    "name": "Bitcoin",
                    "price": 105.0,
                    "priceChangePercent": 1.0,
                    "volume": 2.0,
                }
            ]

    monkeypatch.setattr(price_stream, "get_connector", FakeConnector)
    monkeypatch.setattr(price_stream, "CURRENT_PRICES", {})
    monkeypatch.setattr(price_stream, "PRICE_HISTORY", {})

    prices = await price_stream.fetch_and_update_prices()
    assert len(prices) == 1
    history = price_stream.PRICE_HISTORY["BTCUSDT"]
    assert len(history) == 1
    assert_utc_iso(history[0]["timestamp"])


def test_signal_generator_has_utc_metadata_without_backtest_claims():
    signal = SignalGenerator().generate_signal("BTCUSDT", 105.0, {})
    assert_utc_iso(signal.timestamp)
    assert signal.id.startswith("BTCUSDT-")


def test_rss_news_normalizes_missing_naive_and_offset_times_to_utc():
    utc = NewsArticle("UTC", "https://example.com/1", "fixture")
    naive = NewsArticle(
        "Naive",
        "https://example.com/2",
        "fixture",
        published_at=datetime(2026, 11, 13, 12, 30),
    )
    offset = NewsArticle(
        "Offset",
        "https://example.com/3",
        "fixture",
        published_at=datetime(
            2026, 11, 13, 7, 30, tzinfo=timezone(timedelta(hours=-5))
        ),
    )
    for article in (utc, naive, offset):
        assert_utc_iso(article.to_dict()["published_at"])

    assert naive.published_at == offset.published_at
    assert naive.published_at == datetime(2026, 11, 13, 12, 30, tzinfo=UTC)
    ordered = sorted((naive, utc, offset), key=lambda item: item.published_at)
    assert ordered[0].published_at <= ordered[1].published_at
    assert ordered[1].published_at <= ordered[2].published_at
