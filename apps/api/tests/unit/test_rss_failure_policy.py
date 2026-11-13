"""Failure-policy tests for best-effort RSS aggregation.

HTTP outages are recoverable per source; unexpected parser faults must be
logged by the aggregator rather than silently converted into an empty feed.
These tests use synthetic responses and do not contact live providers.
"""

import asyncio
from unittest.mock import Mock

import httpx
import pytest

import news_scraper


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 503])
async def test_http_error_is_logged_and_isolated(monkeypatch, status_code):
    logger = Mock()
    monkeypatch.setattr(news_scraper, "log", logger)

    def respond(request):
        return httpx.Response(status_code, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        articles = await news_scraper.fetch_rss(
            client, "https://example.invalid/rss", "fixture"
        )

    assert articles == []
    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[0] == "rss_fetch_failed"
    assert logger.warning.call_args.kwargs["source"] == "fixture"


@pytest.mark.asyncio
async def test_transport_timeout_is_logged_as_provider_failure(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(news_scraper, "log", logger)

    def timed_out(request):
        raise httpx.ReadTimeout("provider timeout", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(timed_out)
    ) as client:
        assert (
            await news_scraper.fetch_rss(
                client, "https://example.invalid/rss", "fixture"
            )
            == []
        )

    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[0] == "rss_fetch_failed"


@pytest.mark.asyncio
async def test_unexpected_parser_bug_is_not_misreported_as_http_outage(
    monkeypatch,
):
    logger = Mock()
    monkeypatch.setattr(news_scraper, "log", logger)

    def respond(request):
        return httpx.Response(200, text="<rss/>", request=request)

    def broken_parser(content, source):
        raise RuntimeError("unexpected parser bug")

    monkeypatch.setattr(news_scraper, "parse_rss", broken_parser)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(RuntimeError, match="unexpected parser bug"):
            await news_scraper.fetch_rss(
                client, "https://example.invalid/rss", "fixture"
            )

    logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_aggregator_records_broken_source_and_keeps_healthy_feed(
    monkeypatch,
):
    logger = Mock()
    monkeypatch.setattr(news_scraper, "log", logger)
    monkeypatch.setattr(
        news_scraper,
        "NEWS_SOURCES",
        {
            "broken": {"type": "rss", "url": "https://example.invalid/broken"},
            "healthy": {"type": "rss", "url": "https://example.invalid/good"},
        },
    )

    async def fake_fetch(client, url, source):
        if source == "broken":
            raise RuntimeError("invalid parser implementation")
        return [
            news_scraper.NewsArticle(
                "Fixture title", "https://example.invalid/story", source
            )
        ]

    monkeypatch.setattr(news_scraper, "fetch_rss", fake_fetch)
    articles = await news_scraper.fetch_all_news()

    assert len(articles) == 1
    assert articles[0].source == "healthy"
    logger.error.assert_called_once()
    assert logger.error.call_args.args[0] == "rss_source_failed"
    assert logger.error.call_args.kwargs["source"] == "broken"


@pytest.mark.asyncio
async def test_aggregator_does_not_consume_feed_cancellation(monkeypatch):
    monkeypatch.setattr(
        news_scraper,
        "NEWS_SOURCES",
        {"cancelled": {"type": "rss", "url": "https://example.invalid/rss"}},
    )

    async def cancelled_fetch(client, url, source):
        raise asyncio.CancelledError()

    monkeypatch.setattr(news_scraper, "fetch_rss", cancelled_fetch)
    with pytest.raises(asyncio.CancelledError):
        await news_scraper.fetch_all_news()
