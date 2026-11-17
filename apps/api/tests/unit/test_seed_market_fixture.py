"""Fixture publisher runs offline and never silently enables itself."""

import pytest

import seed_market_fixture


@pytest.mark.asyncio
async def test_fixture_publisher_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("P1_FIXTURE_SEED_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="explicitly set"):
        await seed_market_fixture.seed()


@pytest.mark.asyncio
async def test_fixture_publisher_preserves_fixture_source_and_provider_ids(
    monkeypatch,
):
    captured = []

    class FakeProducer:
        async def start(self):
            return None

        async def stop(self):
            return None

        async def send_price_event_v1(self, event):
            captured.append(event)

    monkeypatch.setenv("P1_FIXTURE_SEED_ENABLED", "1")
    monkeypatch.setattr(seed_market_fixture, "PriceProducer", FakeProducer)

    count = await seed_market_fixture.seed()
    assert count == 2
    assert [event["event_id"] for event in captured] == ["900001", "900002"]
    assert all(event["source"] == "binance_aggtrade_fixture" for event in captured)
    assert all(event["provenance"]["mode"] == "fixture" for event in captured)
    assert all(event["data"]["price"] for event in captured)
