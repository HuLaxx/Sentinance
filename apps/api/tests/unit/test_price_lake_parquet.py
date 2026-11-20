"""Deterministic offline fixture tests for all three immutable local Parquet layers."""

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from forecast_baselines import evaluate_snapshot
from forecast_model import evaluate_model_snapshot, report_bytes
from kafka_price_models import RawKafkaPriceEvent
from price_event_contract import make_price_v1
from price_lake import export_snapshot, verify_snapshot

pq = pytest.importorskip("pyarrow.parquet")


def fixture(event_id: str, price: str, stamp: datetime, offset: int):
    raw = make_price_v1(
        source="fixture_exchange",
        event_id=event_id,
        symbol="BTCUSDT",
        observed_at=stamp,
        price=price,
        volume="2",
    )
    return RawKafkaPriceEvent(
        topic="sentinance.prices.v1",
        partition=0,
        offset=offset,
        schema_version=1,
        source="fixture_exchange",
        source_event_id=event_id,
        symbol="BTCUSDT",
        observed_at=stamp,
        price=Decimal(price),
        volume=Decimal(2),
        raw_event=raw,
    )


def test_raw_validated_curated_and_checksum_replay(tmp_path):
    base = datetime(2026, 11, 17, 10, 25, tzinfo=UTC)
    later = base.replace(second=30)
    earlier = base.replace(second=10)
    rows = [fixture("2", "102", later, 4), fixture("1", "100", earlier, 3)]
    manifest = export_snapshot(rows, tmp_path)
    snapshot = tmp_path / "snapshots" / manifest["snapshot_id"]
    assert manifest["input_count"] == 2
    assert manifest["curated_minute_count"] == 1
    assert set(manifest["files"]) == {
        "raw/raw.parquet",
        "validated/validated.parquet",
        "curated/minute_bars.parquet",
    }
    assert export_snapshot(list(reversed(rows)), tmp_path) == manifest
    assert verify_snapshot(snapshot) == manifest

    raw = pq.read_table(snapshot / "raw" / "raw.parquet").to_pylist()
    validated = pq.read_table(snapshot / "validated" / "validated.parquet").to_pylist()
    bars = pq.read_table(snapshot / "curated" / "minute_bars.parquet").to_pylist()
    assert [r["event_id"] for r in raw] == ["1", "2"]
    assert [r["price"] for r in validated] == [Decimal(100), Decimal(102)]
    assert bars[0]["open"] == Decimal(100)
    assert bars[0]["close"] == Decimal(102)
    assert bars[0]["high"] == Decimal(102)
    assert bars[0]["low"] == Decimal(100)
    assert bars[0]["base_volume"] == Decimal(4)
    assert bars[0]["vwap"] == Decimal(101)
    assert bars[0]["trade_count"] == 2

    corrupted = snapshot / "raw" / "raw.parquet"
    corrupted.write_bytes(corrupted.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_snapshot(snapshot)


def test_late_arrival_creates_new_immutable_snapshot(tmp_path):
    stamp = datetime(2026, 11, 17, 10, 25, 30, tzinfo=UTC)
    first = fixture("100", "100", stamp, 100)
    early = fixture("99", "90", stamp.replace(second=0), 99)
    older = export_snapshot([first], tmp_path)
    refreshed = export_snapshot([first, early], tmp_path)
    assert older["snapshot_id"] != refreshed["snapshot_id"]
    assert older["input_count"] == 1
    assert refreshed["input_count"] == 2
    assert verify_snapshot(tmp_path / "snapshots" / older["snapshot_id"]) == older
    latest = pq.read_table(
        tmp_path
        / "snapshots"
        / refreshed["snapshot_id"]
        / "curated"
        / "minute_bars.parquet"
    ).to_pylist()
    assert latest[0]["open"] == Decimal(90)
    assert latest[0]["close"] == Decimal(100)


def test_mismatched_persisted_raw_payload_is_not_exported(tmp_path):
    stamp = datetime(2026, 11, 17, 10, 25, tzinfo=UTC)
    row = fixture("100", "100", stamp, 100)
    row.price = Decimal(999)
    with pytest.raises(ValueError, match="does not match"):
        export_snapshot([row], tmp_path)
    assert not (tmp_path / "snapshots").exists()


def test_empty_ledger_does_not_create_false_snapshot(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        export_snapshot([], tmp_path)


def test_pinned_parquet_snapshot_to_walk_forward_report(tmp_path):
    start = datetime(2026, 11, 17, 10, 25, tzinfo=UTC)
    rows = [
        fixture(str(i), str(price), start + timedelta(minutes=i), i)
        for i, price in enumerate([100, 110, 120, 130])
    ]
    manifest = export_snapshot(rows, tmp_path)
    snapshot = tmp_path / "snapshots" / manifest["snapshot_id"]
    trusted_hash = hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    report = evaluate_snapshot(
        snapshot, trusted_hash, "fixture_exchange", "BTCUSDT", 2, 2
    )
    assert report["snapshot_id"] == manifest["snapshot_id"]
    assert report["manifest_sha256"] == trusted_hash
    assert report["metrics"]["last_close"]["mae"] == 10.0
    assert report["metrics"]["moving_average"]["rmse"] == 15.0
    assert [p["actual"] for p in report["predictions"]] == [120.0, 130.0]
    assert report == evaluate_snapshot(
        snapshot, trusted_hash, "fixture_exchange", "BTCUSDT", 2, 2
    )
    with pytest.raises(ValueError, match="warmup"):
        evaluate_snapshot(snapshot, trusted_hash, "fixture_exchange", "ETHUSDT", 2, 2)
    with pytest.raises(ValueError, match="pinned hash"):
        evaluate_snapshot(snapshot, "0" * 64, "fixture_exchange", "BTCUSDT", 2, 2)
    curated = snapshot / "curated" / "minute_bars.parquet"
    curated.write_bytes(curated.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_snapshot(snapshot, trusted_hash, "fixture_exchange", "BTCUSDT", 2, 2)


def test_chronological_model_from_pinned_parquet(tmp_path):
    start = datetime(2026, 11, 17, 10, 25, tzinfo=UTC)
    rows = [
        fixture(str(i), str(100 + i + i % 3), start + timedelta(minutes=i), i)
        for i in range(24)
    ]
    manifest = export_snapshot(rows, tmp_path)
    snapshot = tmp_path / "snapshots" / manifest["snapshot_id"]
    trusted_hash = hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    report = evaluate_model_snapshot(
        snapshot,
        trusted_hash,
        "fixture_exchange",
        "BTCUSDT",
        train_end=12,
        validation_end=18,
    )
    assert report["snapshot_id"] == manifest["snapshot_id"]
    assert report["manifest_sha256"] == trusted_hash
    assert report["test_metrics"]["ridge_ar1"]["count"] == 6
    assert report_bytes(report) == report_bytes(
        evaluate_model_snapshot(
            snapshot,
            trusted_hash,
            "fixture_exchange",
            "BTCUSDT",
            train_end=12,
            validation_end=18,
        )
    )
    with pytest.raises(ValueError, match="pinned hash"):
        evaluate_model_snapshot(
            snapshot,
            "0" * 64,
            "fixture_exchange",
            "BTCUSDT",
            train_end=12,
            validation_end=18,
        )
