"""Immutable local raw/validated/curated Parquet snapshots from the durable ledger.

This exports *already persisted* source-identified aggregate-trade events.
The sample is not a live deployment, trading signal, or forecast dataset.
Parquet is optional to the API runtime, but mandatory in the dedicated lake CI.
"""

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database import async_session
from db_time import db_utc_to_aware
from kafka_price_models import RawKafkaPriceEvent
from price_event_contract import validate_price_v1

SCHEMA_VERSION = 1


def _utc(value: datetime) -> datetime:
    return db_utc_to_aware(value)


def _canonical(row: RawKafkaPriceEvent) -> dict[str, Any]:
    """Cross-check stored columns against raw source JSON before any export."""
    event = validate_price_v1(row.raw_event)
    if (
        row.source != event.source
        or row.symbol != event.symbol
        or row.source_event_id != event.source_event_id
        or _utc(row.observed_at) != event.observed_at
        or row.price != event.price
        or row.volume != event.volume
        or row.schema_version != SCHEMA_VERSION
    ):
        raise ValueError("Raw event does not match its persisted identity and values")
    if row.partition < 0 or row.offset < 0:
        raise ValueError("Kafka positions must be nonnegative")
    return {
        "topic": row.topic,
        "partition": row.partition,
        "offset": row.offset,
        "source": row.source,
        "symbol": row.symbol,
        "event_id": row.source_event_id,
        "observed_at": event.observed_at,
        "price": event.price,
        "volume": event.volume,
        "raw_json": json.dumps(row.raw_event, sort_keys=True, separators=(",", ":")),
    }


def _ordered(records: Sequence[RawKafkaPriceEvent]) -> list[dict[str, Any]]:
    rows = [_canonical(record) for record in records]
    return sorted(
        rows,
        key=lambda x: (
            x["observed_at"],
            x["source"],
            x["symbol"],
            x["event_id"],
        ),
    )


def _minute_bars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, datetime], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        timestamp: datetime = row["observed_at"]
        minute = timestamp.replace(second=0, microsecond=0)
        groups[(row["source"], row["symbol"], minute)].append(row)

    result: list[dict[str, Any]] = []
    for (source, symbol, minute), events in sorted(groups.items()):
        prices = [event["price"] for event in events]
        volume = sum((event["volume"] for event in events), Decimal(0))
        weighted = sum(
            (event["price"] * event["volume"] for event in events),
            Decimal(0),
        )
        result.append(
            {
                "source": source,
                "symbol": symbol,
                "minute_utc": minute,
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "base_volume": volume,
                "vwap": (weighted / volume).quantize(Decimal("0.000000000001")) if volume else None,
                "trade_count": len(events),
            }
        )
    return result


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    timestamp = pa.timestamp("us", tz="UTC")
    decimal = pa.decimal128(30, 12)
    if path.name == "raw.parquet":
        schema = pa.schema(
            [
                ("topic", pa.string()),
                ("partition", pa.int32()),
                ("offset", pa.int64()),
                ("source", pa.string()),
                ("symbol", pa.string()),
                ("event_id", pa.string()),
                ("observed_at", timestamp),
                ("price", decimal),
                ("volume", decimal),
                ("raw_json", pa.string()),
            ]
        )
    elif path.name == "validated.parquet":
        schema = pa.schema(
            [
                ("source", pa.string()),
                ("symbol", pa.string()),
                ("event_id", pa.string()),
                ("observed_at", timestamp),
                ("price", decimal),
                ("volume", decimal),
            ]
        )
    else:
        schema = pa.schema(
            [
                ("source", pa.string()),
                ("symbol", pa.string()),
                ("minute_utc", timestamp),
                ("open", decimal),
                ("high", decimal),
                ("low", decimal),
                ("close", decimal),
                ("base_volume", decimal),
                ("vwap", decimal),
                ("trade_count", pa.int64()),
            ]
        )
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def export_snapshot(
    records: Sequence[RawKafkaPriceEvent], root: Path
) -> dict[str, Any]:
    """Create immutable content-addressed Parquet files with a checksum manifest.

    A new late event changes the snapshot hash; previously written snapshots
    are never overwritten. Consumers must validate file checksums first.
    """
    rows = _ordered(records)
    if not rows:
        raise ValueError("Refuse to publish an empty evidence snapshot")
    identity = [
        {
            **{key: row[key] for key in ("topic", "partition", "offset", "source",
                                         "symbol", "event_id", "raw_json")},
            "observed_at": row["observed_at"].isoformat(),
        }
        for row in rows
    ]
    stable_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    snapshot_id = hashlib.sha256(stable_bytes).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    snapshots = root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    target = snapshots / snapshot_id
    if target.exists():
        return verify_snapshot(target)

    staging = Path(tempfile.mkdtemp(dir=snapshots, prefix=".staging-"))
    try:
        raw_path = staging / "raw" / "raw.parquet"
        validated_path = staging / "validated" / "validated.parquet"
        curated_path = staging / "curated" / "minute_bars.parquet"
        _write_parquet(raw_path, rows)
        validated = [
            {key: row[key] for key in
             ("source", "symbol", "event_id", "observed_at", "price", "volume")}
            for row in rows
        ]
        _write_parquet(validated_path, validated)
        bars = _minute_bars(rows)
        _write_parquet(curated_path, bars)
        checksums = {}
        for path in (raw_path, validated_path, curated_path):
            checksums[str(path.relative_to(staging))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "source_kind": "ledger_snapshot",
            "input_count": len(rows),
            "curated_minute_count": len(bars),
            "sources": sorted({row["source"] for row in rows}),
            "symbols": sorted({row["symbol"] for row in rows}),
            "min_observed_at": rows[0]["observed_at"].isoformat(),
            "max_observed_at": rows[-1]["observed_at"].isoformat(),
            "files": checksums,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
        return verify_snapshot(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("snapshot_id") != snapshot.name:
        raise ValueError("Snapshot identifier does not match its directory")
    for relative, expected in manifest.get("files", {}).items():
        path = snapshot / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Parquet snapshot checksum mismatch: {relative}")
    if set(manifest.get("files", {})) != {
        "raw/raw.parquet",
        "validated/validated.parquet",
        "curated/minute_bars.parquet",
    }:
        raise ValueError("Snapshot must include all three Parquet layers")
    return manifest


async def export_ledger_snapshot(
    root: Path,
    session_factory: async_sessionmaker[AsyncSession] = async_session,
) -> dict[str, Any]:
    async with session_factory() as session:
        result = await session.scalars(
            select(RawKafkaPriceEvent).order_by(
                RawKafkaPriceEvent.observed_at,
                RawKafkaPriceEvent.source,
                RawKafkaPriceEvent.symbol,
                RawKafkaPriceEvent.source_event_id,
            )
        )
        rows = list(result.all())
    return export_snapshot(rows, root)
