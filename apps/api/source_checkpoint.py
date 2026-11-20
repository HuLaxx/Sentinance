"""Local durable broker-acknowledged cursor; not a sink watermark."""

import sqlite3
from contextlib import closing
from pathlib import Path

from binance_aggtrades_v1 import ALLOWED_SYMBOLS, BinanceAggregateTrades, SourceMode
from kafka_producer import PriceProducer

MAX_ID = 2**63 - 1


class CursorConflict(RuntimeError):
    """Concurrent worker advanced the checkpoint; restart from stored state."""


class LocalCheckpoint:
    """SQLite compare-and-set cursor, scoped to adapter mode and symbol.

    Use an operator-owned local directory. Separate workers may publish the
    same page concurrently, but only one can advance its checkpoint.
    """

    def __init__(self, path: Path):
        self.path = path
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS source_cursor (
                    mode TEXT NOT NULL, symbol TEXT NOT NULL,
                    initial_id INTEGER NOT NULL, next_id INTEGER NOT NULL,
                    PRIMARY KEY (mode, symbol),
                    CHECK (initial_id >= 0 AND next_id >= initial_id)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @staticmethod
    def validate_scope(mode: SourceMode, symbol: str, value: int) -> None:
        if mode not in ("fixture", "provider") or symbol not in ALLOWED_SYMBOLS:
            raise ValueError("Unknown checkpoint scope")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_ID:
            raise ValueError("Cursor must be a nonnegative signed 64-bit integer")

    def load(self, mode: SourceMode, symbol: str, initial_id: int) -> int:
        self.validate_scope(mode, symbol, initial_id)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR IGNORE INTO source_cursor VALUES (?, ?, ?, ?)",
                (mode, symbol, initial_id, initial_id),
            )
            row = conn.execute(
                "SELECT initial_id, next_id FROM source_cursor WHERE mode=? AND symbol=?",
                (mode, symbol),
            ).fetchone()
            if row is None or row[0] != initial_id:
                raise CursorConflict(
                    "Checkpoint already initialized with another origin"
                )
            return int(row[1])

    @classmethod
    def read_existing(
        cls, path: Path, mode: SourceMode, symbol: str, initial_id: int
    ) -> int:
        """Inspect checkpoint state without initializing or writing any rows."""
        cls.validate_scope(mode, symbol, initial_id)
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as conn:
            row = conn.execute(
                "SELECT initial_id, next_id FROM source_cursor WHERE mode=? AND symbol=?",
                (mode, symbol),
            ).fetchone()
        if row is None or row[0] != initial_id:
            raise CursorConflict("No checkpoint with the requested origin")
        return int(row[1])

    def advance(
        self, mode: SourceMode, symbol: str, expected: int, next_id: int
    ) -> None:
        self.validate_scope(mode, symbol, expected)
        self.validate_scope(mode, symbol, next_id)
        if not expected < next_id <= expected + 1000:
            raise ValueError("Checkpoint must advance by one nonempty bounded page")
        with closing(self._connect()) as conn, conn:
            changed = conn.execute(
                "UPDATE source_cursor SET next_id=? WHERE mode=? AND symbol=? AND next_id=?",
                (next_id, mode, symbol, expected),
            )
            if changed.rowcount != 1:
                raise CursorConflict(
                    "Checkpoint changed concurrently; reopen and retry"
                )


async def ingest_checkpointed(
    *,
    source: BinanceAggregateTrades,
    producer: PriceProducer,
    checkpoint: LocalCheckpoint,
    symbol: str,
    initial_id: int,
    limit: int = 100,
    max_pages: int = 1,
) -> int:
    """Resume inclusive source ID; checkpoint only after every page ACK.

    A crash before checkpoint commit republishes its page on retry; the
    downstream ledger deduplicates by source, symbol and upstream ID.
    This cursor does not prove PostgreSQL receipt or broker retention safety.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= 10:
        raise ValueError("max_pages must be between 1 and 10")
    cursor = checkpoint.load(source.mode, symbol, initial_id)
    for _ in range(max_pages):
        if cursor > MAX_ID - limit:
            raise ValueError("Cursor too close to signed 64-bit limit")
        count = await source.publish_page(producer, symbol, from_id=cursor, limit=limit)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= limit:
            raise ValueError("Invalid acknowledged page count")
        if count:
            checkpoint.advance(source.mode, symbol, cursor, cursor + count)
            cursor += count
        if count < limit:
            break
    return cursor
