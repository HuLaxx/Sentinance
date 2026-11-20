"""Bounded source-to-PostgreSQL receipt reconciliation and explicit gap repair."""

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from binance_aggtrades_v1 import BinanceAggregateTrades, SourceMode
from database import async_session, close_db
from kafka_price_models import RawKafkaPriceEvent
from kafka_producer import PriceProducer
from source_checkpoint import LocalCheckpoint

MAX_WINDOW = 1000


@dataclass(frozen=True)
class ReceiptReport:
    mode: SourceMode
    symbol: str
    start_id: int
    stop_id: int
    received_count: int
    missing_ranges: tuple[tuple[int, int], ...]

    @property
    def complete(self) -> bool:
        return not self.missing_ranges


async def reconcile_receipts(
    session: AsyncSession,
    *,
    mode: SourceMode,
    symbol: str,
    start_id: int,
    stop_id: int,
) -> ReceiptReport:
    """Inspect a bounded half-open window [start, stop) with one read query."""
    LocalCheckpoint.validate_scope(mode, symbol, start_id)
    LocalCheckpoint.validate_scope(mode, symbol, stop_id)
    if not 0 < stop_id - start_id <= MAX_WINDOW:
        raise ValueError("Receipt window must contain 1 to 1000 aggregate IDs")
    namespace = "binance_aggtrade_fixture" if mode == "fixture" else "binance_aggtrade"
    expected = range(start_id, stop_id)
    rows = await session.execute(
        select(RawKafkaPriceEvent.source_event_id, RawKafkaPriceEvent.raw_event).where(
            RawKafkaPriceEvent.source == namespace,
            RawKafkaPriceEvent.symbol == symbol,
            RawKafkaPriceEvent.source_event_id.in_([str(i) for i in expected]),
        )
    )
    received: set[int] = set()
    for identity, event in rows:
        provenance = event.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("mode") != mode
            or type(provenance.get("aggregate_trade_id")) is not int
            or provenance.get("aggregate_trade_id") != int(identity)
        ):
            raise ValueError("Stored source provenance does not match receipt scope")
        received.add(int(identity))
    gaps: list[tuple[int, int]] = []
    for identifier in expected:
        if identifier not in received:
            if gaps and gaps[-1][1] == identifier:
                gaps[-1] = (gaps[-1][0], identifier + 1)
            else:
                gaps.append((identifier, identifier + 1))
    return ReceiptReport(mode, symbol, start_id, stop_id, len(received), tuple(gaps))


async def republish_missing(
    report: ReceiptReport, source: BinanceAggregateTrades, producer: PriceProducer
) -> int:
    """Explicit repair of reported holes; ACK count is not a receipt guarantee."""
    if source.mode != report.mode:
        raise ValueError("Repair source mode must match receipt report")
    total = 0
    for start, stop in report.missing_ranges:
        events = await source.fetch_page(report.symbol, from_id=start, limit=stop - start)
        # Never publish a truncated repair page as if it filled the gap.
        if len(events) != stop - start:
            raise ValueError("Provider did not return the entire missing interval")
        for event in events:
            await producer.send_price_event_v1(event)
            total += 1
    return total


async def run(
    checkpoint_path: Path,
    symbol: str,
    initial_id: int,
    start_id: int,
    stop_id: int,
    repair: bool,
) -> dict:
    """Scan only a previously ACKed provider interval; repair requires consent."""
    if repair and (
        os.getenv("SENTINANCE_BINANCE_LIVE_ENABLED") != "1"
        or os.getenv("SENTINANCE_MARKET_DATA_RIGHTS_CONFIRMED") != "1"
    ):
        raise RuntimeError("Live repair requires source and market-data permissions")
    # Fail closed before touching a nonexistent file or unacknowledged range.
    if not checkpoint_path.is_file():
        raise ValueError("An existing source checkpoint file is required")
    cursor = LocalCheckpoint.read_existing(
        checkpoint_path, "provider", symbol, initial_id
    )
    if not initial_id <= start_id < stop_id <= cursor:
        raise ValueError("Receipt range must be within the acknowledged interval")
    try:
        async with async_session() as session:
            report = await reconcile_receipts(
                session,
                mode="provider",
                symbol=symbol,
                start_id=start_id,
                stop_id=stop_id,
            )
        result = asdict(report)
        result["complete"] = report.complete
        result["repair_broker_acks"] = 0
        if repair and not report.complete:
            async with httpx.AsyncClient(follow_redirects=False) as client:
                source = BinanceAggregateTrades(client, mode="provider")
                producer = PriceProducer()
                await producer.start()
                try:
                    result["repair_broker_acks"] = await republish_missing(
                        report, source, producer
                    )
                finally:
                    await producer.stop()
        # Re-run after the sink has caught up; no cursor advancement here.
        return result
    finally:
        await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--initial-id", type=int, required=True)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--stop-id", type=int, required=True)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        run(
            args.checkpoint,
            args.symbol,
            args.initial_id,
            args.start_id,
            args.stop_id,
            args.repair,
        )
    )
    print(json.dumps(result, sort_keys=True))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
