"""Opt-in standalone Kafka-to-PostgreSQL price-event ledger consumer.

Run only after applying Alembic migrations and explicitly enabling the v1
worker. This does not start a live exchange connector or model pipeline.
"""

import asyncio
import os
import signal

from durable_price_ingestion import create_durable_price_consumer


async def main() -> None:
    if os.getenv("P1_DURABLE_PRICE_SINK_ENABLED") != "1":
        raise RuntimeError(
            "Set P1_DURABLE_PRICE_SINK_ENABLED=1 after applying database migrations"
        )

    consumer = create_durable_price_consumer()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)

    await consumer.start()
    worker = consumer._task
    if worker is None:
        raise RuntimeError("Durable Kafka consumer failed to start")

    interrupted = asyncio.create_task(stop.wait())
    try:
        finished, _ = await asyncio.wait(
            (worker, interrupted), return_when=asyncio.FIRST_COMPLETED
        )
        if worker in finished:
            await worker  # Propagate unexpected consumer failures.
    finally:
        interrupted.cancel()
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(main())
