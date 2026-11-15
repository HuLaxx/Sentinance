"""Real Redis stop/start and two-OS-process fanout contract.

This script is invoked only by the dedicated GitHub CI resilience job, where
a disposable Redis Docker container is created. It never targets production.
Redis Pub/Sub is non-durable: no replay or exactly-once claim is made.
"""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import redis.asyncio as redis
from redis.exceptions import RedisError

CHANNEL = "sentinance:updates"
WORKER = Path(__file__).with_name("redis_restart_worker.py")
CONTAINER = "sentinance-redis-resilience"


async def docker(*args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=12)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"Docker fault injection failed: {args} {output.decode(errors='replace')}"
        )


async def read_events(
    name: str,
    process: asyncio.subprocess.Process,
    events: asyncio.Queue[dict],
) -> None:
    assert process.stdout is not None
    while line := await process.stdout.readline():
        text = line.decode(errors="replace").strip()
        try:
            record = json.loads(text)
        except ValueError:
            print(f"[{name}] {text}", flush=True)
            continue
        if isinstance(record, dict) and "event" in record:
            record["worker"] = name
            events.put_nowait(record)
        else:
            print(f"[{name}] {text}", flush=True)


async def check(
    description: str,
    predicate,
    events: asyncio.Queue[dict],
    history: list[dict],
    processes: dict[str, asyncio.subprocess.Process],
    *,
    timeout: float = 10,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate(history):
        failed = [
            f"{name} exited with {proc.returncode}"
            for name, proc in processes.items()
            if proc.returncode is not None
        ]
        if failed:
            raise AssertionError(f"{description}: {'; '.join(failed)}")
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise AssertionError(f"{description}: timed out; events={history[-15:]}")
        try:
            record = await asyncio.wait_for(
                events.get(),
                timeout=min(remaining, 0.5),
            )
        except TimeoutError:
            continue
        history.append(record)
        if record["event"] in {"ready", "offline", "message", "fatal"}:
            print(f"[{record['worker']}] {record}", flush=True)
        if record["event"] == "fatal":
            raise AssertionError(f"{description}: {record}")


def has_each(history: list[dict], event: str, **details: object) -> bool:
    for worker in ("alpha", "beta"):
        if not any(
            record.get("worker") == worker
            and record.get("event") == event
            and all(record.get(key) == val for key, val in details.items())
            for record in history
        ):
            return False
    return True


def received(history: list[dict], sequence: int) -> bool:
    for worker in ("alpha", "beta"):
        if not any(
            item.get("worker") == worker
            and item.get("event") == "message"
            and item.get("payload", {}).get("sequence") == sequence
            for item in history
        ):
            return False
    return True


async def wait_for_redis(client: redis.Redis, *, timeout: float = 10) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            if await client.ping():
                return
        except (RedisError, OSError, TimeoutError):
            pass
        await asyncio.sleep(0.1)
    raise AssertionError("Restarted Redis did not answer PING")


async def subscriber_count(client: redis.Redis) -> int:
    counts = await client.pubsub_numsub(CHANNEL)
    return int(counts[0][1])


async def run() -> None:
    url = os.environ["SENTINANCE_RESTART_REDIS_URL"]
    container = os.environ["SENTINANCE_RESTART_REDIS_CONTAINER"]
    if container != CONTAINER:
        raise ValueError("Only the disposable dedicated CI Redis container is allowed")

    client = redis.from_url(url, decode_responses=True, socket_connect_timeout=1)
    processes: dict[str, asyncio.subprocess.Process] = {}
    pumps: list[asyncio.Task[None]] = []
    events: asyncio.Queue[dict] = asyncio.Queue()
    history: list[dict] = []

    try:
        await wait_for_redis(client)
        # Prove a message published before subscriptions is NOT replayed.
        before = await client.publish(
            CHANNEL, json.dumps({"sequence": 0, "phase": "no-subscribers"})
        )
        assert before == 0, "Unexpected existing Redis channel subscribers"

        for name in ("alpha", "beta"):
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                str(WORKER),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "SENTINANCE_RESTART_REDIS_URL": url},
            )
            processes[name] = proc
            pumps.append(asyncio.create_task(read_events(name, proc, events)))

        assert len({proc.pid for proc in processes.values()}) == 2
        assert all(proc.pid != os.getpid() for proc in processes.values())

        await check(
            "two independent subscribers online",
            lambda h: has_each(h, "ready", retries=0),
            events,
            history,
            processes,
            timeout=10,
        )
        assert await subscriber_count(client) == 2
        count = await client.publish(
            CHANNEL, json.dumps({"sequence": 1, "phase": "before-restart"})
        )
        assert count == 2
        await check(
            "fanout before Redis restart",
            lambda h: received(h, 1),
            events,
            history,
            processes,
        )
        print(
            "PASS: separate OS processes received the pre-restart message", flush=True
        )

        # Stop a real Redis *process*, not a simulated Python transport stub.
        await docker("stop", "-t", "1", CONTAINER)
        await check(
            "each subscriber detects lost broker",
            lambda h: has_each(h, "offline"),
            events,
            history,
            processes,
            timeout=8,
        )
        await docker("start", CONTAINER)
        await wait_for_redis(client)

        await check(
            "both subscribers resubscribe after real broker restart",
            lambda h: all(
                any(
                    item.get("worker") == name
                    and item.get("event") == "ready"
                    and item.get("retries", 0) >= 1
                    for item in h
                )
                for name in ("alpha", "beta")
            ),
            events,
            history,
            processes,
            timeout=12,
        )
        assert await subscriber_count(client) == 2
        count = await client.publish(
            CHANNEL, json.dumps({"sequence": 2, "phase": "after-restart"})
        )
        assert count == 2
        await check(
            "fanout after restart",
            lambda h: received(h, 2),
            events,
            history,
            processes,
        )

        for name in ("alpha", "beta"):
            for sequence in (1, 2):
                messages = [
                    x
                    for x in history
                    if x.get("worker") == name
                    and x.get("event") == "message"
                    and x.get("payload", {}).get("sequence") == sequence
                ]
                assert len(messages) == 1, (name, sequence, messages)
        assert not any(
            x.get("event") == "message" and x.get("payload", {}).get("sequence") == 0
            for x in history
        ), "Redis Pub/Sub unexpectedly replayed a pre-subscription message"
        print(
            "PASS: 2 isolated OS processes recovered fanout after Redis stop/start; "
            "no replay is claimed for offline gaps",
            flush=True,
        )
    finally:
        for proc in processes.values():
            if proc.returncode is None:
                proc.send_signal(signal.SIGTERM)
        for proc in processes.values():
            try:
                await asyncio.wait_for(proc.wait(), timeout=4)
            except TimeoutError:
                proc.kill()
                await proc.wait()
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(run())
