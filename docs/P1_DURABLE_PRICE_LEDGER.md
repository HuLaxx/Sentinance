# P1: durable Kafka → PostgreSQL price-event ledger

**Scope:** first reviewable P1 ingestion slice in
[draft PR #31](https://github.com/HuLaxx/Sentinance/pull/31). The live market
provider, default `price_stream_task()`, ML/agent and browser are **not**
connected to this new topic. No deployment or provider licensing is implied.

## Implemented path

```
Explicit fixture/source event (stable source event_id, observed time)
  -> PriceProducer.send_price_v1() [broker ack, acks=all]
  -> Kafka topic sentinance.prices.v1 [JSON schema_version=1]
  -> PriceConsumer [manual Kafka commits; dedicated group]
  -> DurablePriceSink [PostgreSQL transaction; unique source ID and Kafka offset]
  -> Kafka offset commit ONLY after PostgreSQL commit
```

Example **synthetic** event, not an observed exchange quote:

```json
{
  "type": "price_update",
  "schema_version": 1,
  "source": "fixture_exchange",
  "event_id": "synthetic-stable-tick-001",
  "symbol": "BTCUSDT",
  "timestamp": "2026-11-16T10:00:00+00:00",
  "data": { "price": "101.25", "volume": "2" }
}
```

The timestamp is supplied by the event source, **not fabricated at ingestion**;
naive times, invalid schema versions, nonfinite/nonpositive prices, negative
volumes and values outside PostgreSQL NUMERIC(30,12) are rejected. The
insertion timestamp is recorded separately as `received_at`.

The Postgres table `raw_kafka_price_events` is append-only for accepted
events, with primary key `(topic, partition, offset)` and a unique
`(source, source_event_id)` constraint. Replayed records and republished events
with the *same* source ID and identical payload do not create duplicates. If
either idempotency key is reused with **different content**, the consumer fails,
leaving the Kafka record unacknowledged for operator investigation.

## Local or isolated execution (manual opt-in only)

1. Supply a real isolated `DATABASE_URL` using the asyncpg dialect
   (`postgresql+asyncpg://...`) and a locally accessible Kafka broker via
   `KAFKA_BOOTSTRAP_SERVERS`.
2. Inside `apps/api`, apply `alembic upgrade head` to that **isolated**
   PostgreSQL database. Do not run this against production without a reviewed
   migration plan.
3. Start the standalone worker with
   `P1_DURABLE_PRICE_SINK_ENABLED=1 python durable_price_worker.py`. It does
   not start unless explicitly enabled.
4. To create **synthetic smoke events** use
   `PriceProducer.send_price_v1(source=..., event_id=..., symbol=...,
   observed_at=..., price=..., volume=...)`. Reuse the same source ID when
   retrying the same source event.
5. Query `raw_kafka_price_events`; inspect actual Kafka group offsets and
   logs before considering a replay test successful.

This worker is deliberately not enabled in `main.py` or Kubernetes manifests.
Deploying it requires separate consumer lifecycle, secret handling, database
migration, broker quotas and resource review. Neither duplicate rejection nor
replay can reconstruct provider observations that never reached Kafka.

## Verified vs open

The dedicated CI workflow
`.github/workflows/p1-durable-kafka-postgres.yml` uses disposable Kafka
and PostgreSQL, applies the real Alembic migration and injects a consumer
crash **after** storing an event but **before** committing the Kafka offset.
A restarted consumer must replay the uncommitted record; another broker
publication with the same source event ID must leave exactly one database row.
These checks prove a local at-least-once processing contract backed by a
transactional idempotent sink, **not** a Kafka/PostgreSQL atomic transaction
or exactly-once system-wide delivery.

Still required before P1 is complete:

- [ ] Connect a clearly labeled real/licensed or fixture source adapter to
      `sentinance.prices.v1` without inventing unavailable event IDs.
- [ ] Establish raw/validated/curated Parquet/S3 or equivalent object storage
      with immutable partition provenance and tested schema evolution.
- [ ] Define provider sequence gaps, event-time watermarks, bounded backfills,
      late arrivals and explicit replay windows.
- [ ] Validate concurrent consumers, partition assignment changes, broker
      retention, poison-event dead-letter quarantine and operational alerts.
- [ ] Join curated records to reproducible feature/model evaluations,
      then to a clearly live/mock-labeled API and browser journey.
- [ ] Review the PostgreSQL migration in dependency order and obtain
      authorization before any production deployment.

Any claims in `docs/data_lineage.md` describing Spark, dbt, Feast, MLflow,
live exchanges or E2E latency remain **design goals** unless backed by separate
passing, linked evidence.
