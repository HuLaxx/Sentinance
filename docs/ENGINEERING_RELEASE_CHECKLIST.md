# Sentinance — Engineering Release Checklist (evidence-gated)

**Updated:** 2026-11-13 | **Target:** credible AI Systems / Market Data Engineering flagship | **Status:** DRAFT, P0 STRICT LINT RED, NOT 10/10

[x] denotes a **verified limited claim on a review branch**, not a main-branch merge, production deployment or full-stack certification. Support factual claims in [EVIDENCE_REGISTER.md](EVIDENCE_REGISTER.md). Always distinguish source market observations, derived indicators, forecasts, language-model interpretations and synthetic/mocked output. Never promise trading returns.

## P0 — CI, truthful claims and provider safety
- [x] API functional tests pass separately: **155 passed, five warnings** (including four new authentication/schema tests); MyPy and frontend tests/build passed. The full CI is still red due to strict Ruff. [API test run 43435715012](https://github.com/HuLaxx/Sentinance/actions/runs/43435715012).
- [x] Ruff formatter repaired 12 Python files and applied 35 safe fixes; **69/69 files format-clean** on later review run. [PR #2](https://github.com/HuLaxx/Sentinance/pull/2), [PR Checks 43428370907](https://github.com/HuLaxx/Sentinance/actions/runs/43428370907).
- [x] Security and commit-message check jobs pass on same review branch; a passing security scan is not an audit of all secrets. [PR Checks 43428370907](https://github.com/HuLaxx/Sentinance/actions/runs/43428370907).
- [ ] Fix **107 remaining strict Ruff findings** without blanket suppressions: **82 BLE001 broad exception handlers + 25 DTZ003 UTC-naive timestamps**. Latest exact-head code quality verifies 72/72 Python files formatted. [Strict PR checks 43440292164](https://github.com/HuLaxx/Sentinance/actions/runs/43440292164).
- [x] Add **12 scoped unit regression cases** for UTC-aware market anomaly alerts, offset-based ordering, legacy UTC-naive readings, cooldowns and subscriber error isolation. [PR #3](https://github.com/HuLaxx/Sentinance/pull/3), [151 passing API tests on initial code revision](https://github.com/HuLaxx/Sentinance/actions/runs/43433535905). The database UTC contract remains unverified.
- [x] Replace eight FastAPI `Depends` defaults with type-safe `Annotated` dependency injection, retain unauthenticated alert-list/delete rejection, preserve Pydantic JSON schema example and model routing. All four new regression tests and 155 API tests passed on implementation SHA `0fceb37411f2c397b9b5b7caa4ed46433957231b`; formatting fixes were later applied. [PR #4](https://github.com/HuLaxx/Sentinance/pull/4), [API run 43435715012](https://github.com/HuLaxx/Sentinance/actions/runs/43435715012).
- [x] Verify strict Python formatter clean across **69/69 files**. Overall PR checks intentionally remain red due to 124 remaining semantic lint findings. [Run 43436003716](https://github.com/HuLaxx/Sentinance/actions/runs/43436003716).
- [x] Establish a documented **UTC-naive storage convention** for existing SQLAlchemy `DateTime(timezone=False)` columns, with a named conversion helper; normalize `postgresql://` / `postgres://` into an explicit asyncpg URL. [Draft PR #5](https://github.com/HuLaxx/Sentinance/pull/5). **Schema migration not performed.** This is checked as an implemented local contract, not a verified PostgreSQL migration.
- [x] Verify six new offline exchange-provider regression tests: 429/503, malformed payload, fail-closed adapter programming errors, fallback deduplication and UTC stamps. [API run 43439519761](https://github.com/HuLaxx/Sentinance/actions/runs/43439519761) — provider tests passed; overall job had one unrelated SQLite test-fixture failure, since corrected but awaiting its own CI.
- [x] Verify 72/72 Python files Ruff-format clean on the code revision `45ec0b0348cbb613a71f368373868b93e285b9ac`. [PR checks 43440292164](https://github.com/HuLaxx/Sentinance/actions/runs/43440292164).
- [ ] Verify the **full database UTC SQLite round-trip test suite** after updating a fixture to provide required OHLC columns; rerun exact-head API pytest and MyPy. A local SQLite round trip does not certify existing PostgreSQL data.
- [ ] Plan and test a **separate** PostgreSQL `TIMESTAMP WITHOUT TIME ZONE` → timezone-aware migration, including rollback, historical-data semantics and API serialization if justified.
- [ ] Extend the now-tested adapter **rate-limit/invalid-payload** failure policy to all feed/UI states, with freshness indicators, retry/backoff and honest stale/partial-data presentation.
- [ ] Verify provider terms, licensing, symbols, rate limits, timestamp adjustments and no-data responses.
- [ ] Prove **both** strict PR Checks and API/web workflow green on exact current PR head.
- [ ] Audit main vs draft branches and merge only with human review; preserve legitimate loading UX.

## P1 — Reproducible market-data pipeline
- [ ] Create normalized versioned event schemas: provider, exchange, symbol, UTC event time, ingestion time, revision, adjusted/unadjusted policy.
- [ ] Implement one verifiable market feed ingestion path using a licensed source, or clearly labelled reproducible historical fixture.
- [ ] Demonstrate source → raw → validated → curated time-series Parquet/table with provenance and source hash.
- [ ] Prevent duplicated, late, missing and out-of-order candles/ticks from silently corrupting indicators.
- [ ] Implement gap detection, clock skew checks, retry+jitter, bounded rate limiting and replay/backfill.
- [ ] Add dbt-backed price/indicator/provider-health marts with data-quality and freshness tests.
- [ ] Show source timestamp, staleness, provider outage and explicit mock/sample markers in browser UI.
- [ ] Prove outage/fallback and late-event tests with a one-command seeded golden path.

## P2 — Forecasts and backtesting without hype
- [ ] Separate indicators, statistical forecasts, simulated trading backtests and agent commentary in code/UI.
- [ ] Version dataset snapshots, experiment seeds, horizon and feature availability with lineage.
- [ ] Use leakage-free rolling/walk-forward evaluation; compare last-price, moving-average and other relevant naive baselines.
- [ ] Evaluate LSTM or other complex models only after baseline comparisons; publish negative results honestly.
- [ ] Report MAE/RMSE or task-relevant errors, uncertainty intervals and regime/window stratification with artifacts.
- [ ] Test price adjustments, missing days, transaction fees/slippage and out-of-sample backtest limitations.
- [ ] No profitable-trading claim without a robust verified methodology and results.

## P3 — Agent quality, operations and cloud
- [ ] LangGraph nodes access only constrained, provenance-labelled tools with timestamps and no secret exposure.
- [ ] Agent returns structured observation / calculation / interpretation / limitations sections.
- [ ] RAG responses cite actually indexed supporting documents; test prompt-injection, unavailable data and citation failure.
- [ ] Golden agent trace tests: deterministic tool input/output, hallucination resistance, latency, token and cost budget.
- [ ] Structured health/quality alerts and runbooks for provider outage, stale market data, feed gaps and bad model output.
- [ ] Publish measurable replay throughput/p95/cost benchmarks and transparent dataset limitations.
- [ ] Terraform design for optional EventBridge → Lambda/short-lived ECS → S3, Athena/Glue as necessary, CloudWatch, IAM/OIDC and Secrets Manager.
- [ ] Obtain explicit paid AWS approval, cost plan and destroy path before any apply. Verify actually provisioned resources separately.

## P4 — Credible demo and professional evidence
- [ ] Responsive accessible site, loading screen, real error/empty states, functional controls and no broken URLs.
- [ ] README/API contracts/architecture match actual source→model→agent→UI path and distinguish mocks.
- [ ] Archive evaluation scripts, test logs, provider-policy checks, timestamps, CI artifacts and agent traces.
- [ ] Draft honest recruiter tagline, 120-word Featured summary, 250-word case study and interview walkthrough.
- [ ] Link only verified deploys/benchmarks/active checks; keep Sentinance as **AI Systems** flagship after evidence exists.

## Additional verified UTC-alert checkpoint (2026-11-13)
- Twelve new cases passed in GitHub Actions code SHA `d7b93233ba106d088bb61bcef9fb3ee885d5970c`; 151 total API tests passed, five warnings; MyPy and web tests/build passed. Ruff still failed (not a green end-to-end pipeline).
- On later source SHA `cbfbbad841db79418045a11ddac4ac2dbbb050c1`, [strict PR checks 43433851215](https://github.com/HuLaxx/Sentinance/actions/runs/43433851215) reported 68 format-clean files and 155 strict lint findings. Latest-head API CI requires its own completed test result.
- The issue affects in-memory anomaly/alert timestamps, not database column definitions or live market-data ingestion. Earlier timezone-naive alert strings are parsed as UTC; new alerts explicitly include a UTC offset.
- P0 security follow-up: an npm install job log reports vulnerabilities despite a separate passing Trivy scan. Reconcile `npm audit --json` against the scan's scope before claiming dependency security is complete.

## P0 database and multi-exchange checkpoint — 13 November 2026
- [Draft PR #5](https://github.com/HuLaxx/Sentinance/pull/5) contains seven code/test files initially, with subsequent async-driver and test-fixture repairs. It preserves existing PostgreSQL UTC-naive timestamp column types and does **not** migrate data.
- [CI run 43439519761](https://github.com/HuLaxx/Sentinance/actions/runs/43439519761): 166 API tests passed, one DB SQLite fixture failed on omitted required OHLC values. All six new exchange failure tests passed; MyPy and frontend passed. Corrected test fixture must be retested; do not claim 167 passing yet.
- [PR check 43440292164](https://github.com/HuLaxx/Sentinance/actions/runs/43440292164): 72/72 Python files formatter clean, 107 strict Ruff findings, security and commit checks passed. Overall CI is still **red**.
- Next P0: complete SQLite test verification, then work through the remaining 82 broad exception catches and 25 naive timestamps without suppression. Separately reconcile frontend `npm audit` findings.

## Release decision
- **Current gate:** API/web functional tests pass and Python formatting is fixed, but **strict Ruff and overall CI are failing**; full source → curated → forecast/agent → browser path remains unverified.
- **Next smallest PR:** fix first semantic batch of UTC time handling and provider exception policies with regression tests, then rerun strict quality.
- **Never claim:** live market coverage, profitable predictions, secure autonomous financial advice or AWS deployment merely because code/template exists.
