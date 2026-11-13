# Sentinance — Engineering Release Checklist (evidence-gated)

**Updated:** 2026-11-13 | **Target:** credible AI Systems / Market Data Engineering flagship | **Status:** DRAFT, P0 STRICT LINT RED, NOT 10/10

[x] denotes a **verified limited claim on a review branch**, not a main-branch merge, production deployment or full-stack certification. Support factual claims in [EVIDENCE_REGISTER.md](EVIDENCE_REGISTER.md). Always distinguish source market observations, derived indicators, forecasts, language-model interpretations and synthetic/mocked output. Never promise trading returns.

## P0 — CI, truthful claims and provider safety
- [x] **201 API tests passed, five warnings** on code SHA `a88255204106695d9dcb5596fdb3c01928ae552b`, including seven UTC emission/normalization tests; MyPy and web checks passed. **Overall CI remains red due to strict Ruff.** [CI run 43454910082](https://github.com/HuLaxx/Sentinance/actions/runs/43454910082).
- [x] **77/77 Python files pass Ruff formatting** on code SHA `72801cfcf6aedb61beee5dddcd2b5dae5cd976b8`. [PR Checks 43455287738](https://github.com/HuLaxx/Sentinance/actions/runs/43455287738).
- [x] Security and commit-message check jobs pass on same review branch; a passing security scan is not an audit of all secrets. [PR Checks 43428370907](https://github.com/HuLaxx/Sentinance/actions/runs/43428370907).
- [ ] Fix **82 remaining strict Ruff findings** without blanket suppressions: **70 BLE001 broad exception handlers + 12 DTZ003 UTC-naive timestamps**. [Strict PR checks 43455287738](https://github.com/HuLaxx/Sentinance/actions/runs/43455287738).
- [x] Add **12 scoped unit regression cases** for UTC-aware market anomaly alerts, offset-based ordering, legacy UTC-naive readings, cooldowns and subscriber error isolation. [PR #3](https://github.com/HuLaxx/Sentinance/pull/3), [151 passing API tests on initial code revision](https://github.com/HuLaxx/Sentinance/actions/runs/43433535905). The database UTC contract remains unverified.
- [x] Replace eight FastAPI `Depends` defaults with type-safe `Annotated` dependency injection, retain unauthenticated alert-list/delete rejection, preserve Pydantic JSON schema example and model routing. All four new regression tests and 155 API tests passed on implementation SHA `0fceb37411f2c397b9b5b7caa4ed46433957231b`; formatting fixes were later applied. [PR #4](https://github.com/HuLaxx/Sentinance/pull/4), [API run 43435715012](https://github.com/HuLaxx/Sentinance/actions/runs/43435715012).
- [x] Verify strict Python formatter clean across **69/69 files**. Overall PR checks intentionally remain red due to 124 remaining semantic lint findings. [Run 43436003716](https://github.com/HuLaxx/Sentinance/actions/runs/43436003716).
- [x] Establish a documented **UTC-naive storage convention** for existing SQLAlchemy `DateTime(timezone=False)` columns, with a named conversion helper; normalize `postgresql://` / `postgres://` into an explicit asyncpg URL. [Draft PR #5](https://github.com/HuLaxx/Sentinance/pull/5). **Schema migration not performed.** This is checked as an implemented local contract, not a verified PostgreSQL migration.
- [x] Verify six new offline exchange-provider regression tests: 429/503, malformed payload, fail-closed adapter programming errors, fallback deduplication and UTC stamps. [API run 43439519761](https://github.com/HuLaxx/Sentinance/actions/runs/43439519761) — provider tests passed; overall job had one unrelated SQLite test-fixture failure, since corrected but awaiting its own CI.
- [x] Verify 72/72 Python files Ruff-format clean on the code revision `45ec0b0348cbb613a71f368373868b93e285b9ac`. [PR checks 43440292164](https://github.com/HuLaxx/Sentinance/actions/runs/43440292164).
- [x] Verify **167/167 API tests passed, five warnings** (including corrected SQLite UTC database round trip and six new provider tests), with **MyPy + web CI passed**, on code SHA `45ec0b0348cbb613a71f368373868b93e285b9ac`. This verifies a local SQLite fixture only; PostgreSQL migration and historical-data correctness remain unverified. [API run 43440291737](https://github.com/HuLaxx/Sentinance/actions/runs/43440291737). Overall CI **still red** due to Ruff.
- [x] Verify **15 new Redis/cache regression cases**: missing Redis, counter/expiry timeouts, invalid limit settings, malformed JSON, programmer exceptions and healthy/unhealthy client lifecycle. The helper now **denies** rate-limit calls on Redis outage rather than bypassing the limit; 12 blanket exception catches removed. [Draft PR #6](https://github.com/HuLaxx/Sentinance/pull/6), [API test run 43445179266](https://github.com/HuLaxx/Sentinance/actions/runs/43445179266).
- [x] **Auth endpoints invoke the limiter:** login, register and refresh each execute a FastAPI rate-limit dependency. Tested HTTP 429 vs Redis-unavailable 503, allowed credential rejection and forwarded-header spoofing. [PR #7](https://github.com/HuLaxx/Sentinance/pull/7), [194 passing API tests](https://github.com/HuLaxx/Sentinance/actions/runs/43450390878).
- [x] Replace split INCR/EXPIRE with a **single Lua atomic fixed-window command**, test 20 concurrent real-Redis operations against a seven-request limit, repair counters without expiry, and preserve fail-closed semantics. This is **not a sliding-window algorithm**. [PR #7](https://github.com/HuLaxx/Sentinance/pull/7), [CI 43450390878](https://github.com/HuLaxx/Sentinance/actions/runs/43450390878).
- [x] Replace **13 timestamp-naive event emission calls** with explicit UTC in experiments, Kafka envelopes, market statistics, predictions, price history, signals and news fallback; normalize naive and offset-aware RSS input to UTC. **Seven new regression tests passed**, but provider/source timestamp attribution remains unverified. [Draft PR #8](https://github.com/HuLaxx/Sentinance/pull/8), [CI 43454910082](https://github.com/HuLaxx/Sentinance/actions/runs/43454910082).
- [ ] Validate real reverse-proxy peer attribution, NAT/shared-IP fairness, distributed Redis failover, multi-replica load behaviour and slow/down Redis timeouts; do not call this production-hardened.
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
- [CI run 43439519761](https://github.com/HuLaxx/Sentinance/actions/runs/43439519761): an earlier test revision had 166 passed and one incomplete SQLite OHLC fixture. That fixture was corrected; [run 43440291737](https://github.com/HuLaxx/Sentinance/actions/runs/43440291737) now verifies **167 passed, five warnings, MyPy/web green, overall API red only from strict Ruff**.
- [PR check 43440292164](https://github.com/HuLaxx/Sentinance/actions/runs/43440292164): 72/72 Python files formatter clean, 107 strict Ruff findings, security and commit checks passed. Overall CI is still **red**.
- Next P0: complete SQLite test verification, then work through the remaining 82 broad exception catches and 25 naive timestamps without suppression. Separately reconcile frontend `npm audit` findings.

## P0 Redis fail-closed verification (2026-11-13)
- [Source PR #6](https://github.com/HuLaxx/Sentinance/pull/6) stacked on provider/database PR #5: `cache.py` now handles known Redis errors explicitly and preserves visibility of programming faults, while cached invalid JSON returns no data; failed ping never installs an unhealthy Redis client.
- [Test run 43445179266](https://github.com/HuLaxx/Sentinance/actions/runs/43445179266) on source SHA `915102a169e81a7228d51f04509474e889a2a78c`: **182 tests passed**, including all **15** Redis/cache cases, MyPy success and web checks passed. Overall API job red only because existing Ruff gate is still failing.
- [PR checks 43445417231](https://github.com/HuLaxx/Sentinance/actions/runs/43445417231) on formatter-corrected SHA `76f102df11ee3615a866354cab53f05974ab4db5`: **73/73 Python files format-clean; 95 strict lint findings** (70 BLE001, 25 DTZ003). Trivy security job passed; npm audit and provider security remain separate acceptance items.
- **Do not claim live enforcement**: actual endpoint wiring was not demonstrated, Redis INCR/EXPIRE is non-atomic, and cross-replica tests, rate-limit response codes, and real broker/DB/provider behaviour remain outstanding. No main merge or cloud deployment.

## P0 atomic auth-boundary checkpoint (2026-11-13)
- [Draft PR #7](https://github.com/HuLaxx/Sentinance/pull/7), stacked on Redis failure-policy PR #6, implements a Redis Lua **atomic fixed-window** counter and routes it through FastAPI dependencies for login, registration and refresh. It distinguishes rate-limit exceeded (HTTP 429) from Redis unavailable (HTTP 503). Client identifiers derive from ASGI socket peer + route hash, not caller-provided forwarded IP headers.
- [CI 43450390878](https://github.com/HuLaxx/Sentinance/actions/runs/43450390878) on source SHA `48a2ccd9d058bc06bb548fba4cebc64a9ee6c129` verified **194/194 API tests passed** (five warnings), including two **real-Redis 7** integration tests for concurrent increments and expiry repair, plus auth dependency tests and a Redis event-loop lifecycle regression. MyPy and web checks passed. Overall CI is red from Ruff.
- [PR Checks 43450658775](https://github.com/HuLaxx/Sentinance/actions/runs/43450658775) on next code revision `d49f46aeb964a49b51cfaa5067a8493d67de5978` confirmed **76/76 files format-clean** and **95 strict Ruff findings** (70 BLE001, 25 DTZ003). Formatting-only revision followed the passing source test commit; full CI on that exact later revision is a separate check.
- Outstanding: trusted proxy configuration and shared-IP fairness, Redis outage recovery under load, all remaining Ruff/security/dependency findings, live market source correctness and complete browser golden path. No merge, rollout or paid infrastructure.

## P0 UTC event-boundary evidence (2026-11-13)
- [Draft PR #8](https://github.com/HuLaxx/Sentinance/pull/8), stacked on the atomic auth-boundary #7, changes **13 generated timestamp call sites** from naive UTC to explicit ISO 8601 UTC offsets across seven source modules. This is metadata correctness; it does **not** validate source exchange timestamps, production Kafka delivery, ML accuracy or trading returns.
- [CI 43454910082](https://github.com/HuLaxx/Sentinance/actions/runs/43454910082) on code SHA `a88255204106695d9dcb5596fdb3c01928ae552b`: **201/201 API tests passed, five warnings, MyPy passed and web checks/build passed**. All seven new event-time contract tests passed, including fake Kafka envelopes, in-memory market/experiment metadata, seeded price history, signal metadata and mixed RSS date conversion.
- [PR quality 43455287738](https://github.com/HuLaxx/Sentinance/actions/runs/43455287738) on formatter-corrected SHA `72801cfcf6aedb61beee5dddcd2b5dae5cd976b8`: **77/77 Python files correctly formatted**, **82 strict Ruff findings remain** (70 BLE001, 12 DTZ003). Security scan and frontend checks passed; overall quality job **still fails strict lint**. No lint suppression.
- **P0 remains open**: audited exception types (70), 12 more UTC sites with legacy behaviour review, npm audit reconciliation, trusted proxy policy and real multi-replica/failover validation. Separate P1 licensed source → curated data → UI test still missing.

## P0 RSS failure visibility and cancellation safety — 13 November 2026

- [Draft PR #9](https://github.com/HuLaxx/Sentinance/pull/9), stacked on #8, narrows `news_scraper.py` to `httpx.HTTPError` for expected HTTP status/transport failures; removes broad parser-error suppression, logs unexpected failures with the failing source in best-effort aggregation, and propagates feed cancellation.
- [API CI 43460271229](https://github.com/HuLaxx/Sentinance/actions/runs/43460271229) on implementation SHA `2e280503b7f9195ab2a3308cc4c077e55de88cdd` verified **207/207 tests passed, five warnings** (six new synthetic RSS cases), **MyPy passed**, web job passed. Aggregate API workflow failed because Ruff remains red.
- [PR quality 43460702032](https://github.com/HuLaxx/Sentinance/actions/runs/43460702032) on formatting-corrected SHA `39047847e1db996b71146e59ebdea22168a59961` verified **78/78 Python files formatter-clean** and **80 strict Ruff findings** (down from 82; two `BLE001` removed), with security/commit-message/frontend checks passing. Strict-lint and overall PR quality gates **remain red**.
- The source and test content changes are synthetic and unit-scoped; no live RSS provider delivery, uptime guarantees, UI-level outage state, complete source provenance or consumer end-to-end flow has been verified. **P0 release gate remains unchecked.** [Exact-head API CI 43460702036](https://github.com/HuLaxx/Sentinance/actions/runs/43460702036) on the formatter-corrected source SHA `39047847e1db996b71146e59ebdea22168a59961` independently reconfirmed **207 tests passed (five warnings)**, MyPy success, web success; overall workflow red from Ruff only.

## P0 Kafka broker-acknowledgment and failed-delivery visibility — 13 November 2026

- [Draft PR #10](https://github.com/HuLaxx/Sentinance/pull/10), stacked on #9, switches the mocked/unwired `PriceProducer` interface from local-buffer `send` to broker-acknowledged `send_and_wait` for price, alert and news envelopes. The constructor requests `acks="all"` and `enable_idempotence=True`; a missing or stopped producer now raises rather than silently dropping records. Known Kafka/timeout exceptions are logged and re-raised; unintended serialization/programming errors bubble.
- [Initial API CI 43462637843](https://github.com/HuLaxx/Sentinance/actions/runs/43462637843) on implementation SHA `5371f9cecbb4f14433f5bcfbc05d30cb65c74329`: **219 API tests passed, five warnings**, including **12 new mocked Kafka producer contract cases**; MyPy successful, frontend CI successful, overall API gate **red** due to outstanding Ruff findings.
- [PR quality 43462922969](https://github.com/HuLaxx/Sentinance/actions/runs/43462922969) on Ruff-format-corrected SHA `e71f1e7d5fa18a1e79c86683de3d6551e3671d17`: **79/79 Python files formatter-clean**, **76 strict Ruff findings** (64 BLE001, 12 DTZ003; down from 80), security scan/frontend lint/commit-message check successful. **Full strict CI remains red.** [Exact formatted-source API CI 43462922589](https://github.com/HuLaxx/Sentinance/actions/runs/43462922589) independently confirms **219 passing API tests (five warnings), MyPy success and frontend success** on `e71f1e7d5fa18a1e79c86683de3d6551e3671d17`. Overall API job remains red solely from Ruff.
- [ ] A **real broker** producer→topic→consumer integration with topic persistence, failures/restarts, duplicate/replay behavior and event-key/idempotency policy has **not** been tested. No event outbox or cross-service exactly-once claim is supported.
- [ ] **Full P0** still requires resolving 76 Ruff findings, proxy/rate-limit operational validation, dependency audit and other remaining P0 security/availability checks. No deployment or main-branch merge.

## Release decision
- **Current gate:** API/web functional tests pass and Python formatting is fixed, but **strict Ruff and overall CI are failing**; full source → curated → forecast/agent → browser path remains unverified.
- **Next smallest PR:** audit the remaining 64 broad catches and 12 UTC call sites in small functional slices with failure-policy tests; separately reconcile dependency audit findings.
- **Never claim:** live market coverage, profitable predictions, secure autonomous financial advice or AWS deployment merely because code/template exists.
