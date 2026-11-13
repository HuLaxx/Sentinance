## P0 auth, stream reliability and security review — 12 November 2026

- [API run 43263271539](https://github.com/HuLaxx/Sentinance/actions/runs/43263271539): **134 passing API tests**, including four new JWT security tests and four Kafka handler/offset tests. No test failures. **Ruff 36 errors and MyPy 49 errors remain**, so the overall API job is red.
- The same run confirms **frontend ESLint, TypeScript and Next production build passed**.
- [PR checks 43263271473](https://github.com/HuLaxx/Sentinance/actions/runs/43263271473): blocking Trivy high/critical dependency scan **passed**, and real commit-message validation passed; Python code-quality gate remained red.
- Auth migrated from vulnerable python-jose to PyJWT **2.15.1**. Next.js and eslint-config-next upgraded to **16.4.6**; locked jsondiffpatch **0.7.6** and source-map-js **1.2.2**.
- **Open:** Ruff and MyPy defects, broader clean baseline security/license proof, replayable market ingestion, leakage-free forecasting and traceable agent quality. No AWS or production change.

# Sentinance — Engineering Evidence Register

Updated: 2026-11-12. Evidence is separated from architectural intent.

## Status definitions
**Verified** = exact current-commit execution artifact; **Implemented, unverified** = source exists but end-to-end behaviour has not been demonstrated; **Mock/optional** = explicit fallbacks or simulations; **Planned** = config/diagram only.

| Capability | Current status | Evidence | Next acceptance gate |
|---|---|---|---|
| Exchange provider adapters and normalized endpoints | Implemented, unverified | [Aggregator](../apps/api/multi_exchange.py), [API](../apps/api/main.py) | Contract tests, gap/backfill/retry fixtures, provenance |
| WebSockets / UI feed | Implemented, availability unverified | [API](../apps/api/), [frontend](../apps/web/) | Real seeded feed through browser; reconnect/stale states |
| Indicators and optional LSTM | Implemented, experimental / fallbacks | [Indicators](../apps/api/indicators.py), [predictor](../apps/api/predictor.py), [model docs](adr/009-lstm-price-prediction.md) | Walk-forward results against naive baselines, no leakage |
| LangGraph / RAG | Implemented, mock/optional providers | [Agent](../apps/api/agent.py), [RAG](../apps/api/rag_service.py) | Provenance-rich tool traces, outage and injection tests |
| dbt and data quality | Implemented, unverified | [dbt](../apps/dbt/), [quality](../apps/data_quality/) | Reproducible raw-to-mart ingest and schema assertions |
| K8s, Terraform, cloud services | Configured / planned, not independently deployed | [Infrastructure](../infra/), [README](../README.md) | Cost/security review, explicit approval, verified endpoint and teardown |
| API/Web CI | **Failed** in latest inspected main run | [GitHub Actions, 2026-11-06](https://github.com/HuLaxx/Sentinance/actions/runs/42282639705) | Current green checks with real assertions |

## New PR verification (2026-11-12)

- [PR #1 code quality run 43199449931](https://github.com/HuLaxx/Sentinance/actions/runs/43199449931) failed `ruff format --check`: **60 files require reformatting**, with 2 already formatted. This is pre-existing formatting debt, not a verified runtime failure.
- [PR #1 CI run 43199450283](https://github.com/HuLaxx/Sentinance/actions/runs/43199450283) failed API Ruff lint and web ESLint. The revised workflow now collects formatter, linter, typecheck, test and build outcomes before enforcing a failing gate; check its latest run for complete diagnostics.
- The new deployment guard requires a manually dispatched workflow on `main` with the exact `DEPLOY` confirmation, and preserves the kubeconfig environment between steps. **No deployment has been run**.

### API test recovery

After adding the missing Pydantic email validator, [run 43200615578](https://github.com/HuLaxx/Sentinance/actions/runs/43200615578) progressed to **122 passed, 4 failed**. The remaining failures were three test-environment JWT secret omissions and a bcrypt/Passlib incompatibility. A test-only secret and temporary bcrypt major-version constraint are now committed; their effectiveness must be verified from the newest run. Ruff, MyPy and frontend lint remain failing independently.

## P0 workflow finding
The frontend package uses Next.js 16 but its lint script invoked `next lint`, a removed CLI command. The main workflow and PR checks also used `continue-on-error: true` for lint, typing and/or tests, which can mask failure. The amended workflow records each tool's result and finishes with a required failing aggregate gate; treat red status as a real blocker rather than disabling tests.

The exact historical failure traceback was unavailable during this audit. Other dependency/build failures may remain; do not assume correcting lint configuration resolves every job.

## Required golden-path proof
1. Replay licensed or synthetic timestamped feed into raw and validated/curated records.
2. Verify UTC ordering, duplicates, late events, provider outage and replay deterministically.
3. Reproduce technical indicators and walk-forward forecast baselines with archived results.
4. Trace one LangGraph response to dated provider/curated observations without invented prices.
5. Show live, derived, model-generated, LLM-generated and mock states separately in the UI.

## Priority 1 — additional verified evidence (2026-11-12)

- [PR run 43201591473](https://github.com/HuLaxx/Sentinance/actions/runs/43201591473) completed **126 passing API tests** with no test failures. API Ruff and MyPy remain red, so overall CI failed.
- A genuine parse error in `enhanced_ai.py` was repaired; a follow-up MyPy run exposed duplicate module names for `core.redis` and then `managers.connection_manager`. Missing Python package initializers were added; their final static-analysis results remain unverified.
- The full web build passed in the inspected diagnostics, while frontend ESLint and type checking were red. Work is not production-ready simply because tests pass.
- Manual deployment approval remains enforced; no AWS account was accessed and no production resources were changed.

## Priority 1 — deterministic AI-analysis and type-safety work (2026-11-13)

- [PR CI run 43285213238](https://github.com/HuLaxx/Sentinance/actions/runs/43285213238) executed **139 passing API tests** including five new evidence-only analyst tests; the web CI job passed. Its aggregate API gate correctly failed on Ruff (E731, optional LangGraph lambda fallback) and 33 MyPy errors in the analyst and OpenTelemetry module. The GitHub UI's individual green step *conclusion* was misleading because those steps used `continue-on-error`; the actual gate outcomes were failures. This is a **red CI run**, not a verified green release.
- Changed the optional LangGraph metadata fallback to a properly typed `None` marker, introduced `TypeGuard[int | float]` for finite metrics and added an explicit trace module availability check. Later workflows must verify these changes.
- The analyst now refuses to convert placeholder whale strings into transfer claims, refuses missing/nonfinite price and indicator values, avoids unsupported MACD crossover and moving-average claims, and labels sentiment as a keyword heuristic. The returned numerical `confidence` is **source evidence coverage**, not a model reliability probability; this should be relabelled in public UX.
- [PR quality run 43285213422](https://github.com/HuLaxx/Sentinance/actions/runs/43285213422) remained red, with **12 files needing formatting** and **196 Ruff policy findings** in its strict checker, plus a successful frontend quality gate. Eleven redundant E402 suppression comments have since been removed without disabling lint rules; remaining policy/formatting debt is not resolved.
- All changes are on the draft PR branch; no AWS account, live deployment or model benchmark was verified in this work.

## P0 API gate success (verified 2026-11-13)

- [Successful CI run 43286247680](https://github.com/HuLaxx/Sentinance/actions/runs/43286247680), commit `8c406b12b300647b4f70e3c9196d658223ca754e`, reports **139 passed (5 warnings)** and a passing web job.
- API job log's required aggregate gate explicitly reports `Ruff: success | MyPy: success | pytest: success`. Unlike the earlier continuing-on-error statuses, this is a **verified successful API quality gate**.
- The PR-wide stricter quality workflow remains **red** for formatting and additional policy findings. The current branch contains later non-functional lint-comment/doc changes whose own latest checks must be consulted. A passing CI job does not establish a complete real-time market-data-to-agent integration or model skill.
- No AWS or production resources were provisioned.

## Verification log
- 2026-11-12: inspected repository tree, published README, CI files and recent job outcomes through GitHub.
- 2026-11-12: **No current API/web test suite, model backtest, full-stack demo or AWS deployment was independently executed** in this audit.
- Do not assign a verified status before linking a current run, dataset, output artifact, screenshot or deployment proof.

## P0 strict-quality cleanup checkpoint — 13 November 2026

- [Draft stacked PR #2](https://github.com/HuLaxx/Sentinance/pull/2) is based on P0 PR #1; main is untouched.
- [One-time formatter run 43426044240](https://github.com/HuLaxx/Sentinance/actions/runs/43426044240) executed pinned Ruff 0.17.5: **12 Python files reformatted, 35 safe lint fixes applied, 165 findings remain**. The one-time write-enabled formatter workflow was removed from the review branch immediately after its successful execution.
- [Current API/web workflow 43426353222](https://github.com/HuLaxx/Sentinance/actions/runs/43426353222) on code SHA `d118bbda41b2bbedcf27076a18774ee9bbc23550`: **139 API tests passed (5 warnings)**; MyPy and pytest succeeded, frontend test/build/lint job succeeded. The overall API CI is **red** because Ruff still fails. This is a real blocker, not a test failure.
- [Strict PR quality run 43426353182](https://github.com/HuLaxx/Sentinance/actions/runs/43426353182) on the same code SHA: `ruff format --check` passed (**67 Python files already formatted**); strict `ruff check .` reported **165 remaining findings**; security scan and commit-message check passed. Do not mark PR green.
- The two checks now run on `fix/**` push branches in addition to main-targeting PRs, so stacked work can be inspected before merging. This does not authorize production deployment; deploy guards remain unchanged.
- **Next fixes must be semantic and reviewed:** timezone-naive `datetime.utcnow()` paths (DTZ003), overly broad exception handling (BLE001) and import-order/runtime-initializer placement (E402). Do not blanket-disable rules; add regression tests where timestamp or failure handling changes.
- The verified 139 unit/API tests and frontend build do **not** prove a live market feed -> curated tables -> forecast baselines -> LangGraph trace -> browser golden path, provider licensing, cloud hosting or trading performance. No AWS resources provisioned.

## P0 UTC anomaly and alert contract — 13 November 2026
- [Stacked draft PR #3](https://github.com/HuLaxx/Sentinance/pull/3) contains changes limited to `apps/api/anomaly_detection.py`, `apps/api/anomaly_alerts.py`, and `apps/api/tests/unit/test_anomaly_utc.py`, alongside this evidence/checklist documentation. This is **not** a live exchange feed integration or an update to database timezone storage.
- [CI run 43433535905](https://github.com/HuLaxx/Sentinance/actions/runs/43433535905), source commit `d7b93233ba106d088bb61bcef9fb3ee885d5970c`: **151 API tests passed, five warnings**, of which **12 newly parameterized/scoped timezone and cooldown cases passed**; MyPy and frontend CI passed, while overall API job was red because strict Ruff remained red.
- [PR quality run 43433851215](https://github.com/HuLaxx/Sentinance/actions/runs/43433851215), code SHA `cbfbbad841db79418045a11ddac4ac2dbbb050c1`: **68/68 Python files Ruff format clean**, **155 remaining strict Ruff findings** (87 BLE001, 37 DTZ003, plus smaller categories). The parent branch had 165, so this is a verified reduction of ten timezone findings, **not complete P0**.
- New market anomalies and in-memory alert records use timezone-aware UTC timestamps; previous naive UTC strings are interpreted as UTC when sorting/filtering; legacy naive cooldowns are normalised. Tests cover explicit offsets, Z, timezone-naive legacy input, 24h filtering, ordering, cooldowns and failure of one subscriber.
- Separate PR/API tests on the latest documentation commit require new verification before claiming current-head CI outcomes. No main merge or paid AWS resources.
- **Security scope caution:** an npm CI install log reported dependency vulnerabilities even though the separate Trivy scan passed; reconcile scan scope and actual installed frontend lockfile before asserting clean dependencies.

## P0 typed FastAPI dependencies and contract-preserving lint cleanup — 13 November 2026

- [Draft PR #4](https://github.com/HuLaxx/Sentinance/pull/4) is stacked on P0 UTC-alert PR #3; implementation commit `0fceb37411f2c397b9b5b7caa4ed46433957231b` changed 15 source/test paths, followed by a four-file Ruff-format-only commit `d92d9584c923157a57464db39d889b3f6d6fcdea`. No main merge or production deployment.
- [API CI 43435715012](https://github.com/HuLaxx/Sentinance/actions/runs/43435715012): **155 Python/API tests passed, five warnings**; all four new contract tests passed, including anonymous alert-list/delete rejection, Pydantic market analysis schema example and query classification. MyPy and web CI passed. Aggregate API job remained **red** because strict Ruff failed.
- [Strict PR checks 43436003716](https://github.com/HuLaxx/Sentinance/actions/runs/43436003716) on formatted code SHA `d92d9584c923157a57464db39d889b3f6d6fcdea`: **69 files Ruff-format clean**, strict lint **124 findings** (87 BLE001 broad catches, 37 DTZ003 naive UTC timestamps). Earlier P0 branch had 155; 31 findings cleared without adding exclusions/ignores. Web lint and high/critical security scan passed. This is **not a green release**.
- Scoped corrections: eight FastAPI `Depends` defaults replaced by `Annotated` dependency declarations; intentional constants marked `ClassVar`; previously unused unpacked variables and redundant iteration removed; Pydantic schema example preserved; native Python 3.11 `fromisoformat` correctly parses Z; monitoring condition flattened.
- Formatting-only commit requires exact-head API CI verification before making a final latest-head full test claim.
- **Remaining P0 priorities:** 87 BLE001 exception handling findings should be resolved with provider-specific error and outage policies, not blanket suppression; 37 DTZ003 timestamps must be reconciled with DB column types and serializer contracts before mass replacement. Frontend dependency scan scope vs `npm audit` must also be reconciled.
- Nothing in this PR establishes provider licensing, historic feed replay, validated ML predictions, an agent golden path, cloud infrastructure or a production live user flow.

## P0 exchange failure contracts and legacy-UTC SQL boundary — 13 November 2026
- [Draft PR #5](https://github.com/HuLaxx/Sentinance/pull/5) implements named UTC-naive database storage helpers without a schema change; SQLAlchemy defaults and repository writes now share the same explicit UTC policy. `database.py` normalizes bare PostgreSQL URLs to `postgresql+asyncpg`, fixing test-environment async engine driver selection. Local SQLite tests do **not** prove PostgreSQL migration correctness.
- Multi-exchange provider adapters narrow blind `Exception` catches to HTTP/network and malformed payload failures; 429/503 and unavailable Coinbase statistics no longer create fabricated data; unexpected adapter programming exceptions bubble rather than masquerading as successful empty exchange data. Fallback exchange rows are not duplicated. New market timestamps include a UTC offset.
- [API run 43439519761](https://github.com/HuLaxx/Sentinance/actions/runs/43439519761) on earlier source revision: **166 tests passed, one failed** because the new SQLite price-fixture row omitted required high/low values; all **six provider tests passed**, while MyPy and frontend checks succeeded. That fixture was corrected on code commit `45ec0b0348cbb613a71f368373868b93e285b9ac`. [API CI 43440291737](https://github.com/HuLaxx/Sentinance/actions/runs/43440291737) subsequently verified **167 passing tests, five warnings, with MyPy and web build passing**; the overall API job is still **red solely due to strict Ruff findings**.
- [Strict PR checks 43440292164](https://github.com/HuLaxx/Sentinance/actions/runs/43440292164) on same corrected code: Python Ruff formatter **72/72 files clean**; **107 remaining strict Ruff findings** (82 BLE001, 25 DTZ003); security-scan/commit checks passed. This is **not** a green whole-project release.
- No existing PostgreSQL timestamps were modified or migrated. Cloud/IaC, licensed market feeds, successful forecasters and actual production/deployed status remain unverified. `npm audit` dependency risk needs separate reconciliation with Trivy's scan.

## P0 Redis rate-limit failure policy — 13 November 2026
- [Draft PR #6](https://github.com/HuLaxx/Sentinance/pull/6), stacked on provider/UTC PR #5: `apps/api/cache.py` now denies rate-limit calls on missing Redis or Redis counter/expiry errors (instead of returning `True`), validates positive thresholds/windows, avoids reusing failed-ping clients, and catches `redis.RedisError` / cache JSON decode errors explicitly. Broad catches reduced by 12 without lint suppression; programmer and encoding faults are not reported as cache outages.
- [API run 43445179266](https://github.com/HuLaxx/Sentinance/actions/runs/43445179266), source SHA `915102a169e81a7228d51f04509474e889a2a78c`: **182 tests passed, five warnings** including **15 new Redis/cache failure policy regression cases**; MyPy and frontend checks succeeded. Overall workflow still fails strict Ruff.
- [PR Checks 43445417231](https://github.com/HuLaxx/Sentinance/actions/runs/43445417231), formatter-corrected code SHA `76f102df11ee3615a866354cab53f05974ab4db5`: **73/73 Python files Ruff-format clean**, **95 strict lint findings** (70 BLE001 broad catches, 25 DTZ003 naïve UTC timestamps); web lint and security scan passed. Previous code branch contained 107. **Do not claim a green full CI gate**.
- Security caveat: `check_rate_limit` is a helper only until actual endpoint/middleware wiring is verified. INCR and conditional EXPIRE are two operations, not an atomic sliding-window counter, and a post-INCR failure can leave state ambiguous. Required next: API-level rate limit enforcement/integration tests, safe error status contract, atomic multi-replica limiter, provider outage metadata and npm vulnerability reconciliation.
- The work is draft only; no main-branch merge, production deployment, AWS provisioning, model evaluation, real licensed live feed or complete UI golden path was established.

## P0 atomic auth-route rate limit verification — 13 November 2026
- [Draft PR #7](https://github.com/HuLaxx/Sentinance/pull/7), stacked on #6, changes the previously split `INCR`/`EXPIRE` counter into **one atomic Redis Lua command** and binds a fail-closed dependency to `POST /api/auth/login`, `/register` and `/refresh`. The rate limiter differentiates legitimate quota exceeded (HTTP 429, Retry-After 60) from Redis unavailable (HTTP 503, Retry-After 5). Per-peer address/route identifiers are SHA-256 digests and forwarded IP headers are not directly trusted.
- [CI run 43450390878](https://github.com/HuLaxx/Sentinance/actions/runs/43450390878) source commit `48a2ccd9d058bc06bb548fba4cebc64a9ee6c129`: **194 Python/API tests passed, 5 warnings**, including API-level auth rejection paths and *real Redis 7* concurrent increment tests (20 requests; exactly 7 admitted), no-expiry legacy counter repair, and loop-isolated Redis transport tests. MyPy, web lint/build succeeded. **Overall run is red from 95 remaining Ruff findings.**
- [PR quality checks 43450658775](https://github.com/HuLaxx/Sentinance/actions/runs/43450658775), formatting-only revision `d49f46aeb964a49b51cfaa5067a8493d67de5978`: **76 files correctly formatted**, 95 remaining Ruff (70 BLE001, 25 DTZ003). Security scan passed. No lint rules were globally suppressed.
- CI initially found a real async-client cross-event-loop reuse issue and an inaccurate Redis `EVAL` type-stub signature; these were corrected in code along with input fixture validation. The passing API run applies to the corrected source revision prior to the later formatting-only change.
- **Still not production-ready:** IP attribution behind reverse proxies may collapse many clients into one quota; the fixed-window algorithm is not sliding-window fair; Redis failover, cross-replica load, end-to-end authenticated user flows beyond fixtures, npm dependency risks and full live-market data path need independent evidence. No main merge, paid resources, AWS deployment or externally verified live service.

## P0 UTC market/event emissions — 13 November 2026
- [Draft PR #8](https://github.com/HuLaxx/Sentinance/pull/8), stacked on #7, replaces **13 naive UTC timestamp calls** across `ab_testing.py` (4), `kafka_producer.py` (3), `market_stats.py` (1), `predictor.py` (1), `services/price_stream.py` (1), `trading_signals.py` (2), and `news_scraper.py` (1), with explicit `datetime.now(UTC)` and UTC-offset ISO strings. The news article adapter normalizes naive timestamps using an explicit **UTC assumption** and offset-aware timestamps via conversion; provider time-zone semantics still need validation.
- [API CI 43454910082](https://github.com/HuLaxx/Sentinance/actions/runs/43454910082), source SHA `a88255204106695d9dcb5596fdb3c01928ae552b`: **201 passing API tests, five warnings** including **seven new UTC regression cases** (experiments, mock Kafka envelopes, market statistics, mock predictions, fake-feed price history, signal metadata, mixed-timezone RSS sorting). MyPy succeeded and frontend CI succeeded. Overall workflow is red **only due to strict Ruff**.
- [PR checks 43455287738](https://github.com/HuLaxx/Sentinance/actions/runs/43455287738), formatting-only revised SHA `72801cfcf6aedb61beee5dddcd2b5dae5cd976b8`: **77/77 Python files formatter-clean**; strict Ruff now **82 findings (70 BLE001, 12 DTZ003)**, down from 95. Security-scan and web quality jobs passed. No global ignore rules or lint suppression.
- Out of scope and unverified: actual Kafka cluster delivery, licensed exchange event timestamps, forecasts and ML backtests, upstream provenance, source-to-browser golden path, complete dependency audit, cloud deployment, and acceptance of historical PostgreSQL time migrations. All work remains in a stacked DRAFT PR for review.


## P0 explicit RSS provider failures — 13 November 2026

- [Draft PR #9](https://github.com/HuLaxx/Sentinance/pull/9) base #8; code SHA `2e280503b7f9195ab2a3308cc4c077e55de88cdd`, Ruff-format revision `39047847e1db996b71146e59ebdea22168a59961`; changed `apps/api/news_scraper.py` and added `apps/api/tests/unit/test_rss_failure_policy.py` (six synthetic regression cases).
- [CI 43460271229](https://github.com/HuLaxx/Sentinance/actions/runs/43460271229) on code SHA: **207 API tests passed**, five warnings; MyPy success, frontend job success, overall API red due to existing Ruff debt. Cases assert HTTP 429, HTTP 503, timeout isolation, parser exception surfacing, mixed-source isolation/logging, and cancellation propagation.
- [Strict PR check 43460702032](https://github.com/HuLaxx/Sentinance/actions/runs/43460702032) on formatter SHA: **78/78 files format clean, 80 strict Ruff findings remain**, from 82 before this PR. Security scan, commit message, frontend lint succeeded; aggregate strict gate red. No globally disabled rule.
- [Exact-head API CI 43460702036](https://github.com/HuLaxx/Sentinance/actions/runs/43460702036) on formatted code SHA `39047847e1db996b71146e59ebdea22168a59961`: **207 tests passed, five warnings**, MyPy success, web success; aggregate API gate red solely due to Ruff. This confirms the post-format source and tests, but does not make strict CI green.
- Observability scope: only server-side structured logs expose outage/source failures; no new API outage status contract or user-visible freshness indicator. No real live feed, licensed data, broker delivery or production deploy claimed.

## P0 Kafka producer acknowledgment and transparent failures — 13 November 2026

- [Stacked draft PR #10](https://github.com/HuLaxx/Sentinance/pull/10) on #9: `apps/api/kafka_producer.py` changes `send()` to `send_and_wait()`, requests `acks="all"` plus idempotent producer retries, propagates known startup/delivery errors and does not cache failed singleton startup. Updated `apps/api/tests/unit/test_utc_event_boundaries.py` for the new method and added `apps/api/tests/unit/test_kafka_producer_ack.py` with 12 mocked cases. Code SHA `5371f9cecbb4f14433f5bcfbc05d30cb65c74329`; subsequent formatter-only source revision `e71f1e7d5fa18a1e79c86683de3d6551e3671d17`.
- [API CI 43462637843](https://github.com/HuLaxx/Sentinance/actions/runs/43462637843) verified **219 tests passed (5 warnings)**, MyPy and frontend CI passed on the *initial* source SHA; aggregate API failed Ruff.
- [Quality run 43462922969](https://github.com/HuLaxx/Sentinance/actions/runs/43462922969) on corrected source SHA verified **79 Python files format-clean**, **76 strict Ruff errors** (64 BLE001 and 12 DTZ003), security scan and frontend lint success; overall PR quality job failed strict lint. Exact source-SHA unit/API run for the corrected revision must be confirmed separately.
- [Exact formatted-source API CI 43462922589](https://github.com/HuLaxx/Sentinance/actions/runs/43462922589) on `e71f1e7d5fa18a1e79c86683de3d6551e3671d17` independently confirms **219 passing API tests**, five warnings, MyPy and frontend successful; required aggregate API gate failed solely on Ruff.
- The tests use an in-memory mock of the broker API: no live Kafka broker, delivery trace, persistence, producer-consumer workflow, exactly-once consumer processing, outbox, data licence, cloud provisioned infrastructure or end-user notification was exercised. A broker timeout can leave write outcome uncertain; application retries may require idempotent event identifiers/deduplication.


## P0/P1 real-broker Kafka consumer-lifecycle checks — 13 November 2026

- [Stacked draft PR #11](https://github.com/HuLaxx/Sentinance/pull/11) on #10; changed `apps/api/kafka_consumer.py`, existing `apps/api/tests/test_kafka_delivery_semantics.py`, added `apps/api/tests/unit/test_kafka_consumer_lifecycle.py` and `apps/api/tests/integration/test_kafka_broker_roundtrip.py`, and configured separate ephemeral broker job in `.github/workflows/ci.yml`. Initial source SHA `96911c776af4c7864adf6f70fa3e0d4141002513`, formatting/CI-path fix `aacd311fbbfb1cc760563781aabcef1b1407fbd5`, tightened failure assertion `a415d7559b6708757a64c880ee7e6d219ea02ca6`.
- [Real Kafka broker workflow 43472276017](https://github.com/HuLaxx/Sentinance/actions/runs/43472276017) on `aacd311...`: 2/2 integration tests passed against `apache/kafka:3.9.1` on GitHub-hosted ephemeral Docker. [Later broker workflow 43472651801](https://github.com/HuLaxx/Sentinance/actions/runs/43472651801) on `a415d755...` independently reconfirmed **2/2**, including failure-record observability. Success is limited to the observed group/offset/consumer behaviours with synthetic generated events.
- [API workflow 43472276017](https://github.com/HuLaxx/Sentinance/actions/runs/43472276017) on `aacd311...`: **223 tests passed, 2 skipped, five warnings**, MyPy passed, web CI passed; two integration cases intentionally skipped outside the broker job. Aggregate API workflow **FAILED** strict Ruff.
- [Quality workflow 43472652016](https://github.com/HuLaxx/Sentinance/actions/runs/43472652016) on `a415d755...`: **81 files formatter-clean, 75 strict Ruff findings** (63 BLE001, 12 DTZ003), security scan/commit check/frontend lint passed, aggregate quality job failed Ruff. Exact-head API verification is separate.
- [Exact-source CI 43472651801](https://github.com/HuLaxx/Sentinance/actions/runs/43472651801), source SHA `a415d7559b6708757a64c880ee7e6d219ea02ca6`: **223 API tests passed, 2 integration tests skipped in standard API suite, 5 warnings**, MyPy/frontend successful, and **separate ephemeral Kafka broker job 2/2 passed**. Aggregate API job failed only because strict Ruff still has 75 findings, not because tests failed.
- No production broker, multi-node Kafka, source-to-DB durable sink, user-visible AI pipeline, historical deduplication, network partition chaos testing, paid cloud or deployment verified. Replay with `earliest` in an uncommitted new group does not guarantee replay when the deployed consumer continues to use default `latest`; operating policy must be deliberately selected.


## P0 last UTC sites and Redis/WebSocket reliability — 13 November 2026

- [Draft PR #12](https://github.com/HuLaxx/Sentinance/pull/12) source commit `30f90b4eab4c04bb099b22adbc1570dfdabc6598`, formatting revision `2df4f169a59777ddab8db733f2715383bc31a66a`: changed `enhanced_ai.py`, `llm_wrapper.py`, `main.py`, `rag_service.py`, `streaming_llm.py` and three `train_*.py` paths, adding `tests/unit/test_remaining_utc_contracts.py` (7 synthetic tests). All **12 naive-UTC calls** in remaining strict-lint findings converted to offset-aware UTC; RSS missing-zone assumed UTC, not externally validated. Historical prediction ID and UTC timestamp derived from one instant, not a proof of actual prediction reliability.
- [CI 43478543609](https://github.com/HuLaxx/Sentinance/actions/runs/43478543609) on exact formatter SHA `2df4f169...`: **230 API passed, 2 broker-only skipped, 5 warnings**, MyPy/web/real-ephemeral-broker jobs passed; overall API gate failed strict Ruff. [PR checks 43478543594](https://github.com/HuLaxx/Sentinance/actions/runs/43478543594): **82 formatted files, 63 BLE001, zero DTZ003**, security/frontend passed; strict quality gate still red.
- [Draft PR #13](https://github.com/HuLaxx/Sentinance/pull/13) source `65e675cdb7d96d4f8c62385e629680a542623511`, test import style revision `74b27182060630b5d285eebf972b784adbe6035e`. Modified `apps/api/managers/connection_manager.py`, added 8 unit tests in `apps/api/tests/unit/test_redis_websocket_transport.py`. Known Redis exceptions now log+raise, expected closed sockets are isolated, generic failures are not masked, and listener shutdown awaits cleanup. No live Redis failover or cross-replica WebSocket fanout was executed.
- [CI 43479281852](https://github.com/HuLaxx/Sentinance/actions/runs/43479281852) on initial code SHA: **238 API passed, 2 skipped, 5 warnings**, MyPy/frontend/real broker passed, aggregate API failed Ruff. [PR checks 43479723536](https://github.com/HuLaxx/Sentinance/actions/runs/43479723536) on formatter/test-import SHA: **83 files formatter-clean, 60 BLE001, 0 DTZ003**, security, frontend lint and commit checks success; aggregate PR quality failed on remaining strict lint. [Exact corrected-source CI 43479723686](https://github.com/HuLaxx/Sentinance/actions/runs/43479723686) on source SHA `74b27182060630b5d285eebf972b784adbe6035e` reconfirmed **238 API tests passed, 2 broker-only skips, 5 warnings**, MyPy and web passed and **2/2 real Kafka integration tests passed**. Aggregate API red solely from 60 existing Ruff findings.
- No main merge, AWS provision, licensed market-data validation, model benchmark, production deployment or complete source-to-browser experience. All verified changes are stacked **draft review branches**; tests are synthetic except the independent two real ephemeral Kafka broker tests inherited from PR #11.


## P0 researched market provider failure contracts — 14 November 2026

- [Draft PR #14](https://github.com/HuLaxx/Sentinance/pull/14) stacks on PR #13 without modifying `main`; source `apps/api/agent.py` narrows all six broad catches. Synthetic research paths distinguish provider HTTP failures and malformed JSON from unexpected programming faults. Binance order-book and kline HTTP 4xx/5xx no longer silently create absent measurements. Added `tests/unit/test_agent_provider_failures.py`, 13 offline cases.
- [Code-source API CI 43488686694](https://github.com/HuLaxx/Sentinance/actions/runs/43488686694), SHA `774d351ce8faf8c39572038d57545bdeb909bbe3`: **251 API tests passed, 2 broker-only skipped, 5 warnings**; MyPy passed; frontend passed; independently run **2/2 ephemeral Kafka broker tests passed**. API aggregate **FAILED strict Ruff**, not pytest.
- [Formatter-corrected strict PR checks 43489001591](https://github.com/HuLaxx/Sentinance/actions/runs/43489001591), source SHA `147d78d7f39eb3bfc1c3c4520c11c4c77df9b34d`: **84 Python files formatter-clean**, **54 remaining BLE001** (previously 60), **zero DTZ003**. Commit checks and security scan passed; overall strict code-quality job **FAILED Ruff**.
- Exact corrected-code API workflow and subsequent docs-only heads require separate status checks. A green web/Kafka job and passing unit/API tests are not proof of all-green CI, real exchange data, Kafka-to-DB persistence, market forecasting accuracy, provider licensing or a verified deployed user path.
- No paid cloud provisioning, production deployment or main-branch merge.

## P0 14 November 2026 checkpoint — typed subsystem failures and legacy CI gates

- [PR #15](https://github.com/HuLaxx/Sentinance/pull/15) through [PR #23](https://github.com/HuLaxx/Sentinance/pull/23) are **stacked drafts**, not merged to `main`. Small slices narrowed broad catches in Gemini/LLM streams, structured retry, Qdrant/RAG, chat API, price stream, historical memory, legacy smoke scripts, data downloads and alert fanout. Synthetic regression tests added and previous passing services preserved in review runs.
- Inspected strict lint trend: **54 BLE001** after PR #14, then **49** (#15), **44** (#16), **40** (#17), **34** (#18, plus 1 import-order finding), **26** (#19, plus 1 import-order finding), then **zero BLE001** on [PR #22 quality run 43495952790](https://github.com/HuLaxx/Sentinance/actions/runs/43495952790) with **six formatting differences and one import-order finding**. PR #23 attempts to resolve these, and its final checks must independently confirm green.
- [PR #18 API CI 43493834929](https://github.com/HuLaxx/Sentinance/actions/runs/43493834929): **291 tests passed, two broker-only skipped**, MyPy/web and real Kafka broker passed; overall API failed Ruff. [PR #19 API CI 43494261202](https://github.com/HuLaxx/Sentinance/actions/runs/43494261202): **302 tests passed, two skipped**, MyPy/web and Kafka passed; overall API failed Ruff.
- [PR #20 API run 43494921721](https://github.com/HuLaxx/Sentinance/actions/runs/43494921721): **310 tests passed, one failed** because an older test directly instantiated a wrapper without `_last_provider`; an additional correction exists on later branch. Do not represent the 310 passes as a green suite.
- [PR #23](https://github.com/HuLaxx/Sentinance/pull/23) addresses legacy API lint independently of strict PR checks: Redis counter E721, deliberate environment-first E402 imports and obsolete test sys.path mutation. Narrow line-scoped E402 annotations remain intentional for module bootstrapping, not a blanket disable.
- A successful independent security scan coexists with npm install output reporting **30 vulnerable packages (two critical)**. Run a scoped `npm audit` and pin/replace affected packages before calling the dependency audit resolved.
- **Still unverified:** final exact-head whole CI pass, real licensed exchange feeds, market-source freshness and provenance, full persistent Kafka-to-DB pipeline, leakage-free model benchmark, deployed AI source-to-browser demo, cloud production or any profitable trading claim. No paid AWS deployment or main merge.

