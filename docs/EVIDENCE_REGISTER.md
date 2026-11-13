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
