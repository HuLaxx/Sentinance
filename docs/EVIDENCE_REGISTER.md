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
