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

## Verification log
- 2026-11-12: inspected repository tree, published README, CI files and recent job outcomes through GitHub.
- 2026-11-12: **No current API/web test suite, model backtest, full-stack demo or AWS deployment was independently executed** in this audit.
- Do not assign a verified status before linking a current run, dataset, output artifact, screenshot or deployment proof.
