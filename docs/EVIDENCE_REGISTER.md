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

## P0 workflow finding
The frontend package uses Next.js 16 but its lint script invoked `next lint`, a removed CLI command. The main workflow and PR checks also used `continue-on-error: true` for lint, typing and/or tests, which can mask failure. Change the workflow to fail on real regressions; treat any new red statuses as genuine blockers rather than disabling tests.

The exact historical failure traceback was unavailable during this audit. Other dependency/build failures may remain; do not assume correcting lint configuration resolves every job.

## Required golden-path proof
1. Replay licensed or synthetic timestamped feed into raw and validated/curated records.
2. Verify UTC ordering, duplicates, late events, provider outage and replay deterministically.
3. Reproduce technical indicators and walk-forward forecast baselines with archived results.
4. Trace one LangGraph response to dated provider/curated observations without invented prices.
5. Show live, derived, model-generated, LLM-generated and mock states separately in the UI.

## Verification log
- 2026-11-12: inspected repository tree, published README, CI files and recent job outcomes through GitHub.
- 2026-11-12: **No current API/web test suite, model backtest, full-stack demo or AWS deployment was independently executed** in this audit.
- Do not assign a verified status before linking a current run, dataset, output artifact, screenshot or deployment proof.
