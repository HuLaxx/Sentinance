# Sentinance — Engineering Release Checklist (evidence-gated)

**Updated:** 2026-11-13 | **Target:** credible AI Systems / Market Data Engineering flagship | **Status:** DRAFT, P0 STRICT LINT RED, NOT 10/10

[x] denotes a **verified limited claim on a review branch**, not a main-branch merge, production deployment or full-stack certification. Support factual claims in [EVIDENCE_REGISTER.md](EVIDENCE_REGISTER.md). Always distinguish source market observations, derived indicators, forecasts, language-model interpretations and synthetic/mocked output. Never promise trading returns.

## P0 — CI, truthful claims and provider safety
- [x] API functional tests pass separately: **139 passed, five warnings**; MyPy and frontend tests/build passed. The full CI is still red due to strict Ruff. [CI 43428370982](https://github.com/HuLaxx/Sentinance/actions/runs/43428370982).
- [x] Ruff formatter repaired 12 Python files and applied 35 safe fixes; **67/67 files format-clean** on later review run. [PR #2](https://github.com/HuLaxx/Sentinance/pull/2), [PR Checks 43428370907](https://github.com/HuLaxx/Sentinance/actions/runs/43428370907).
- [x] Security and commit-message check jobs pass on same review branch; a passing security scan is not an audit of all secrets. [PR Checks 43428370907](https://github.com/HuLaxx/Sentinance/actions/runs/43428370907).
- [ ] Fix **165 remaining strict Ruff findings** without globally suppressing rules: about 87 BLE001 broad catches, 47 DTZ003 naive UTC calls plus B008/RUF059/etc. [Current P0 blocker](https://github.com/HuLaxx/Sentinance/actions/runs/43428370907).
- [ ] Add timezone/API/database regression tests before changing `utcnow()` and persistence semantics.
- [ ] Distinguish real provider outage, invalid input, rate limiting and safe fallback in API and UI; test each.
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

## Release decision
- **Current gate:** API/web functional tests pass and Python formatting is fixed, but **strict Ruff and overall CI are failing**; full source → curated → forecast/agent → browser path remains unverified.
- **Next smallest PR:** fix first semantic batch of UTC time handling and provider exception policies with regression tests, then rerun strict quality.
- **Never claim:** live market coverage, profitable predictions, secure autonomous financial advice or AWS deployment merely because code/template exists.
