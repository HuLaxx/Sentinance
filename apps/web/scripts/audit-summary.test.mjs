import assert from "node:assert/strict";
import { test } from "node:test";
import { inspectAudit, formatSummary } from "./audit-summary.mjs";

function fixture(vulnerabilities = {}, counts = {}) {
  return {
    vulnerabilities,
    metadata: {
      vulnerabilities: {
        critical: 0,
        high: 0,
        moderate: 0,
        low: 0,
        ...counts,
      },
    },
  };
}

test("a clean dependency tree passes without invented findings", () => {
  const result = inspectAudit(fixture(), "prod");
  assert.equal(result.blocked, false);
  assert.match(formatSummary(result), /release gate: passed/);
});

test("a critical transitive issue blocks a release and retains fix information", () => {
  const result = inspectAudit(
    fixture(
      {
        "test-package": {
          severity: "critical",
          isDirect: false,
          range: "<3.0.0",
          fixAvailable: { name: "test-package", version: "3.0.0", isSemVerMajor: true },
        },
      },
      { critical: 1 },
    ),
    "all",
  );
  assert.equal(result.blocked, true);
  assert.match(formatSummary(result), /CRITICAL test-package \(transitive/);
  assert.match(formatSummary(result), /breaking upgrade/);
});

test("moderate-only issues are reported without crossing high-severity threshold", () => {
  const result = inspectAudit(
    fixture({ minor: { severity: "moderate", isDirect: true } }, { moderate: 1 }),
    "prod",
  );
  assert.equal(result.blocked, false);
  assert.equal(result.counts.moderate, 1);
});

test("registry audit failures cannot be misreported as a clean scan", () => {
  assert.throws(
    () => inspectAudit({ error: { code: "EAI_AGAIN" } }, "prod"),
    /Invalid or failed npm audit/,
  );
});

test("missing metadata and malformed totals fail closed", () => {
  assert.throws(() => inspectAudit({ vulnerabilities: {} }, "prod"), /Invalid/);
  assert.throws(
    () => inspectAudit(fixture({}, { high: -5 }), "all"),
    /Invalid vulnerability totals/,
  );
});
