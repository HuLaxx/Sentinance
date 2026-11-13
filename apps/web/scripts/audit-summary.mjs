/**
 * Inspect npm's machine-readable advisories without hiding a failed registry request.
 * Exports pure helpers for deterministic offline regression tests.
 */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const LEVELS = ["critical", "high", "moderate", "low"];

export function inspectAudit(report, label) {
  if (
    !report ||
    typeof report !== "object" ||
    report.error ||
    !report.vulnerabilities ||
    !report.metadata?.vulnerabilities
  ) {
    throw new Error(`Invalid or failed npm audit report: ${label}`);
  }

  const entries = Object.entries(report.vulnerabilities).map(([name, advisory]) => ({
    name,
    severity: advisory.severity,
    isDirect: advisory.isDirect === true,
    fixAvailable: advisory.fixAvailable,
    range: advisory.range ?? "unknown",
  }));
  const counts = Object.fromEntries(
    LEVELS.map((level) => [
      level,
      Number(report.metadata.vulnerabilities[level] ?? 0),
    ]),
  );
  if (Object.values(counts).some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error(`Invalid vulnerability totals: ${label}`);
  }

  return {
    label,
    counts,
    entries: entries.sort((a, b) => {
      const bySeverity = LEVELS.indexOf(a.severity) - LEVELS.indexOf(b.severity);
      return bySeverity || a.name.localeCompare(b.name);
    }),
    blocked: counts.critical + counts.high > 0,
  };
}

export function formatSummary(result) {
  const { label, counts, entries, blocked } = result;
  const output = [
    `${label}: critical=${counts.critical} high=${counts.high} moderate=${counts.moderate} low=${counts.low}`,
    `High/critical release gate: ${blocked ? "FAILED" : "passed"}`,
  ];
  for (const item of entries.filter(
    (entry) => entry.severity === "critical" || entry.severity === "high",
  )) {
    const fix =
      item.fixAvailable === false
        ? "no published fix"
        : item.fixAvailable === true
          ? "fix available"
          : "major/breaking upgrade may be required";
    output.push(
      `  ${item.severity.toUpperCase()} ${item.name} (${item.isDirect ? "direct" : "transitive"}; ${fix}; range ${item.range})`,
    );
  }
  return output.join("\n");
}

function run(paths) {
  if (paths.length !== 2) {
    throw new Error("Usage: node scripts/audit-summary.mjs <full.json> <production.json>");
  }
  const [allPath, productionPath] = paths;
  const all = inspectAudit(JSON.parse(readFileSync(allPath, "utf8")), "all dependencies");
  const production = inspectAudit(
    JSON.parse(readFileSync(productionPath, "utf8")),
    "production dependencies",
  );
  console.log(formatSummary(all));
  console.log(formatSummary(production));
  if (all.blocked || production.blocked) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  try {
    run(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
