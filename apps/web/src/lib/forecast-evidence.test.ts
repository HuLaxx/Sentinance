import { describe, expect, it, vi } from "vitest";
import { loadForecastEvidence } from "./forecast-evidence";

describe("forecast evidence loader", () => {
  it("does not fetch when the backend is missing or unsafe", async () => {
    const request = vi.fn<typeof fetch>();
    expect(await loadForecastEvidence(undefined, request)).toBeNull();
    expect(await loadForecastEvidence("http://example.com", request)).toBeNull();
    expect(request).not.toHaveBeenCalled();
  });

  it("rejects invented or live-claimed responses", async () => {
    const request = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ status: "historical_evaluation", live_market_verified: true })),
    );
    expect(await loadForecastEvidence("https://example.com", request)).toBeNull();
  });

  it("accepts a bounded, cited historical report", async () => {
    const metric = { mae: 1, rmse: 2 };
    const fixture = {
      status: "historical_evaluation",
      statistical_results: {
        symbol: "BTCUSDT", manifest_sha256: "a".repeat(64),
        metrics: { last_close: metric, moving_average: metric, ridge_ar1: metric },
      },
      citation: "sha256:" + "b".repeat(64),
      interpretation_mode: "deterministic_no_llm",
      live_market_verified: false,
      llm_tokens: 0, external_calls: 0, latency_ms: 1,
      trace: ["retrieve", "analyze", "finalize"].map(step => ({ step, status: "complete" })),
    };
    const request = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify(fixture)),
    );
    expect(await loadForecastEvidence("https://example.com", request)).toEqual(fixture);
  });
});
