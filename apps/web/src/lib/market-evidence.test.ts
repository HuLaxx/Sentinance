import { describe, expect, it, vi } from "vitest";
import { loadMarketEvidence } from "@/lib/market-evidence";

const storedFixture = {
  schema_version: 1,
  status: "stored_events",
  as_of: "2026-11-17T14:30:00+00:00",
  live_feed_verified: false,
  warning: "Stored events only; recent timestamps are not a live guarantee.",
  events: [
    {
      source: "binance_aggtrade",
      source_mode: "fixture",
      symbol: "BTCUSDT",
      source_event_id: "500",
      schema_version: 1,
      observed_at: "2026-11-17T14:00:00+00:00",
      received_at: "2026-11-17T14:30:00+00:00",
      price: "100.123456789012",
      volume: "0.25",
      freshness: "stale",
      provenance_status: "verified_schema_not_provider_uptime",
    },
  ],
};

function fetchFixture(payload: unknown, status = 200) {
  return vi.fn(async () => new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  })) as unknown as typeof fetch;
}

describe("durable market evidence boundary", () => {
  it("never invents prices if backend URL is absent", async () => {
    const neverFetch = vi.fn() as unknown as typeof fetch;
    const result = await loadMarketEvidence(undefined, neverFetch);
    expect(result.state).toBe("unavailable");
    expect(neverFetch).not.toHaveBeenCalled();
  });

  it("rejects unencrypted remote URLs and embedded credentials", async () => {
    const mock = fetchFixture(storedFixture);
    expect((await loadMarketEvidence("http://remote.example", mock)).state)
      .toBe("unavailable");
    expect((await loadMarketEvidence("https://user:pass@remote.example", mock)).state)
      .toBe("unavailable");
    expect(mock).not.toHaveBeenCalled();
  });

  it("preserves fixture provenance, precision and stale flags", async () => {
    const mock = fetchFixture(storedFixture);
    const result = await loadMarketEvidence("https://api.example.org/base", mock);
    expect(result.state).toBe("ready");
    if (result.state !== "ready") throw new Error("Expected real parsed fixture");
    expect(result.data.events[0].source_mode).toBe("fixture");
    expect(result.data.events[0].price).toBe("100.123456789012");
    expect(result.data.events[0].freshness).toBe("stale");
    expect(result.data.live_feed_verified).toBe(false);
    const requestUrl = new URL((mock as ReturnType<typeof vi.fn>).mock.calls[0][0]);
    expect(requestUrl.pathname).toBe("/api/v1/market-events");
    expect(requestUrl.searchParams.get("symbol")).toBe("BTCUSDT");
  });

  it("never turns outage or schema drift into a mock live quote", async () => {
    const unavailable = await loadMarketEvidence(
      "https://api.example.org",
      fetchFixture({ detail: "offline" }, 503),
    );
    expect(unavailable.state).toBe("unavailable");
    const malformed = await loadMarketEvidence(
      "https://api.example.org",
      fetchFixture({ ...storedFixture, events: [{ fake: 1 }] }),
    );
    expect(malformed.state).toBe("unavailable");
  });

  it("shows an empty ledger as empty, not synthetic data", async () => {
    const result = await loadMarketEvidence(
      "http://localhost:8000",
      fetchFixture({ ...storedFixture, status: "no_data", events: [] }),
    );
    expect(result.state).toBe("ready");
    if (result.state !== "ready") throw new Error("Expected valid empty response");
    expect(result.data.status).toBe("no_data");
    expect(result.data.events).toHaveLength(0);
  });
});
