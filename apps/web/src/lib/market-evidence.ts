import { z } from "zod";

const eventSchema = z.object({
  source: z.string(),
  source_mode: z.enum(["fixture", "provider", "unverified"]),
  symbol: z.string(),
  source_event_id: z.string(),
  schema_version: z.number().int(),
  observed_at: z.string(),
  received_at: z.string(),
  price: z.string(),
  volume: z.string(),
  freshness: z.enum(["recent", "stale", "future_timestamp"]),
  provenance_status: z.string(),
});

const marketEvidenceSchema = z.object({
  schema_version: z.literal(1),
  status: z.enum(["stored_events", "no_data"]),
  as_of: z.string(),
  events: z.array(eventSchema).max(50),
  live_feed_verified: z.boolean(),
  warning: z.string(),
});

export type MarketEvidence = z.infer<typeof marketEvidenceSchema>;

export type MarketEvidenceResult =
  | { state: "ready"; data: MarketEvidence }
  | { state: "unavailable"; reason: string };

export async function loadMarketEvidence(
  configuredApiUrl: string | undefined,
  fetchImpl: typeof fetch = fetch,
): Promise<MarketEvidenceResult> {
  if (!configuredApiUrl) {
    return {
      state: "unavailable",
      reason: "Backend not configured. No live or fixture database is connected.",
    };
  }
  let origin: URL;
  try {
    origin = new URL(configuredApiUrl);
    const local = ["localhost", "127.0.0.1"].includes(origin.hostname);
    if (
      origin.username ||
      origin.password ||
      (origin.protocol !== "https:" && !(local && origin.protocol === "http:"))
    ) {
      return {
        state: "unavailable",
        reason: "Backend URL must use HTTPS, except for a local development host.",
      };
    }
  } catch {
    return { state: "unavailable", reason: "Backend URL is not valid." };
  }

  const endpoint = new URL("/api/v1/market-events", origin.origin);
  endpoint.searchParams.set("symbol", "BTCUSDT");
  endpoint.searchParams.set("limit", "20");

  try {
    const response = await fetchImpl(endpoint.toString(), {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) {
      return {
        state: "unavailable",
        reason: "Durable market-event API unavailable. No prices are being invented.",
      };
    }
    const parsed = marketEvidenceSchema.safeParse(await response.json());
    if (!parsed.success) {
      return {
        state: "unavailable",
        reason: "Market-event API returned an invalid schema.",
      };
    }
    return { state: "ready", data: parsed.data };
  } catch {
    return {
      state: "unavailable",
      reason: "Could not reach the durable market-event backend.",
    };
  }
}
