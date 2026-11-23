import { z } from "zod";

const metrics = z.object({
  mae: z.number().finite().nonnegative(),
  rmse: z.number().finite().nonnegative(),
});

const schema = z.object({
  status: z.literal("historical_evaluation"),
  statistical_results: z.object({
    symbol: z.string().regex(/^[A-Z0-9]{2,20}$/),
    manifest_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    metrics: z.object({
      last_close: metrics,
      moving_average: metrics,
      ridge_ar1: metrics,
    }),
  }),
  citation: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  interpretation_mode: z.literal("deterministic_no_llm"),
  live_market_verified: z.literal(false),
  llm_tokens: z.literal(0),
  external_calls: z.literal(0),
  latency_ms: z.number().finite().nonnegative(),
  trace: z.array(z.object({
    step: z.enum(["retrieve", "analyze", "finalize"]),
    status: z.string().max(80),
  })).length(3),
});

export async function loadForecastEvidence(
  configuredApiUrl: string | undefined,
  fetchImpl: typeof fetch = fetch,
): Promise<z.infer<typeof schema> | null> {
  if (!configuredApiUrl) return null;
  try {
    const url = new URL(configuredApiUrl);
    const local = ["localhost", "127.0.0.1"].includes(url.hostname);
    if (url.username || url.password || (url.protocol !== "https:" && !(local && url.protocol === "http:"))) return null;
    const response = await fetchImpl(new URL("/api/v1/forecast-evidence", url.origin), {
      cache: "no-store",
      redirect: "error",
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) return null;
    const parsed = schema.safeParse(await response.json());
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}
