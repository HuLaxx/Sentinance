import Link from "next/link";
import { loadForecastEvidence } from "@/lib/forecast-evidence";
import { loadMarketEvidence } from "@/lib/market-evidence";

export const dynamic = "force-dynamic";

export default async function MarketEvidencePage() {
  const [result, forecast] = await Promise.all([
    loadMarketEvidence(process.env.SENTINANCE_API_URL),
    loadForecastEvidence(process.env.SENTINANCE_API_URL),
  ]);

  return (
    <>
    <header className="atlas-chapter">
      <p className="atlas-kicker">
        <Link href="/data" className="underline underline-offset-4">Data</Link>&nbsp;/ Provenance
      </p>
      <h1>Stored market events</h1>
      <p>
        A read-only view of BTCUSDT records in the PostgreSQL price-event ledger.
        These are historical stored observations, not streaming quotes, trading
        signals or forecasts. Each row states whether its origin was a fixture,
        a provider request or unverified.
      </p>
    </header>
    <main className="relative z-10 mx-auto max-w-6xl space-y-8 px-5 pb-20 pt-12 text-slate-100">

      {result.state === "unavailable" ? (
        <section
          className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-7"
          role="status"
        >
          <h2 className="text-lg font-semibold text-amber-200">
            Durable feed not connected
          </h2>
          <p className="mt-3 text-sm leading-7 text-slate-300">
            {result.reason}
          </p>
          <p className="mt-2 text-xs text-slate-400">
            No last-known or fabricated prices are substituted for unavailable data.
          </p>
        </section>
      ) : (
        <>
          <section className="rounded-2xl border border-white/15 bg-slate-900/60 p-6">
            <p className="text-sm text-slate-400">
              Ledger status
            </p>
            <p className="mt-2 font-semibold">
              {result.data.status === "no_data"
                ? "No stored BTCUSDT events"
                : `${result.data.events.length} stored BTCUSDT event(s)`}
            </p>
            <p className="mt-2 text-xs leading-6 text-slate-300">
              {result.data.warning}
            </p>
            <p className="mt-2 text-xs text-slate-400">
              Backend snapshot: {result.data.as_of} · Live feed independently
              verified: no
            </p>
          </section>

          {result.data.events.length > 0 ? (
            <div className="overflow-x-auto rounded-2xl border border-white/15 bg-slate-900/50">
              <table className="w-full min-w-[850px] text-left text-xs sm:text-sm">
                <caption className="px-5 py-4 text-left text-sm text-slate-300">
                  Latest persisted observations; prices retained as decimal strings
                </caption>
                <thead className="border-b border-white/10 bg-slate-900/80 text-slate-300">
                  <tr>
                    <th scope="col" className="px-5 py-4">Origin</th>
                    <th scope="col" className="px-5 py-4">Provider / symbol</th>
                    <th scope="col" className="px-5 py-4">Trade ID</th>
                    <th scope="col" className="px-5 py-4">Price</th>
                    <th scope="col" className="px-5 py-4">Provider time (UTC)</th>
                    <th scope="col" className="px-5 py-4">Freshness</th>
                  </tr>
                </thead>
                <tbody>
                  {result.data.events.map((event) => (
                    <tr
                      key={`${event.source}:${event.symbol}:${event.source_event_id}`}
                      className="border-b border-white/5"
                    >
                      <td className="px-5 py-4 font-medium text-cyan-200">
                        {event.source_mode === "fixture"
                          ? "Synthetic fixture"
                          : event.source_mode === "provider"
                            ? "Provider-sourced (not verified live)"
                            : "Origin unverified"}
                      </td>
                      <td className="px-5 py-4">{event.source} / {event.symbol}</td>
                      <td className="px-5 py-4 font-mono">{event.source_event_id}</td>
                      <td className="px-5 py-4 font-mono">{event.price}</td>
                      <td className="px-5 py-4">{event.observed_at}</td>
                      <td className="px-5 py-4 capitalize">
                        {event.freshness.replaceAll("_", " ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="rounded-2xl border border-white/10 p-6 text-slate-300">
              The configured backend has no stored BTCUSDT events. Start the
              explicitly enabled ingestion and consumer workflow to populate
              a real ledger; this page does not manufacture examples.
            </p>
          )}
        </>
      )}

      <section className="space-y-4 rounded-2xl border border-white/15 bg-slate-900/60 p-6">
        <h2 className="text-xl font-semibold">Historical forecast evaluation</h2>
        <p className="text-sm text-slate-300">
          Deterministic LangGraph evidence, not live forecasts or investment advice.
        </p>
        {forecast ? (
          <>
            <p className="text-sm text-slate-300">{forecast.statistical_results.symbol} · Live market verified: no</p>
            <table className="w-full text-left text-sm">
              <thead><tr><th scope="col">Method</th><th scope="col">MAE</th><th scope="col">RMSE</th></tr></thead>
              <tbody>{Object.entries(forecast.statistical_results.metrics).map(([method, metric]) => (
                <tr key={method}><td>{method}</td><td>{metric.mae.toPrecision(5)}</td><td>{metric.rmse.toPrecision(5)}</td></tr>
              ))}</tbody>
            </table>
            <p className="break-all font-mono text-xs text-slate-400">{forecast.citation}</p>
            <p className="text-xs text-slate-400">Trace: {forecast.trace.map(step => step.step).join(" → ")} · Zero LLM tokens and external calls</p>
          </>
        ) : (
          <p role="status" className="text-sm text-amber-200">No verified forecast report configured.</p>
        )}
      </section>

      <p className="max-w-3xl text-xs leading-6 text-slate-400">
        Provider availability, exchange licensing, ingestion gap repair,
        object-storage data layers and model/agent interpretation are separate
        release gates. A recent timestamp alone cannot verify a live exchange feed.
      </p>
    </main>
    </>
  );
}
