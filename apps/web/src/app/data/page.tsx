import { ArrowUpRight, Database, Globe, Layers } from "lucide-react";
import Link from "next/link";
import SectionHeader from "../../components/site/SectionHeader";
import StackedCard from "../../components/site/StackedCard";
import Tag from "../../components/site/Tag";

const sources = [
  "Binance WebSocket price feeds",
  "News RSS aggregators and scrapers",
  "yFinance historical data downloads",
  "Price history collectors",
];

const storage = [
  "PostgreSQL for persistent data models",
  "Redis for real-time caching and Pub/Sub",
  "Qdrant vector database for semantic search",
  "In-memory price history for charts",
];

const quality = [
  "Pydantic schema validation for API models",
  "Zod validation for frontend WebSocket messages",
  "dbt models for data transformation",
  "Alembic for database migrations",
];

export default function Page() {
  return (
    <div className="pt-36 pb-16 px-6 max-w-5xl mx-auto space-y-12">
      <SectionHeader
        eyebrow="Data"
        title="Unified data foundation for market intelligence"
        lead="Every signal is normalized, validated, and stored with lineage for replay and audit."
        size="page"
      />

      <StackedCard>
        <div className="flex items-center gap-3">
          <Globe size={18} className="text-indigo-300" />
          <h2 className="font-display text-xl font-semibold">Ingestion sources</h2>
        </div>
        <ul className="mt-4 space-y-2 text-sm text-slate-400/90">
          {sources.map((item) => (
            <li key={item} className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-indigo-400" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </StackedCard>

      <section>
        <SectionHeader
          eyebrow="Storage"
          title="Purpose-built stores for each workload"
          lead="Cold storage, fast reads, vector search, and graph analytics work together."
        />
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass rounded-[var(--radius)] p-7">
            <div className="flex items-center gap-3">
              <Database size={18} className="text-indigo-300" />
              <h3 className="font-display text-lg font-semibold">Core storage</h3>
            </div>
            <ul className="mt-4 space-y-2 text-sm text-slate-400/90">
              {storage.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-slate-500" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="glass rounded-[var(--radius)] p-7">
            <div className="flex items-center gap-3">
              <Layers size={18} className="text-indigo-300" />
              <h3 className="font-display text-lg font-semibold">Quality controls</h3>
            </div>
            <ul className="mt-4 space-y-2 text-sm text-slate-400/90">
              {quality.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-slate-500" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <StackedCard>
        <div className="flex items-center gap-3">
          <Database size={18} className="text-indigo-300" />
          <h2 className="font-display text-xl font-semibold">Coverage summary</h2>
        </div>
        <p className="mt-4 text-slate-400/90 leading-relaxed">
          Data is partitioned by symbol and time, with replayable streams for investigations and model
          training. Extend coverage by plugging in additional exchange or regional sources as needed.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag tone="good">Replayable</Tag>
          <Tag>Schema validated</Tag>
          <Tag>Feature-ready</Tag>
        </div>
      </StackedCard>

      <div className="flex flex-wrap gap-4">
        <Link
          href="/techdash"
          className="focus-ring inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold bg-white text-black hover:bg-gray-200 transition"
        >
          Tech Dash
          <ArrowUpRight size={16} />
        </Link>
        <Link
          href="/demo"
          className="focus-ring inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold border border-white/15 text-white hover:border-white/30 hover:bg-white/5 transition"
        >
          Demo terminal
        </Link>
      </div>
    </div>
  );
}
