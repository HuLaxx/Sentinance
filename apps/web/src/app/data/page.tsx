import { ArrowUpRight, Database, Globe, Layers } from "lucide-react";
import Link from "next/link";
import SectionHeader from "../../components/site/SectionHeader";
import StackedCard from "../../components/site/StackedCard";
import Tag from "../../components/site/Tag";

const sources = [
  "Binance aggregate-trade adapter (fixture-tested; live opt-in)",
  "Legacy market ticker / index fetchers (separate path)",
  "News RSS data sources (availability varies)",
  "Synthetic event IDs and source timestamps are never substituted for real trades",
];

const storage = [
  "PostgreSQL raw Kafka event ledger (real broker/database CI tested)",
  "Redis Pub/Sub for temporary fanout (no event replay)",
  "Qdrant integration is optional; not part of the proven data path",
  "In-memory price history remains separate from the durable ledger",
];

const quality = [
  "Versioned source event schema, event-time and numeric validation",
  "Idempotent Kafka-to-PostgreSQL storage with upstream event IDs",
  "DB-backed API and fixture/provider/stale evidence labeling",
  "Parquet and dbt materialization are planned, not yet certified",
];

export default function Page() {
  return (
    <div className="pt-36 pb-16 px-6 max-w-5xl mx-auto space-y-12">
      <SectionHeader
        eyebrow="Data"
        title="Unified data foundation for market intelligence"
        lead="See what is actually tested, what is fixture-driven, and what remains an architecture goal."
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
          lead="These systems have different verification levels; only the raw Kafka ledger is covered by the new durable event tests."
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
          A versioned source-event pipeline now has real Kafka and PostgreSQL
          replay tests. Full Parquet raw/validated/curated partitions, provider
          gap backfills and ML-to-browser lineage remain separate, unverified
          engineering milestones.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag tone="good">Kafka retry tested</Tag>
          <Tag>Source IDs preserved</Tag>
          <Tag>Parquet pending</Tag>
        </div>
      </StackedCard>

      <div className="flex flex-wrap gap-4">
        <Link
          href="/data/evidence"
          className="focus-ring inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold bg-cyan-300 text-slate-950 hover:bg-cyan-200 transition"
        >
          Stored event evidence
          <ArrowUpRight size={16} />
        </Link>
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
