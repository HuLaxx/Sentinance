import { Activity, ArrowUpRight, Database, Server } from "lucide-react";
import Link from "next/link";
import MetricCard from "../../components/site/MetricCard";
import SectionHeader from "../../components/site/SectionHeader";
import StackedCard from "../../components/site/StackedCard";
import Tag from "../../components/site/Tag";

const stackTags = [
  "Next.js 15",
  "FastAPI",
  "Kafka",
  "dbt",
  "Postgres",
  "Redis",
  "Qdrant",  // Now connected with Gemini Embeddings
  "PyTorch",
  "Kubernetes",
  "Docker",
  "TailwindCSS"
];

const statusPoints = [
  "Sub-second alerts from streaming pipeline",
  "Model registry with A/B testing",
  "Rate limiting and policy gates active",
  "Observability across logs, metrics, traces",
];

export default function Page() {
  return (
    <div className="pt-36 pb-16 px-6 max-w-5xl mx-auto space-y-12">
      <SectionHeader
        eyebrow="Tech Dash"
        title="Operational metrics and stack snapshot"
        lead="A concise view of throughput, system health, and the core technology stack."
        size="page"
      />

      <section>
        <SectionHeader
          eyebrow="Key Metrics"
          title="Performance and coverage at a glance"
          lead="These values describe the intended operating targets for Sentinance."
        />
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard label="Events per second" value="50K+" sub="Streaming ingestion" />
          <MetricCard label="Latency p99" value="< 120ms" sub="Alert delivery" accent />
          <MetricCard label="Time to insight" value="15 min" sub="From 4 hours" />
          <MetricCard label="Cache hit rate" value="90%" sub="Semantic cache" accent />
        </div>
      </section>

      <StackedCard>
        <div className="flex items-center gap-3">
          <Database size={18} className="text-indigo-300" />
          <h2 className="font-display text-xl font-semibold">Stack snapshot</h2>
        </div>
        <div className="mt-6 flex flex-wrap gap-2">
          {stackTags.map((tag) => (
            <Tag key={tag}>{tag}</Tag>
          ))}
        </div>
      </StackedCard>

      <StackedCard>
        <div className="flex items-center gap-3">
          <Server size={18} className="text-indigo-300" />
          <h2 className="font-display text-xl font-semibold">Operational posture</h2>
        </div>
        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-400/90">
          {statusPoints.map((item) => (
            <div key={item} className="flex items-start gap-2">
              <Activity size={14} className="text-emerald-300 mt-0.5" />
              <span>{item}</span>
            </div>
          ))}
        </div>
      </StackedCard>

      <div className="flex flex-wrap gap-4">
        <Link
          href="/architecture"
          className="focus-ring inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold bg-white text-black hover:bg-gray-200 transition"
        >
          Architecture
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
