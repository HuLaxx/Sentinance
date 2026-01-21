import { Activity, ArrowUpRight, Cpu, Database, Server, Zap } from "lucide-react";
import Link from "next/link";
import SectionHeader from "../../components/site/SectionHeader";
import StackedCard from "../../components/site/StackedCard";
import Tag from "../../components/site/Tag";

const coreServices = [
  {
    title: "Ingestion layer",
    detail: "Binance WebSocket connector, news RSS aggregators, and price history collectors.",
    icon: Database,
  },
  {
    title: "Event backbone",
    detail: "Kafka topics for replayable streams with Redis Pub/Sub for real-time broadcasting.",
    icon: Zap,
  },
  {
    title: "Stream processing",
    detail: "Real-time price updates and alert triggering via background async tasks.",
    icon: Activity,
  },
  {
    title: "ML & AI services",
    detail: "PyTorch LSTM for predictions, Gemini AI for analysis, and Qdrant for semantic search.",
    icon: Cpu,
  },
  {
    title: "Delivery layer",
    detail: "FastAPI REST/WebSocket gateway and Next.js 16 React frontend.",
    icon: Server,
  },
];

const hotPath = [
  "Exchange WebSocket to price aggregator",
  "Kafka ingestion and stream processing",
  "Redis cache for low-latency reads",
  "WebSocket broadcast to operators",
];

const coldPath = [
  "PostgreSQL storage for price history",
  "dbt models for data transformation",
  "PyTorch model training on historical data",
  "Scheduled predictions and analysis refresh",
];

export default function Page() {
  return (
    <div className="pt-36 pb-16 px-6 max-w-5xl mx-auto space-y-12">
      <SectionHeader
        eyebrow="Architecture"
        title="Event-driven system with CQRS separation"
        lead="Hot path alerts stay fast while cold path analytics power backtests and model retraining."
        size="page"
      />

      <StackedCard>
        <div className="flex items-center gap-3">
          <Server size={18} className="text-indigo-300" />
          <h2 className="font-display text-xl font-semibold">Architecture principles</h2>
        </div>
        <p className="mt-4 text-slate-400/90 leading-relaxed">
          Sentinance separates command and query workloads with event sourcing. Every signal is replayable,
          traceable, and optimized for predictable latency.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag tone="good">CQRS</Tag>
          <Tag>Event sourcing</Tag>
          <Tag>Lambda architecture</Tag>
        </div>
      </StackedCard>

      <section>
        <SectionHeader
          eyebrow="Core Services"
          title="Clear boundaries by layer"
          lead="Each service owns a single responsibility to improve reliability and scale."
        />
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {coreServices.map((service) => (
            <div key={service.title} className="glass rounded-[var(--radius)] p-7">
              <div className="flex items-center gap-3">
                <service.icon size={18} className="text-indigo-300" />
                <h3 className="font-display text-lg font-semibold">{service.title}</h3>
              </div>
              <p className="mt-3 text-sm text-slate-400/90 leading-relaxed">{service.detail}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <SectionHeader
          eyebrow="Data Flow"
          title="Hot path and cold path coordination"
          lead="Real-time alerts and historical analytics share the same event backbone."
        />
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <StackedCard>
            <h3 className="font-display text-lg font-semibold">Hot path</h3>
            <ul className="mt-4 space-y-2 text-sm text-slate-400/90">
              {hotPath.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </StackedCard>
          <StackedCard>
            <h3 className="font-display text-lg font-semibold">Cold path</h3>
            <ul className="mt-4 space-y-2 text-sm text-slate-400/90">
              {coldPath.map((item) => (
                <li key={item} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-slate-500" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </StackedCard>
        </div>
      </section>

      <div className="flex flex-wrap gap-4">
        <Link
          href="/data"
          className="focus-ring inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold bg-white text-black hover:bg-gray-200 transition"
        >
          Data overview
          <ArrowUpRight size={16} />
        </Link>
        <Link
          href="/demo"
          className="focus-ring inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold border border-white/15 text-white hover:border-white/30 hover:bg-white/5 transition"
        >
          View demo
        </Link>
      </div>
    </div>
  );
}
