'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Activity,
  ArrowUpRight,
  Database,
  Server,
  Cpu,
  Zap,
  Globe,
  Code,
  Layers,
  ExternalLink,
  Check,
  TrendingUp
} from 'lucide-react';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

// Tech stack with official links
const techStack = {
  frontend: [
    { name: 'Next.js 16', url: 'https://nextjs.org', color: 'from-zinc-700 to-zinc-900', icon: '▲' },
    { name: 'React 18', url: 'https://react.dev', color: 'from-cyan-500 to-blue-500', icon: '⚛️' },
    { name: 'TypeScript', url: 'https://typescriptlang.org', color: 'from-blue-600 to-blue-800', icon: 'TS' },
    { name: 'TailwindCSS', url: 'https://tailwindcss.com', color: 'from-cyan-400 to-blue-500', icon: '💨' },
    { name: 'Vercel AI SDK', url: 'https://sdk.vercel.ai', color: 'from-zinc-600 to-zinc-800', icon: '🤖' },
    { name: 'Zod', url: 'https://zod.dev', color: 'from-blue-500 to-indigo-600', icon: '✅' },
  ],
  backend: [
    { name: 'FastAPI', url: 'https://fastapi.tiangolo.com', color: 'from-emerald-500 to-teal-600', icon: '⚡' },
    { name: 'Python 3.11', url: 'https://python.org', color: 'from-blue-500 to-yellow-500', icon: '🐍' },
    { name: 'Pydantic', url: 'https://docs.pydantic.dev', color: 'from-pink-500 to-rose-500', icon: '📋' },
    { name: 'SQLAlchemy', url: 'https://sqlalchemy.org', color: 'from-red-600 to-orange-500', icon: '🔗' },
    { name: 'structlog', url: 'https://www.structlog.org', color: 'from-zinc-500 to-zinc-700', icon: '📝' },
    { name: 'httpx', url: 'https://www.python-httpx.org', color: 'from-purple-500 to-indigo-500', icon: '🌐' },
  ],
  data: [
    { name: 'PostgreSQL', url: 'https://postgresql.org', color: 'from-blue-600 to-indigo-700', icon: '🐘' },
    { name: 'Redis', url: 'https://redis.io', color: 'from-red-500 to-red-700', icon: '🔴' },
    { name: 'Apache Kafka', url: 'https://kafka.apache.org', color: 'from-zinc-600 to-zinc-800', icon: '📨' },
    { name: 'Multi-Exchange', url: 'https://binance.com', color: 'from-yellow-500 to-orange-500', icon: '📊' },
    { name: 'yfinance', url: 'https://github.com/ranaroussi/yfinance', color: 'from-purple-500 to-blue-500', icon: '📈' },
    { name: 'BeautifulSoup', url: 'https://beautiful-soup-4.readthedocs.io', color: 'from-green-500 to-emerald-600', icon: '🍲' },
  ],
  ai_ml: [
    { name: 'LangGraph', url: 'https://langchain-ai.github.io/langgraph/', color: 'from-purple-500 to-indigo-600', icon: '🔗' },
    { name: 'Gemini AI', url: 'https://ai.google.dev', color: 'from-blue-500 to-cyan-400', icon: '✨' },
    { name: 'Groq', url: 'https://groq.com', color: 'from-orange-500 to-amber-500', icon: '🚀' },
    { name: 'PyTorch', url: 'https://pytorch.org', color: 'from-orange-500 to-red-500', icon: '🔥' },
    { name: 'Qdrant', url: 'https://qdrant.tech', color: 'from-purple-600 to-pink-500', icon: '🎯' },
    { name: 'SSE Streaming', url: 'https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events', color: 'from-green-500 to-teal-500', icon: '📡' },
  ],
  devops: [
    { name: 'Docker', url: 'https://docker.com', color: 'from-blue-500 to-blue-700', icon: '🐳' },
    { name: 'Kubernetes', url: 'https://kubernetes.io', color: 'from-blue-600 to-indigo-500', icon: '☸️' },
    { name: 'Prometheus', url: 'https://prometheus.io', color: 'from-orange-500 to-red-600', icon: '📊' },
    { name: 'Grafana', url: 'https://grafana.com', color: 'from-orange-400 to-yellow-500', icon: '📈' },
    { name: 'GitHub Actions', url: 'https://github.com/features/actions', color: 'from-zinc-600 to-zinc-800', icon: '⚙️' },
    { name: 'MLflow', url: 'https://mlflow.org', color: 'from-blue-500 to-cyan-500', icon: '🧪' },
  ],
};


const metrics = [
  { label: 'Tests Passing', value: '126+', icon: Check, color: 'text-emerald-400' },
  { label: 'Response Time', value: '<100ms', icon: Zap, color: 'text-yellow-400' },
  { label: 'Uptime', value: '99.9%', icon: TrendingUp, color: 'text-sky-400' },
  { label: 'Coverage', value: '96%', icon: Code, color: 'text-purple-400' },
];

function TechCard({ tech, delay }: { tech: typeof techStack.frontend[0]; delay: number }) {
  const [isVisible, setIsVisible] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay);
    return () => clearTimeout(timer);
  }, [delay]);

  return (
    <a
      href={tech.url}
      target="_blank"
      rel="noopener noreferrer"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`group relative flex items-center gap-3 p-4 rounded-xl border border-zinc-800 
        bg-zinc-900/50 backdrop-blur-sm cursor-pointer
        transition-all duration-300
        ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}
        ${isHovered ? 'border-zinc-600 shadow-xl shadow-blue-500/10 -translate-y-1 scale-[1.02]' : ''}
      `}
    >
      {/* Gradient glow on hover */}
      <div className={`absolute inset-0 rounded-xl bg-gradient-to-r ${tech.color} 
        opacity-0 ${isHovered ? 'opacity-10' : ''} transition-opacity duration-300`}
      />

      <div className={`relative w-10 h-10 rounded-lg bg-gradient-to-br ${tech.color} 
        flex items-center justify-center text-white font-bold text-sm
        transition-transform duration-300 ${isHovered ? 'scale-110 rotate-3' : ''}`}
      >
        {tech.icon}
      </div>

      <div className="relative flex-1">
        <span className={`font-medium transition-colors duration-300 
          ${isHovered ? 'text-sky-400' : 'text-zinc-100'}`}
        >
          {tech.name}
        </span>
      </div>

      <ExternalLink className={`w-4 h-4 text-zinc-600 transition-all duration-300
        ${isHovered ? 'text-sky-400 translate-x-1' : ''}`}
      />
    </a>
  );
}

function MetricCard({ metric, index }: { metric: typeof metrics[0]; index: number }) {
  const [isVisible, setIsVisible] = useState(false);
  const [count, setCount] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), index * 100);
    return () => clearTimeout(timer);
  }, [index]);

  return (
    <div className={`group p-6 rounded-xl border border-zinc-800 bg-zinc-900/30 
      hover:border-zinc-700 hover:bg-zinc-900/50 transition-all duration-300 hover:-translate-y-1
      ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}
    >
      <div className={`w-12 h-12 rounded-xl bg-zinc-800/50 border border-zinc-700/50 
        flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}
      >
        <metric.icon className={`w-6 h-6 ${metric.color}`} />
      </div>
      <p className={`text-3xl font-bold ${metric.color} mb-1`}>{metric.value}</p>
      <p className="text-sm text-zinc-500">{metric.label}</p>
    </div>
  );
}

function CategorySection({ title, icon: Icon, techs, baseDelay }: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  techs: typeof techStack.frontend;
  baseDelay: number;
}) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-zinc-400">
        <Icon className="w-5 h-5" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {techs.map((tech, i) => (
          <TechCard key={tech.name} tech={tech} delay={baseDelay + i * 50} />
        ))}
      </div>
    </div>
  );
}

export default function TechDashPage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen relative z-10">
      <SiteHeader />

      {/* Hero */}
      <section className="relative pt-28 pb-16 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-sky-900/20 via-transparent to-transparent" />

        <div className={`relative max-w-5xl mx-auto px-6 text-center
          transition-all duration-1000 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/80 border border-zinc-800 text-sm text-zinc-400 mb-6">
            <Cpu className="w-4 h-4 text-purple-400" />
            Technology Stack
          </div>

          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="bg-gradient-to-r from-zinc-100 to-zinc-300 bg-clip-text text-transparent">
              Tech Dashboard
            </span>
          </h1>

          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            A complete overview of the technologies powering Sentinance.
            Click any technology to visit its official documentation.
          </p>
        </div>
      </section>

      {/* Metrics */}
      <section className="py-12 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metrics.map((metric, i) => (
              <MetricCard key={metric.label} metric={metric} index={i} />
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto space-y-12">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-zinc-100 mb-4">Complete Tech Stack</h2>
            <p className="text-zinc-400">Click any technology to learn more on their official site</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <CategorySection
              title="Frontend"
              icon={Globe}
              techs={techStack.frontend}
              baseDelay={100}
            />
            <CategorySection
              title="Backend"
              icon={Server}
              techs={techStack.backend}
              baseDelay={200}
            />
            <CategorySection
              title="Data Layer"
              icon={Database}
              techs={techStack.data}
              baseDelay={300}
            />
            <CategorySection
              title="AI & Machine Learning"
              icon={Cpu}
              techs={techStack.ai_ml}
              baseDelay={400}
            />
          </div>

          <div className="pt-8">
            <CategorySection
              title="DevOps & Observability"
              icon={Layers}
              techs={techStack.devops}
              baseDelay={500}
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto flex flex-wrap justify-center gap-4">
          <Link
            href="/architecture"
            className="group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-500 
              text-white rounded-xl font-semibold transition-all shadow-xl shadow-purple-500/20 
              hover:shadow-purple-500/40 hover:scale-[1.02]"
          >
            View Architecture
            <ArrowUpRight className="w-5 h-5 transition-transform group-hover:translate-x-1 group-hover:-translate-y-1" />
          </Link>
          <Link
            href="/demo"
            className="flex items-center gap-2 px-8 py-4 bg-zinc-900/80 border border-zinc-700 
              text-zinc-100 rounded-xl font-semibold hover:bg-zinc-800 hover:border-zinc-600 transition-all"
          >
            View Demo
          </Link>
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}
