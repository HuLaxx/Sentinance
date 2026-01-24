'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Activity,
  ArrowUpRight,
  Cpu,
  Database,
  Server,
  Zap,
  Globe,
  Brain,
  Layers,
  Radio,
  ArrowRight,
  Check,
  Container,
  Cloud,
  Lock,
  GitBranch
} from 'lucide-react';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

// 1. DATA PIPELINE STEPS
const pipelineSteps = [
  {
    id: '01',
    title: 'Ingestion',
    desc: 'Multi-exchange WebSocket feeds & RSS streams',
    icon: Radio,
    color: 'text-blue-400',
    bg: 'bg-blue-500/10'
  },
  {
    id: '02',
    title: 'Streaming',
    desc: 'Apache Kafka & Redis Pub/Sub event bus',
    icon: Zap,
    color: 'text-purple-400',
    bg: 'bg-purple-500/10'
  },
  {
    id: '03',
    title: 'Processing',
    desc: 'FastAPI workers & ML inference engine',
    icon: Cpu,
    color: 'text-orange-400',
    bg: 'bg-orange-500/10'
  },
  {
    id: '04',
    title: 'Storage',
    desc: 'TimescaleDB (Time-series) & Qdrant (Vectors)',
    icon: Database,
    color: 'text-green-400',
    bg: 'bg-green-500/10'
  },
];

// 2. INFRASTRUCTURE TOPOLOGY
const infraNodes = [
  { name: 'Kubernetes Cluster', icon: Cloud, items: ['Autoscaling Node Pool', 'Ingress Controller'] },
  { name: 'Docker Containers', icon: Container, items: ['API Service', 'Web Frontend', 'Celery Workers'] },
  { name: 'Security Layer', icon: Lock, items: ['UFW Firewall', 'JWT Auth', 'Rate Limiting'] },
  { name: 'CI/CD Pipeline', icon: GitBranch, items: ['GitHub Actions', 'Automated Testing', 'Blue/Green Deploy'] },
];

function ArchitectureDiagram() {
  const [isVisible, setIsVisible] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const nodes = [
    { id: 'sources', label: 'Data Sources', color: 'from-blue-500 to-cyan-500', icon: Radio, items: ['Binance WS', 'Coinbase', 'Kraken', 'Yahoo Finance'] },
    { id: 'kafka', label: 'Event Backbone', color: 'from-purple-500 to-pink-500', icon: Zap, items: ['Apache Kafka', 'Redis Pub/Sub', 'SSE Streaming'] },
    { id: 'processing', label: 'Processing', color: 'from-orange-500 to-red-500', icon: Cpu, items: ['FastAPI Cores', 'LangGraph Agents', 'Gemini AI'] },
    { id: 'storage', label: 'Persistence', color: 'from-green-500 to-emerald-500', icon: Database, items: ['PostgreSQL', 'Redis Cache', 'Qdrant Vectors'] },
    { id: 'presentation', label: 'Frontend', color: 'from-sky-500 to-blue-500', icon: Globe, items: ['Next.js 16', 'React 18', 'TailwindCSS'] },
  ];

  return (
    <div className={`relative p-8 md:p-12 rounded-3xl border border-zinc-800/50 bg-zinc-900/20 backdrop-blur-sm overflow-hidden
      transition-all duration-700 ${isVisible ? 'opacity-100' : 'opacity-0'}`}
    >
      {/* Animated background grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.02)_1px,transparent_1px)] bg-[size:40px_40px]" />

      {/* Glow effects */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-cyan-500/5 rounded-full blur-[100px]" />

      <div className="relative">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-900/20 border border-blue-500/20 text-sky-400 text-sm mb-4">
            <Layers className="w-4 h-4" />
            System Blueprint
          </div>
          <h3 className="text-3xl font-bold text-zinc-100 mb-2">Event-Driven Architecture</h3>
          <p className="text-zinc-500 text-sm">Interactive visualization of the data flow</p>
        </div>

        {/* Visual Flow Diagram */}
        <div className="relative flex flex-col gap-6">
          {nodes.map((node, index) => (
            <div key={node.id} className="relative">
              {/* Connection Line */}
              {index < nodes.length - 1 && (
                <div className="absolute left-1/2 -bottom-6 transform -translate-x-1/2 h-6 flex flex-col items-center z-0">
                  <div className="w-px h-full bg-gradient-to-b from-zinc-600 to-zinc-700 relative overflow-hidden">
                    {/* Animated flow indicator */}
                    <div className="absolute w-full h-8 bg-gradient-to-b from-transparent via-cyan-400/50 to-transparent animate-pulse"
                      style={{ animation: `flowDown 2s ease-in-out infinite`, animationDelay: `${index * 0.3}s` }} />
                  </div>
                  <div className="text-cyan-400 animate-pulse" style={{ animationDelay: `${index * 0.3}s` }}>▼</div>
                </div>
              )}

              {/* Node Card */}
              <div
                className={`relative group cursor-pointer transition-all duration-300
                  ${activeNode === node.id ? 'scale-[1.02] z-20' : 'z-10'}
                `}
                onMouseEnter={() => setActiveNode(node.id)}
                onMouseLeave={() => setActiveNode(null)}
              >
                {/* Card glow */}
                <div className={`absolute -inset-px rounded-2xl bg-gradient-to-r ${node.color} opacity-0 
                  group-hover:opacity-30 blur-sm transition-opacity duration-300`} />

                <div className={`relative flex items-center gap-6 p-5 rounded-2xl border border-zinc-800/80 
                  bg-zinc-900/60 backdrop-blur-sm transition-all duration-300
                  group-hover:border-zinc-700 group-hover:bg-zinc-900/80`}
                >
                  {/* Icon */}
                  <div className={`flex-shrink-0 w-14 h-14 rounded-xl bg-gradient-to-br ${node.color} 
                    flex items-center justify-center transition-transform duration-300 
                    group-hover:scale-110 group-hover:rotate-3 shadow-lg`}
                  >
                    <node.icon className="w-7 h-7 text-white" />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <h4 className="text-lg font-semibold text-zinc-100 mb-2 group-hover:text-white transition-colors">
                      {node.label}
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {node.items.map((item, i) => (
                        <span
                          key={item}
                          className={`px-3 py-1 text-xs rounded-full border transition-all duration-300
                            ${activeNode === node.id
                              ? 'bg-zinc-800 border-zinc-600 text-zinc-200'
                              : 'bg-zinc-900/50 border-zinc-800 text-zinc-500'
                            }`}
                          style={{ transitionDelay: `${i * 50}ms` }}
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Arrow indicator */}
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full border border-zinc-700 
                    flex items-center justify-center transition-all duration-300
                    group-hover:border-cyan-500/50 group-hover:bg-cyan-500/10`}
                  >
                    <ArrowRight className={`w-4 h-4 text-zinc-600 transition-all duration-300
                      group-hover:text-cyan-400 group-hover:translate-x-0.5`} />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-10 pt-6 border-t border-zinc-800/50 flex flex-wrap justify-center gap-6 text-xs text-zinc-500">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500" />
            <span>Ingestion</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500" />
            <span>Streaming</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-orange-500 to-red-500" />
            <span>Processing</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-500 to-emerald-500" />
            <span>Storage</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-sky-500 to-blue-500" />
            <span>Frontend</span>
          </div>
        </div>
      </div>

      {/* Flow animation keyframes */}
      <style jsx>{`
        @keyframes flowDown {
          0%, 100% { transform: translateY(-100%); opacity: 0; }
          50% { opacity: 1; }
          100% { transform: translateY(200%); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

export default function ArchitecturePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen relative z-10">
      <SiteHeader />

      {/* Hero */}
      <section className="relative pt-28 pb-16 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/30 via-transparent to-transparent" />

        <div className={`relative max-w-5xl mx-auto px-6 text-center
          transition-all duration-1000 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/80 border border-zinc-800 text-sm text-zinc-400 mb-6">
            <Server className="w-4 h-4 text-sky-400" />
            System Architecture
          </div>

          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="bg-gradient-to-r from-zinc-100 to-zinc-300 bg-clip-text text-transparent">
              Event-Driven
            </span>
            <br />
            <span className="bg-gradient-to-r from-sky-400 to-blue-500 bg-clip-text text-transparent">
              Microservices
            </span>
          </h1>

          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            Built on CQRS principles with separate hot/cold data paths.
            Every event is replayable, traceable, and optimized for sub-100ms latency.
          </p>
        </div>
      </section>

      {/* 2. Data Pipeline Section */}
      <section className="py-24 px-6 relative">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-900/20 border border-purple-500/20 text-purple-400 text-sm mb-4">
              <Zap className="w-3.5 h-3.5" />
              Data Flow Pipeline
            </div>
            <h2 className="text-3xl font-bold text-zinc-100 mb-4">100ms Latency Journey</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">From exchange websocket to client browser in four optimized steps.</p>
          </div>

          <div className="grid md:grid-cols-4 gap-6">
            {pipelineSteps.map((step, i) => (
              <div key={step.id} className="relative group">
                <div className={`p-6 rounded-2xl border border-zinc-800 bg-zinc-900/30 
                  hover:bg-zinc-900/50 hover:border-zinc-700 transition-all duration-300
                  hover:-translate-y-1 h-full`}>
                  <div className="flex justify-between items-start mb-4">
                    <div className={`w-10 h-10 rounded-lg ${step.bg} flex items-center justify-center ${step.color}`}>
                      <step.icon className="w-5 h-5" />
                    </div>
                    <span className="text-4xl font-bold text-zinc-800/50 select-none group-hover:text-zinc-800 transition-colors">
                      {step.id}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-zinc-100 mb-2">{step.title}</h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">{step.desc}</p>
                </div>
                {i < pipelineSteps.length - 1 && (
                  <div className="hidden md:block absolute top-1/2 -right-3 w-6 h-px bg-zinc-800 -z-10" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Main Blueprint Diagram */}
      <section className="py-12 px-6">
        <div className="max-w-5xl mx-auto">
          <ArchitectureDiagram />
        </div>
      </section>

      {/* 3. Infrastructure Topology */}
      <section className="py-24 px-6 bg-zinc-950/50">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-zinc-100 mb-4">Infrastructure Topology</h2>
            <p className="text-zinc-400">Production-grade deployment on Kubernetes</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {infraNodes.map((node) => (
              <div key={node.name} className="flex gap-6 p-6 rounded-2xl border border-zinc-800 bg-zinc-900/20 hover:bg-zinc-900/40 transition-colors">
                <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-zinc-800/50 flex items-center justify-center">
                  <node.icon className="w-6 h-6 text-zinc-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-zinc-200 mb-3">{node.name}</h3>
                  <ul className="space-y-2">
                    {node.items.map((item) => (
                      <li key={item} className="flex items-center gap-2 text-sm text-zinc-500">
                        <div className="w-1.5 h-1.5 rounded-full bg-zinc-700" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto flex flex-wrap justify-center gap-4">
          <Link
            href="/techdash"
            className="group flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-500 
              text-white rounded-xl font-semibold transition-all shadow-xl shadow-blue-500/20 
              hover:shadow-blue-500/40 hover:scale-[1.02]"
          >
            View Tech Stack
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
