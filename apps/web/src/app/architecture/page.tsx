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
  Check
} from 'lucide-react';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

// Architecture layers with animations
const layers = [
  {
    name: 'Data Ingestion',
    color: 'from-blue-500 to-cyan-500',
    icon: Radio,
    services: ['Binance WebSocket', 'Yahoo Finance API', 'News RSS Feeds'],
    delay: 0,
  },
  {
    name: 'Event Streaming',
    color: 'from-purple-500 to-pink-500',
    icon: Zap,
    services: ['Apache Kafka', 'Redis Pub/Sub', 'Event Sourcing'],
    delay: 100,
  },
  {
    name: 'Processing Layer',
    color: 'from-orange-500 to-red-500',
    icon: Cpu,
    services: ['FastAPI Backend', 'Async Workers', 'Stream Processing'],
    delay: 200,
  },
  {
    name: 'AI & ML',
    color: 'from-green-500 to-emerald-500',
    icon: Brain,
    services: ['LangGraph Agents', 'PyTorch LSTM', 'Qdrant Vector DB'],
    delay: 300,
  },
  {
    name: 'Storage',
    color: 'from-indigo-500 to-blue-500',
    icon: Database,
    services: ['TimescaleDB', 'Redis Cache', 'Model Registry'],
    delay: 400,
  },
  {
    name: 'Presentation',
    color: 'from-sky-500 to-blue-500',
    icon: Globe,
    services: ['Next.js 16', 'WebSocket Client', 'Real-time Charts'],
    delay: 500,
  },
];

const principles = [
  { title: 'Event-Driven', desc: 'Every action produces replayable events' },
  { title: 'CQRS Pattern', desc: 'Separate read/write for optimal performance' },
  { title: 'Microservices', desc: 'Independent, scalable service boundaries' },
  { title: 'Observability', desc: 'Prometheus, Grafana, Jaeger tracing' },
];

function LayerCard({ layer, index }: { layer: typeof layers[0]; index: number }) {
  const [isVisible, setIsVisible] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), layer.delay + 200);
    return () => clearTimeout(timer);
  }, [layer.delay]);

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`relative p-6 rounded-2xl border border-zinc-800 bg-zinc-900/50 backdrop-blur-sm
        transition-all duration-500 cursor-pointer
        ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
        ${isHovered ? 'border-zinc-600 shadow-xl shadow-blue-500/10 -translate-y-1' : ''}
      `}
    >
      {/* Gradient glow */}
      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${layer.color} opacity-0 
        ${isHovered ? 'opacity-5' : ''} transition-opacity duration-300`}
      />

      {/* Connection line to next */}
      {index < layers.length - 1 && (
        <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 h-8 w-px bg-gradient-to-b from-zinc-600 to-transparent hidden md:block" />
      )}

      <div className="relative">
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${layer.color} flex items-center justify-center mb-4
          transition-transform duration-300 ${isHovered ? 'scale-110 rotate-3' : ''}`}
        >
          <layer.icon className="w-6 h-6 text-white" />
        </div>

        <h3 className="text-xl font-semibold text-zinc-100 mb-3">{layer.name}</h3>

        <div className="space-y-2">
          {layer.services.map((service, i) => (
            <div
              key={service}
              className={`flex items-center gap-2 text-sm text-zinc-400 
                transition-all duration-300 delay-${i * 50}
                ${isHovered ? 'text-zinc-300 translate-x-1' : ''}`}
            >
              <ArrowRight className={`w-3 h-3 transition-transform ${isHovered ? 'translate-x-1' : ''}`} />
              {service}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ArchitectureDiagram() {
  const [isVisible, setIsVisible] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const nodes = [
    { id: 'sources', label: 'Data Sources', color: 'from-blue-500 to-cyan-500', icon: Radio, items: ['Binance WS', 'Yahoo Finance', 'News RSS', 'Social Media'] },
    { id: 'kafka', label: 'Event Backbone', color: 'from-purple-500 to-pink-500', icon: Zap, items: ['Apache Kafka', 'Event Streaming', 'Message Queues'] },
    { id: 'processing', label: 'Processing', color: 'from-orange-500 to-red-500', icon: Cpu, items: ['FastAPI', 'LangGraph Agents', 'PyTorch ML'] },
    { id: 'storage', label: 'Data Stores', color: 'from-green-500 to-emerald-500', icon: Database, items: ['TimescaleDB', 'Redis', 'Qdrant', 'MLflow'] },
    { id: 'presentation', label: 'Presentation', color: 'from-sky-500 to-blue-500', icon: Globe, items: ['Next.js 16', 'WebSocket', 'Real-time Charts'] },
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
            System Architecture
          </div>
          <h3 className="text-3xl font-bold text-zinc-100 mb-2">Event-Driven Data Flow</h3>
          <p className="text-zinc-500 text-sm">Hover over nodes to explore the architecture</p>
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
            <span>Data Ingestion</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500" />
            <span>Event Streaming</span>
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
            <span>Presentation</span>
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

      {/* Principles */}
      <section className="py-12 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {principles.map((p, i) => (
              <div
                key={p.title}
                className="group p-4 rounded-xl border border-zinc-800 bg-zinc-900/30 hover:border-zinc-700 
                  hover:bg-zinc-900/50 transition-all duration-300 hover:-translate-y-1"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span className="font-semibold text-zinc-100 group-hover:text-sky-400 transition-colors">{p.title}</span>
                </div>
                <p className="text-xs text-zinc-500">{p.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Main Diagram */}
      <section className="py-12 px-6">
        <div className="max-w-5xl mx-auto">
          <ArchitectureDiagram />
        </div>
      </section>

      {/* Layer Cards */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-zinc-100 mb-4">Architecture Layers</h2>
            <p className="text-zinc-400">Each layer has clear boundaries and responsibilities</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {layers.map((layer, index) => (
              <LayerCard key={layer.name} layer={layer} index={index} />
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
            href="/live"
            className="flex items-center gap-2 px-8 py-4 bg-zinc-900/80 border border-zinc-700 
              text-zinc-100 rounded-xl font-semibold hover:bg-zinc-800 hover:border-zinc-600 transition-all"
          >
            Launch Terminal
          </Link>
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}
