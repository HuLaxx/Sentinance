'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import {
  Activity,
  TrendingUp,
  BarChart3,
  Zap,
  Brain,
  Database,
  Layers,
  ArrowRight,
  Play,
  Sparkles,
  Globe,
  Shield,
  Cpu,
  LineChart,
  MessageSquare,
  AlertTriangle,
  TrendingDown
} from 'lucide-react';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

const features = [
  {
    icon: Activity,
    title: 'Real-Time Streaming',
    description: 'WebSocket-powered live feeds for crypto (BTC, ETH, SOL, XRP) and global indices (US, India, UK, Japan).',
    color: 'from-blue-900 to-slate-900',
    delay: 0,
  },
  {
    icon: Brain,
    title: 'Agentic AI',
    description: 'LangGraph multi-agent system for autonomous market analysis and insights.',
    color: 'from-sky-500 to-blue-600',
    delay: 100,
  },
  {
    icon: TrendingUp,
    title: 'ML Predictions',
    description: 'LSTM + momentum models for price forecasting with confidence intervals.',
    color: 'from-cyan-500 to-blue-500',
    delay: 200,
  },
  {
    icon: Zap,
    title: 'Anomaly Detection',
    description: 'Real-time alerts for price spikes, volume surges, and manipulation patterns.',
    color: 'from-blue-400 to-indigo-500',
    delay: 300,
  },
  {
    icon: Database,
    title: 'RAG Pipeline',
    description: 'Qdrant vector store + semantic search for intelligent document retrieval.',
    color: 'from-slate-400 to-zinc-500',
    delay: 400,
  },
  {
    icon: Layers,
    title: 'Full-Stack',
    description: 'Next.js 16, FastAPI, WebSockets, Redis - production-grade architecture.',
    color: 'from-indigo-600 to-blue-800',
    delay: 500,
  },
];

const stats = [
  { value: '126+', label: 'Tests Passing', icon: Brain },
  { value: '24/7', label: 'Live Data', icon: Globe },
  { value: '<100ms', label: 'Response Time', icon: Zap },
  { value: '8', label: 'Markets', icon: Database },
];

function AnimatedNumber({ value, label, icon: Icon }: { value: string; label: string; icon: React.ComponentType<{ className?: string }> }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`text-center transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-zinc-800/50 border border-zinc-700/50 flex items-center justify-center">
        <Icon className="w-5 h-5 text-sky-400" />
      </div>
      <p className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-sky-300 to-blue-500 bg-clip-text text-transparent">
        {value}
      </p>
      <p className="text-sm text-zinc-500 mt-1">{label}</p>
    </div>
  );
}

function FeatureCard({ feature, index }: { feature: typeof features[0]; index: number }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="group relative p-6 bg-zinc-900/50 border border-zinc-800 rounded-2xl hover:border-zinc-700 transition-all duration-300 hover:-translate-y-1 hover:shadow-xl hover:shadow-indigo-500/5"
      style={{ animationDelay: `${feature.delay}ms` }}
    >
      {/* Glow effect on hover */}
      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`} />

      <div className={`relative w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 transition-transform duration-300 ${isHovered ? 'scale-110 rotate-3' : ''}`}>
        <feature.icon className="w-6 h-6 text-white" />
      </div>

      <h3 className="text-xl font-semibold text-zinc-100 mb-2 group-hover:text-sky-400 transition-colors">
        {feature.title}
      </h3>
      <p className="text-zinc-400 text-sm leading-relaxed">
        {feature.description}
      </p>
    </div>
  );
}

export default function HomePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen overflow-x-hidden relative z-10">
      <SiteHeader />

      {/* Hero Section */}
      <section className="relative pt-28 pb-20 overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/30 via-transparent to-transparent" />
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-blue-900/10 rounded-full blur-[150px] animate-pulse" />
          <div className="absolute top-20 left-1/4 w-[400px] h-[400px] bg-sky-500/5 rounded-full blur-[100px]" />
          <div className="absolute top-40 right-1/4 w-[300px] h-[300px] bg-blue-500/5 rounded-full blur-[100px]" />
        </div>

        {/* Grid pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_70%)]" />

        <div className="relative max-w-7xl mx-auto px-6">
          <div className={`text-center max-w-4xl mx-auto transition-all duration-1000 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-zinc-900/80 border border-zinc-800 text-sm text-zinc-400 mb-8 backdrop-blur-sm">
              <Sparkles className="w-4 h-4 text-sky-400" />
              Production-Ready • Enterprise-Grade Architecture
            </div>

            {/* Title */}
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-100 bg-clip-text text-transparent">
                Real-Time Crypto
              </span>
              <br />
              <span className="bg-gradient-to-r from-sky-300 via-blue-500 to-sky-300 bg-clip-text text-transparent animate-gradient-x drop-shadow-sm">
                Market Intelligence
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-lg md:text-xl text-zinc-400 mb-10 max-w-2xl mx-auto leading-relaxed">
              Transform your trading with AI-powered insights, real-time market data,
              and predictive analytics. Built for traders who demand excellence.
            </p>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/demo"
                className="group flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-xl font-semibold text-lg transition-all shadow-xl shadow-blue-500/20 hover:shadow-blue-500/40 hover:scale-[1.02]"
              >
                <Play className="w-5 h-5 transition-transform group-hover:scale-110" />
                Launch Demo
                <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link
                href="/architecture"
                className="flex items-center gap-2 px-8 py-4 bg-zinc-900/80 border border-zinc-700 text-zinc-100 rounded-xl font-semibold text-lg hover:bg-zinc-800 hover:border-zinc-600 transition-all backdrop-blur-sm"
              >
                View Architecture
              </Link>
            </div>
          </div>

          {/* Stats */}
          <div className={`grid grid-cols-2 md:grid-cols-4 gap-8 mt-20 transition-all duration-1000 delay-300 ${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            {stats.map((stat) => (
              <AnimatedNumber key={stat.label} {...stat} />
            ))}
          </div>
        </div>
      </section>

      {/* Live Intelligence Preview Section */}
      <section className="py-20 relative">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
            {/* Card 1: AI Analysis */}
            <div className="p-6 rounded-2xl bg-zinc-900/50 border border-zinc-800 backdrop-blur-sm hover:border-zinc-700 transition-all hover:-translate-y-1 group">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center text-purple-400">
                  <Brain className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-semibold text-zinc-100">AI Sentiment</h3>
                  <p className="text-xs text-zinc-500">Real-time Analysis</p>
                </div>
                <div className="ml-auto px-2 py-1 rounded bg-green-500/10 text-green-400 text-xs font-medium">
                  Bullish
                </div>
              </div>
              <div className="space-y-3">
                <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 w-[75%] animate-pulse" />
                </div>
                <p className="text-sm text-zinc-400">
                  Market sentiment leaning bullish on BTC due to institutional inflow signals.
                </p>
              </div>
            </div>

            {/* Card 2: Market Alerts */}
            <div className="p-6 rounded-2xl bg-zinc-900/50 border border-zinc-800 backdrop-blur-sm hover:border-zinc-700 transition-all hover:-translate-y-1 group mt-0 md:-mt-8">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-orange-500/10 flex items-center justify-center text-orange-400">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-semibold text-zinc-100">Whale Alert</h3>
                  <p className="text-xs text-zinc-500">On-chain Activity</p>
                </div>
                <span className="ml-auto text-xs text-zinc-500">Just now</span>
              </div>
              <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 mb-2">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-bold text-zinc-200">BTC-USDT</span>
                  <span className="text-sm text-red-400">-1.2%</span>
                </div>
                <p className="text-xs text-zinc-500">High volume sell pressure detected on key resistance levels.</p>
              </div>
            </div>

            {/* Card 3: ML Prediction */}
            <div className="p-6 rounded-2xl bg-zinc-900/50 border border-zinc-800 backdrop-blur-sm hover:border-zinc-700 transition-all hover:-translate-y-1 group">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400">
                  <TrendingUp className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-semibold text-zinc-100">ML Forecast</h3>
                  <p className="text-xs text-zinc-500">Next 4 Hours</p>
                </div>
                <div className="ml-auto px-2 py-1 rounded bg-blue-500/10 text-blue-400 text-xs font-medium">
                  +2.4%
                </div>
              </div>
              <div className="flex items-end gap-1 h-16 w-full opacity-80">
                {[40, 60, 45, 70, 65, 85, 80].map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 bg-gradient-to-t from-blue-900/20 to-blue-500 rounded-t-sm transition-all duration-500 hover:opacity-100"
                    style={{ height: `${h}%` }}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="mt-12 text-center relative z-10">
            <Link
              href="/demo"
              className="inline-flex items-center gap-2 px-6 py-3 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded-full font-medium transition-colors border border-zinc-700"
            >
              <Activity className="w-4 h-4 text-green-400" />
              View Demo Dashboard
            </Link>
          </div>

          {/* Background decoration */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-full bg-gradient-to-b from-blue-500/5 via-purple-500/5 to-transparent rounded-[100%] blur-3xl -z-10" />
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 relative">

        <div className="relative max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-900/20 border border-blue-500/20 text-sky-400 text-sm mb-4">
              <Shield className="w-4 h-4" />
              Enterprise Features
            </div>
            <h2 className="text-3xl md:text-5xl font-bold text-zinc-100 mb-4">
              Production-Ready Platform
            </h2>
            <p className="text-zinc-400 max-w-2xl mx-auto text-lg">
              Built with modern technologies and best practices for
              scalability, performance, and reliability.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <FeatureCard key={feature.title} feature={feature} index={index} />
            ))}
          </div>
        </div>
      </section>


      <SiteFooter />
    </div>
  );
}
