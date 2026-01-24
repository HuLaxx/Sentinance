'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
    ArrowLeft,
    TrendingUp,
    TrendingDown,
    Activity,
    BarChart3,
    Brain,
    AlertTriangle,
    RefreshCw,
    Home,
    ChevronRight,
    Sparkles,
    Target,
    Shield,
    Clock,
    Globe,
    Github,
    ArrowRight,
} from 'lucide-react';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

// ============================================
// ASSET CONFIGURATION
// ============================================

const ASSETS: Record<string, { name: string; type: 'crypto' | 'index'; region?: string }> = {
    // Crypto
    BTCUSDT: { name: 'Bitcoin', type: 'crypto' },
    ETHUSDT: { name: 'Ethereum', type: 'crypto' },
    SOLUSDT: { name: 'Solana', type: 'crypto' },
    XRPUSDT: { name: 'XRP', type: 'crypto' },
    BNBUSDT: { name: 'BNB', type: 'crypto' },
    ADAUSDT: { name: 'Cardano', type: 'crypto' },
    DOGEUSDT: { name: 'Dogecoin', type: 'crypto' },
    LINKUSDT: { name: 'Chainlink', type: 'crypto' },
    AVAXUSDT: { name: 'Avalanche', type: 'crypto' },
    DOTUSDT: { name: 'Polkadot', type: 'crypto' },
    MATICUSDT: { name: 'Polygon', type: 'crypto' },
    LTCUSDT: { name: 'Litecoin', type: 'crypto' },
    // Americas
    '^GSPC': { name: 'S&P 500', type: 'index', region: 'Americas' },
    '^DJI': { name: 'Dow Jones', type: 'index', region: 'Americas' },
    '^IXIC': { name: 'NASDAQ', type: 'index', region: 'Americas' },
    '^RUT': { name: 'Russell 2000', type: 'index', region: 'Americas' },
    '^GSPTSE': { name: 'TSX (Canada)', type: 'index', region: 'Americas' },
    '^BVSP': { name: 'Bovespa (Brazil)', type: 'index', region: 'Americas' },
    '^MXX': { name: 'IPC Mexico', type: 'index', region: 'Americas' },
    // Europe
    '^FTSE': { name: 'FTSE 100', type: 'index', region: 'Europe' },
    '^GDAXI': { name: 'DAX (Germany)', type: 'index', region: 'Europe' },
    '^FCHI': { name: 'CAC 40 (France)', type: 'index', region: 'Europe' },
    '^STOXX50E': { name: 'Euro Stoxx 50', type: 'index', region: 'Europe' },
    '^AEX': { name: 'AEX (Netherlands)', type: 'index', region: 'Europe' },
    '^IBEX': { name: 'IBEX 35 (Spain)', type: 'index', region: 'Europe' },
    '^SSMI': { name: 'SMI (Switzerland)', type: 'index', region: 'Europe' },
    // Asia
    '^N225': { name: 'Nikkei 225', type: 'index', region: 'Asia' },
    '^NSEI': { name: 'Nifty 50', type: 'index', region: 'Asia' },
    '^HSI': { name: 'Hang Seng', type: 'index', region: 'Asia' },
    '000001.SS': { name: 'Shanghai Composite', type: 'index', region: 'Asia' },
    '^KS11': { name: 'KOSPI', type: 'index', region: 'Asia' },
    '^TWII': { name: 'Taiwan Weighted', type: 'index', region: 'Asia' },
    '^AXJO': { name: 'ASX 200', type: 'index', region: 'Asia' },
    '^STI': { name: 'Straits Times', type: 'index', region: 'Asia' },
    '^BSESN': { name: 'BSE Sensex', type: 'index', region: 'Asia' },
    // MENA/Africa
    '^TA125': { name: 'TA-125 (Israel)', type: 'index', region: 'MENA' },
    '^TASI': { name: 'Tadawul (Saudi)', type: 'index', region: 'MENA' },
    '^J203': { name: 'JSE Top 40', type: 'index', region: 'Africa' },
};

// Demo prices
const DEMO_PRICES: Record<string, { price: number; change: number; high: number; low: number; volume: string }> = {
    BTCUSDT: { price: 89877, change: 0.09, high: 91225, low: 88578, volume: '13.4K BTC' },
    ETHUSDT: { price: 2965, change: 0.25, high: 3019, low: 2892, volume: '310K ETH' },
    SOLUSDT: { price: 127.94, change: -0.32, high: 130.20, low: 125.28, volume: '1.97M SOL' },
    XRPUSDT: { price: 1.93, change: 0.67, high: 1.97, low: 1.89, volume: '73M XRP' },
    '^GSPC': { price: 6915, change: 0.03, high: 6933, low: 6895, volume: '4.87B' },
    '^NSEI': { price: 25049, change: -0.95, high: 25348, low: 25025, volume: '393K' },
    '^FTSE': { price: 10143, change: -0.07, high: 10184, low: 10132, volume: '630M' },
    '^N225': { price: 53847, change: 0.29, high: 54051, low: 53604, volume: '129M' },
};

// ============================================
// DEMO NOTICE
// ============================================

function DemoNotice() {
    return (
        <div className="bg-gradient-to-r from-amber-900/30 via-amber-800/20 to-amber-900/30 border border-amber-500/30 rounded-xl p-4 mb-6">
            <div className="flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
                <div className="flex-1">
                    <span className="text-amber-200 font-semibold text-sm">Demo Mode</span>
                    <span className="text-amber-300/60 text-sm ml-2">• Simulated data</span>
                </div>
                <a
                    href="https://github.com/HuLaxx/Sentinance"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 rounded-lg text-amber-200 text-xs font-medium transition-colors"
                >
                    <Github className="w-3 h-3" />
                    Run Locally
                    <ArrowRight className="w-3 h-3" />
                </a>
            </div>
        </div>
    );
}

// ============================================
// PREDICTION CARD
// ============================================

function PredictionCard({
    title,
    value,
    change,
    confidence,
    timeframe,
    icon: Icon,
}: {
    title: string;
    value: string;
    change: number;
    confidence: number;
    timeframe: string;
    icon: React.ComponentType<{ className?: string }>;
}) {
    const isPositive = change >= 0;

    return (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 hover:border-zinc-700 transition-all">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4 text-sky-400" />
                    <span className="text-xs text-zinc-400">{title}</span>
                </div>
                <span className="text-[10px] text-zinc-500">{timeframe}</span>
            </div>
            <div className="flex items-end justify-between">
                <div>
                    <p className="text-xl font-bold text-zinc-100">{value}</p>
                    <div className={`flex items-center gap-1 mt-1 ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        <span className="text-xs font-medium">{isPositive ? '+' : ''}{change.toFixed(2)}%</span>
                    </div>
                </div>
                <div className="text-right">
                    <p className="text-[10px] text-zinc-500">Confidence</p>
                    <p className="text-sm font-semibold text-sky-400">{confidence}%</p>
                </div>
            </div>
        </div>
    );
}

// ============================================
// INDICATOR ROW
// ============================================

function IndicatorRow({ name, value, signal }: { name: string; value: string; signal: 'bullish' | 'bearish' | 'neutral' }) {
    const signalStyles = {
        bullish: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
        bearish: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
        neutral: 'text-zinc-400 bg-zinc-500/10 border-zinc-500/20',
    };

    return (
        <div className="flex items-center justify-between py-2.5 border-b border-zinc-800/50 last:border-0">
            <span className="text-sm text-zinc-400">{name}</span>
            <div className="flex items-center gap-2">
                <span className="text-sm font-mono text-zinc-200">{value}</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-medium capitalize border ${signalStyles[signal]}`}>
                    {signal}
                </span>
            </div>
        </div>
    );
}

// ============================================
// MAIN PAGE
// ============================================

export default function AssetDetailPage() {
    const params = useParams();
    const router = useRouter();
    const symbol = decodeURIComponent(params.symbol as string);

    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<{ price: number; change: number; high: number; low: number; volume: string } | null>(null);

    const asset = ASSETS[symbol];

    useEffect(() => {
        setLoading(true);

        // Simulate API fetch
        setTimeout(() => {
            const demoData = DEMO_PRICES[symbol] || {
                price: Math.random() * 10000 + 1000,
                change: (Math.random() - 0.5) * 5,
                high: 0,
                low: 0,
                volume: '1.2M',
            };
            demoData.high = demoData.high || demoData.price * 1.02;
            demoData.low = demoData.low || demoData.price * 0.98;
            setData(demoData);
            setLoading(false);
        }, 300);
    }, [symbol]);

    if (!asset) {
        return (
            <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-zinc-100 mb-4">Asset Not Found</h1>
                    <p className="text-zinc-500 mb-4">{symbol}</p>
                    <Link href="/demo" className="text-sky-400 hover:text-sky-300">
                        ← Back to Demo
                    </Link>
                </div>
            </div>
        );
    }

    const price = data?.price || 0;
    const change = data?.change || 0;
    const isPositive = change >= 0;

    // Generate predictions
    const predictions = {
        short: { value: price * (1 + 0.012), change: 1.2, confidence: 78 },
        medium: { value: price * (1 + 0.035), change: 3.5, confidence: 72 },
        long: { value: price * (1 + 0.082), change: 8.2, confidence: 65 },
    };

    // Mock indicators
    const indicators = [
        { name: 'RSI (14)', value: '62.4', signal: 'neutral' as const },
        { name: 'MACD', value: '+145.2', signal: 'bullish' as const },
        { name: 'Bollinger Bands', value: 'Upper Zone', signal: 'bearish' as const },
        { name: 'SMA 20', value: `${(price * 0.99).toFixed(2)}`, signal: 'bullish' as const },
        { name: 'SMA 50', value: `${(price * 0.97).toFixed(2)}`, signal: 'bullish' as const },
        { name: 'EMA 12', value: `${(price * 0.995).toFixed(2)}`, signal: 'bullish' as const },
        { name: 'Volume Trend', value: '1.2x avg', signal: 'bullish' as const },
        { name: 'ATR (14)', value: `${(price * 0.02).toFixed(2)}`, signal: 'neutral' as const },
    ];

    return (
        <div className="min-h-screen bg-zinc-950">
            <SiteHeader />

            <main className="pt-24 pb-16 px-4 md:px-8 max-w-6xl mx-auto">
                {/* Navigation */}
                <div className="flex items-center gap-4 mb-4">
                    <button
                        onClick={() => router.back()}
                        className="flex items-center gap-2 px-3 py-2 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-lg text-zinc-300 text-sm transition-colors"
                    >
                        <ArrowLeft className="w-4 h-4" />
                        Back
                    </button>

                    <nav className="flex items-center gap-2 text-sm text-zinc-500">
                        <Link href="/" className="hover:text-zinc-300"><Home className="w-4 h-4" /></Link>
                        <ChevronRight className="w-3 h-3" />
                        <Link href="/demo" className="hover:text-zinc-300">Demo</Link>
                        <ChevronRight className="w-3 h-3" />
                        <span className="text-zinc-300">{asset.name}</span>
                    </nav>
                </div>

                {/* Demo Notice */}
                <DemoNotice />

                {/* Asset Header Card */}
                <div className="bg-zinc-900/50 border border-zinc-800 rounded-2xl p-6 mb-6">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div>
                            <div className="flex items-center gap-2 mb-2">
                                <span className={`px-2 py-1 rounded text-xs font-medium uppercase ${asset.type === 'crypto' ? 'bg-sky-500/10 text-sky-400 border border-sky-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                    }`}>
                                    {asset.type === 'crypto' ? 'Crypto' : asset.region || 'Index'}
                                </span>
                                <span className="flex items-center gap-1.5">
                                    <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                                    <span className="text-xs text-zinc-500">Demo</span>
                                </span>
                            </div>
                            <h1 className="text-3xl font-bold text-zinc-100">{asset.name}</h1>
                            <p className="text-zinc-500 text-sm">{symbol}</p>
                        </div>

                        <div className="text-right">
                            {loading ? (
                                <RefreshCw className="w-6 h-6 animate-spin text-zinc-500" />
                            ) : (
                                <>
                                    <p className="text-4xl font-bold font-mono text-zinc-100">
                                        {asset.type === 'crypto' ? '$' : ''}{price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                    </p>
                                    <div className={`flex items-center justify-end gap-2 mt-1 ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                                        {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                                        <span className="text-sm font-medium">{isPositive ? '+' : ''}{change.toFixed(2)}% (24h)</span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Stats Row */}
                    {data && (
                        <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-zinc-800">
                            <div>
                                <p className="text-xs text-zinc-500">24h High</p>
                                <p className="text-sm font-mono text-zinc-200">{data.high.toLocaleString()}</p>
                            </div>
                            <div>
                                <p className="text-xs text-zinc-500">24h Low</p>
                                <p className="text-sm font-mono text-zinc-200">{data.low.toLocaleString()}</p>
                            </div>
                            <div>
                                <p className="text-xs text-zinc-500">Volume</p>
                                <p className="text-sm font-mono text-zinc-200">{data.volume}</p>
                            </div>
                            <div>
                                <p className="text-xs text-zinc-500">Market Status</p>
                                <p className="text-sm text-emerald-400">Open</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column - Predictions */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* ML Predictions */}
                        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-5">
                            <div className="flex items-center gap-2 mb-4">
                                <Brain className="w-5 h-5 text-sky-400" />
                                <h2 className="text-lg font-semibold text-zinc-100">ML Price Predictions</h2>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <PredictionCard
                                    title="Short Term"
                                    value={`$${predictions.short.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                                    change={predictions.short.change}
                                    confidence={predictions.short.confidence}
                                    timeframe="4 Hours"
                                    icon={Clock}
                                />
                                <PredictionCard
                                    title="Medium Term"
                                    value={`$${predictions.medium.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                                    change={predictions.medium.change}
                                    confidence={predictions.medium.confidence}
                                    timeframe="24 Hours"
                                    icon={Target}
                                />
                                <PredictionCard
                                    title="Long Term"
                                    value={`$${predictions.long.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                                    change={predictions.long.change}
                                    confidence={predictions.long.confidence}
                                    timeframe="7 Days"
                                    icon={TrendingUp}
                                />
                            </div>
                        </div>

                        {/* AI Analysis */}
                        <div className="bg-gradient-to-br from-sky-900/20 to-blue-900/20 border border-sky-500/20 rounded-2xl p-5">
                            <div className="flex items-center gap-2 mb-4">
                                <Sparkles className="w-5 h-5 text-sky-400" />
                                <h2 className="text-lg font-semibold text-zinc-100">AI Market Analysis</h2>
                            </div>

                            <p className="text-sm text-zinc-300 leading-relaxed mb-4">
                                {asset.name} is showing <span className="text-emerald-400 font-medium">bullish momentum</span> with
                                RSI in neutral territory indicating room for growth. MACD shows positive crossover while
                                volume is {change >= 0 ? 'above' : 'below'} average. Key resistance at{' '}
                                <span className="font-mono text-sky-300">${(price * 1.05).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span> and
                                support at <span className="font-mono text-sky-300">${(price * 0.95).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>.
                            </p>

                            <div className="flex items-center justify-between text-xs text-zinc-500 pt-4 border-t border-zinc-700/50">
                                <span>Powered by Sentinance AI</span>
                                <span className="text-sky-400">Confidence: 72%</span>
                            </div>
                        </div>

                        {/* Order Flow (placeholder) */}
                        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-5">
                            <div className="flex items-center gap-2 mb-4">
                                <Activity className="w-5 h-5 text-purple-400" />
                                <h2 className="text-lg font-semibold text-zinc-100">Order Flow Analysis</h2>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4">
                                    <p className="text-xs text-emerald-400 mb-1">Buy Pressure</p>
                                    <p className="text-2xl font-bold text-emerald-400">62%</p>
                                    <div className="h-2 bg-zinc-800 rounded-full mt-2">
                                        <div className="h-full bg-emerald-500 rounded-full" style={{ width: '62%' }} />
                                    </div>
                                </div>
                                <div className="bg-rose-500/10 border border-rose-500/20 rounded-lg p-4">
                                    <p className="text-xs text-rose-400 mb-1">Sell Pressure</p>
                                    <p className="text-2xl font-bold text-rose-400">38%</p>
                                    <div className="h-2 bg-zinc-800 rounded-full mt-2">
                                        <div className="h-full bg-rose-500 rounded-full" style={{ width: '38%' }} />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right Column - Indicators */}
                    <div className="space-y-6">
                        {/* Technical Indicators */}
                        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-5">
                            <div className="flex items-center gap-2 mb-4">
                                <BarChart3 className="w-5 h-5 text-emerald-400" />
                                <h2 className="text-lg font-semibold text-zinc-100">Technical Indicators</h2>
                            </div>

                            <div>
                                {indicators.map((ind) => (
                                    <IndicatorRow key={ind.name} {...ind} />
                                ))}
                            </div>
                        </div>

                        {/* Risk Assessment */}
                        <div className="bg-zinc-900/30 border border-zinc-800 rounded-2xl p-5">
                            <div className="flex items-center gap-2 mb-4">
                                <Shield className="w-5 h-5 text-amber-400" />
                                <h2 className="text-lg font-semibold text-zinc-100">Risk Assessment</h2>
                            </div>

                            <div className="space-y-3">
                                <div className="flex justify-between text-sm">
                                    <span className="text-zinc-400">Volatility</span>
                                    <span className="text-amber-400">Medium</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-zinc-400">Trend Strength</span>
                                    <span className="text-emerald-400">Strong</span>
                                </div>
                                <div className="flex justify-between text-sm">
                                    <span className="text-zinc-400">Risk/Reward</span>
                                    <span className="text-sky-400">1:2.5</span>
                                </div>
                            </div>
                        </div>

                        {/* Quick Actions */}
                        <div className="flex flex-col gap-3">
                            <Link
                                href="/demo"
                                className="flex items-center justify-center gap-2 px-4 py-3 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-xl text-zinc-200 font-medium transition-colors"
                            >
                                <Globe className="w-4 h-4" />
                                View All Assets
                            </Link>
                            <Link
                                href="/"
                                className="flex items-center justify-center gap-2 px-4 py-3 bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 rounded-xl text-zinc-400 text-sm transition-colors"
                            >
                                <Home className="w-4 h-4" />
                                Back to Home
                            </Link>
                        </div>
                    </div>
                </div>
            </main>

            <SiteFooter />
        </div>
    );
}
