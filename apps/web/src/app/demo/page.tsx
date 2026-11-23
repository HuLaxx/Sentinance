'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Home,
  Plus,
  X,
  Sparkles,
  Globe,
  ArrowRight,
  Github,
} from 'lucide-react';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

// ============================================
// ASSET CONFIGURATION - Extended Global Coverage
// ============================================

type Asset = {
  symbol: string;
  name: string;
  type: 'crypto' | 'index';
  price: number;
  change: number;
  enabled: boolean;
  region?: string;
};

const INITIAL_ASSETS: Asset[] = [
  // ===== CRYPTOCURRENCIES =====
  // Top 4 (enabled by default)
  { symbol: 'BTCUSDT', name: 'Bitcoin', type: 'crypto', price: 89877, change: 0.09, enabled: true },
  { symbol: 'ETHUSDT', name: 'Ethereum', type: 'crypto', price: 2965, change: 0.25, enabled: true },
  { symbol: 'SOLUSDT', name: 'Solana', type: 'crypto', price: 127.94, change: -0.32, enabled: true },
  { symbol: 'XRPUSDT', name: 'XRP', type: 'crypto', price: 1.93, change: 0.67, enabled: true },
  // Additional cryptos
  { symbol: 'BNBUSDT', name: 'BNB', type: 'crypto', price: 710, change: 1.23, enabled: false },
  { symbol: 'ADAUSDT', name: 'Cardano', type: 'crypto', price: 0.95, change: 3.45, enabled: false },
  { symbol: 'DOGEUSDT', name: 'Dogecoin', type: 'crypto', price: 0.38, change: 8.92, enabled: false },
  { symbol: 'LINKUSDT', name: 'Chainlink', type: 'crypto', price: 28.50, change: 2.15, enabled: false },
  { symbol: 'AVAXUSDT', name: 'Avalanche', type: 'crypto', price: 42.80, change: 4.55, enabled: false },
  { symbol: 'DOTUSDT', name: 'Polkadot', type: 'crypto', price: 8.45, change: -1.20, enabled: false },
  { symbol: 'MATICUSDT', name: 'Polygon', type: 'crypto', price: 1.25, change: 2.30, enabled: false },
  { symbol: 'LTCUSDT', name: 'Litecoin', type: 'crypto', price: 125.60, change: 1.80, enabled: false },

  // ===== GLOBAL INDICES =====
  // Americas
  { symbol: '^GSPC', name: 'S&P 500', type: 'index', price: 6915, change: 0.03, enabled: true, region: 'Americas' },
  { symbol: '^DJI', name: 'Dow Jones', type: 'index', price: 44156, change: 0.12, enabled: true, region: 'Americas' },
  { symbol: '^IXIC', name: 'NASDAQ', type: 'index', price: 19756, change: -0.18, enabled: true, region: 'Americas' },
  { symbol: '^RUT', name: 'Russell 2000', type: 'index', price: 2287, change: 0.45, enabled: false, region: 'Americas' },
  { symbol: '^GSPTSE', name: 'TSX (Canada)', type: 'index', price: 25120, change: 0.22, enabled: false, region: 'Americas' },
  { symbol: '^BVSP', name: 'Bovespa (Brazil)', type: 'index', price: 128450, change: -0.35, enabled: false, region: 'Americas' },
  { symbol: '^MXX', name: 'IPC Mexico', type: 'index', price: 56780, change: 0.18, enabled: false, region: 'Americas' },

  // Europe
  { symbol: '^FTSE', name: 'FTSE 100', type: 'index', price: 10143, change: -0.07, enabled: true, region: 'Europe' },
  { symbol: '^GDAXI', name: 'DAX (Germany)', type: 'index', price: 21520, change: 0.35, enabled: true, region: 'Europe' },
  { symbol: '^FCHI', name: 'CAC 40 (France)', type: 'index', price: 7856, change: 0.28, enabled: false, region: 'Europe' },
  { symbol: '^STOXX50E', name: 'Euro Stoxx 50', type: 'index', price: 5089, change: 0.42, enabled: false, region: 'Europe' },
  { symbol: '^AEX', name: 'AEX (Netherlands)', type: 'index', price: 912, change: 0.15, enabled: false, region: 'Europe' },
  { symbol: '^IBEX', name: 'IBEX 35 (Spain)', type: 'index', price: 11890, change: -0.22, enabled: false, region: 'Europe' },
  { symbol: '^SSMI', name: 'SMI (Switzerland)', type: 'index', price: 12456, change: 0.08, enabled: false, region: 'Europe' },

  // Asia Pacific
  { symbol: '^N225', name: 'Nikkei 225', type: 'index', price: 53847, change: 0.29, enabled: true, region: 'Asia' },
  { symbol: '^NSEI', name: 'Nifty 50 (India)', type: 'index', price: 25049, change: -0.95, enabled: true, region: 'Asia' },
  { symbol: '^HSI', name: 'Hang Seng', type: 'index', price: 19876, change: 1.25, enabled: true, region: 'Asia' },
  { symbol: '000001.SS', name: 'Shanghai Composite', type: 'index', price: 3256, change: 0.45, enabled: false, region: 'Asia' },
  { symbol: '^KS11', name: 'KOSPI (Korea)', type: 'index', price: 2534, change: 0.67, enabled: false, region: 'Asia' },
  { symbol: '^TWII', name: 'Taiwan Weighted', type: 'index', price: 22890, change: 0.38, enabled: false, region: 'Asia' },
  { symbol: '^AXJO', name: 'ASX 200 (Australia)', type: 'index', price: 8456, change: 0.22, enabled: false, region: 'Asia' },
  { symbol: '^STI', name: 'Straits Times', type: 'index', price: 3678, change: -0.12, enabled: false, region: 'Asia' },
  { symbol: '^BSESN', name: 'BSE Sensex', type: 'index', price: 82560, change: -0.88, enabled: false, region: 'Asia' },

  // Middle East & Africa
  { symbol: '^TA125', name: 'TA-125 (Israel)', type: 'index', price: 2156, change: 0.55, enabled: false, region: 'MENA' },
  { symbol: '^TASI', name: 'Tadawul (Saudi)', type: 'index', price: 12340, change: 0.18, enabled: false, region: 'MENA' },
  { symbol: '^J203', name: 'JSE Top 40 (SA)', type: 'index', price: 78650, change: -0.42, enabled: false, region: 'Africa' },
];

// ============================================
// DEMO NOTICE COMPONENT
// ============================================

function DemoNotice({ isLive }: { isLive: boolean }) {
  return (
    <div className={`bg-gradient-to-r ${isLive ? 'from-emerald-900/40 via-emerald-800/30 to-emerald-900/40 border-emerald-500/40' : 'from-amber-900/40 via-amber-800/30 to-amber-900/40 border-amber-500/40'} border rounded-2xl p-5 mb-8`}>
      <div className="flex flex-col md:flex-row md:items-center gap-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl ${isLive ? 'bg-emerald-500/20' : 'bg-amber-500/20'} flex items-center justify-center`}>
            {isLive ? (
              <span className="h-3 w-3 rounded-full bg-emerald-400 animate-pulse" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            )}
          </div>
          <div>
            <h3 className={`${isLive ? 'text-emerald-200' : 'text-amber-200'} font-bold`}>
              {isLive ? 'Live Data' : 'Demo Mode'}
            </h3>
            <p className={`${isLive ? 'text-emerald-300/70' : 'text-amber-300/70'} text-xs`}>
              {isLive ? 'Connected to API • Real-time updates' : 'Using fallback data'}
            </p>
          </div>
        </div>

        <div className="flex-1 md:text-center">
          <p className={`${isLive ? 'text-emerald-100/80' : 'text-amber-100/80'} text-sm`}>
            {isLive ? (
              <>Streaming live prices from <span className="text-emerald-300 font-semibold">Binance + yfinance</span></>
            ) : (
              <>Free hosted demo. For <span className="text-amber-300 font-semibold">full live experience</span> →</>
            )}
          </p>
        </div>

        <a
          href="https://github.com/HuLaxx/Sentinance"
          target="_blank"
          rel="noopener noreferrer"
          className={`inline-flex items-center gap-2 px-4 py-2.5 ${isLive ? 'bg-emerald-500/20 hover:bg-emerald-500/30 border-emerald-500/50' : 'bg-amber-500/20 hover:bg-amber-500/30 border-amber-500/50'} border rounded-xl ${isLive ? 'text-emerald-100' : 'text-amber-100'} font-medium text-sm transition-all hover:scale-[1.02]`}
        >
          <Github className="w-4 h-4" />
          {isLive ? 'View Source' : 'Clone & Run Locally'}
          <ArrowRight className="w-3 h-3" />
        </a>
      </div>
    </div>
  );
}

// ============================================
// ASSET CARD COMPONENT (Simplified)
// ============================================

function AssetCard({ asset, onRemove }: { asset: Asset; onRemove?: () => void }) {
  const isPositive = asset.change >= 0;

  return (
    <Link
      href={`/asset/${encodeURIComponent(asset.symbol)}`}
      className={`relative group block p-4 rounded-xl border transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg ${asset.type === 'crypto'
        ? 'bg-zinc-900/60 border-zinc-800 hover:border-sky-500/50 hover:shadow-sky-500/5'
        : 'bg-zinc-900/60 border-emerald-500/20 hover:border-emerald-500/50 hover:shadow-emerald-500/5'
        }`}
    >
      {/* Remove button */}
      {onRemove && (
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onRemove();
          }}
          className="absolute top-2 right-2 p-1 rounded bg-zinc-800 hover:bg-rose-500/20 text-zinc-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition-all z-10"
        >
          <X className="w-3 h-3" />
        </button>
      )}

      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded ${asset.type === 'crypto' ? 'bg-sky-500/10 text-sky-400' : 'bg-emerald-500/10 text-emerald-400'
            }`}>
            {asset.type === 'crypto' ? 'Crypto' : asset.region || 'Index'}
          </span>
        </div>
        <div className={`flex items-center gap-1 text-xs font-medium ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
          {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          {isPositive ? '+' : ''}{asset.change.toFixed(2)}%
        </div>
      </div>

      <h3 className="font-semibold text-zinc-100 text-sm">{asset.name}</h3>
      <p className="text-[10px] text-zinc-500 mb-2">{asset.symbol}</p>

      <p className="text-lg font-bold font-mono text-zinc-100">
        {asset.type === 'crypto' ? '$' : ''}{asset.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </p>

      {/* Click hint */}
      <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <ArrowRight className="w-4 h-4 text-zinc-500" />
      </div>
    </Link>
  );
}

// ============================================
// ADD ASSET MODAL
// ============================================

function AddAssetModal({
  availableAssets,
  onAdd,
  onClose
}: {
  availableAssets: Asset[];
  onAdd: (symbol: string) => void;
  onClose: () => void;
}) {
  const cryptoAssets = availableAssets.filter(a => a.type === 'crypto');
  const indexAssets = availableAssets.filter(a => a.type === 'index');

  // Group indices by region
  const indexesByRegion = indexAssets.reduce((acc, asset) => {
    const region = asset.region || 'Other';
    if (!acc[region]) acc[region] = [];
    acc[region].push(asset);
    return acc;
  }, {} as Record<string, Asset[]>);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 w-full max-w-2xl max-h-[85vh] overflow-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-zinc-100">Add Assets</h2>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Crypto Section */}
        {cryptoAssets.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-sky-400 mb-3 flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              Cryptocurrencies
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {cryptoAssets.map((asset) => (
                <button
                  key={asset.symbol}
                  onClick={() => onAdd(asset.symbol)}
                  className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 border border-zinc-700/50 hover:border-sky-500/30 transition-all text-left"
                >
                  <div>
                    <p className="font-medium text-zinc-100 text-sm">{asset.name}</p>
                    <p className="text-xs text-zinc-500">{asset.symbol}</p>
                  </div>
                  <Plus className="w-4 h-4 text-zinc-400" />
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Indices by Region */}
        {Object.entries(indexesByRegion).map(([region, assets]) => (
          <div key={region} className="mb-6">
            <h3 className="text-sm font-semibold text-emerald-400 mb-3 flex items-center gap-2">
              <Globe className="w-4 h-4" />
              {region}
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {assets.map((asset) => (
                <button
                  key={asset.symbol}
                  onClick={() => onAdd(asset.symbol)}
                  className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 border border-zinc-700/50 hover:border-emerald-500/30 transition-all text-left"
                >
                  <div>
                    <p className="font-medium text-zinc-100 text-sm">{asset.name}</p>
                    <p className="text-xs text-zinc-500">{asset.symbol}</p>
                  </div>
                  <Plus className="w-4 h-4 text-zinc-400" />
                </button>
              ))}
            </div>
          </div>
        ))}

        {availableAssets.length === 0 && (
          <p className="text-zinc-500 text-center py-8">All available assets are on your dashboard.</p>
        )}
      </div>
    </div>
  );
}

// ============================================
// MAIN PAGE COMPONENT
// ============================================

export default function DemoPage() {
  const [assets, setAssets] = useState<Asset[]>(INITIAL_ASSETS);
  const [showAddModal, setShowAddModal] = useState(false);
  const [isLive, setIsLive] = useState(false);

  // Fetch live prices from API
  useEffect(() => {
    const fetchPrices = async () => {
      try {
        const res = await fetch('/api/prices', { cache: 'no-store' });
        if (!res.ok) throw new Error('API unavailable');
        const data = await res.json();

        if (data.prices && Array.isArray(data.prices)) {
          setIsLive(typeof data.isLive === 'boolean' ? data.isLive : true);
          setAssets(prev => prev.map(asset => {
            const livePrice = data.prices.find((p: { symbol: string }) => p.symbol === asset.symbol);
            if (livePrice) {
              return {
                ...asset,
                price: livePrice.price,
                change: livePrice.change_24h ?? livePrice.change ?? asset.change,
              };
            }
            return asset;
          }));
        }
      } catch {
        setIsLive(false);
        // Keep using static fallback data
      }
    };

    fetchPrices();
    const interval = setInterval(fetchPrices, 5000);
    return () => clearInterval(interval);
  }, []);

  const enabledAssets = assets.filter(a => a.enabled);
  const disabledAssets = assets.filter(a => !a.enabled);
  const cryptoAssets = enabledAssets.filter(a => a.type === 'crypto');
  const indexAssets = enabledAssets.filter(a => a.type === 'index');

  const handleAddAsset = (symbol: string) => {
    setAssets(prev => prev.map(a =>
      a.symbol === symbol ? { ...a, enabled: true } : a
    ));
    setShowAddModal(false);
  };

  const handleRemoveAsset = (symbol: string) => {
    setAssets(prev => prev.map(a =>
      a.symbol === symbol ? { ...a, enabled: false } : a
    ));
  };

  return (
    <div className="min-h-screen bg-zinc-950">
      <SiteHeader />

      <main className="pt-24 pb-16 px-4 md:px-8 max-w-7xl mx-auto">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-zinc-500 mb-6">
          <Link href="/" className="hover:text-zinc-300 transition-colors flex items-center gap-1">
            <Home className="w-4 h-4" />
            Home
          </Link>
          <span>/</span>
          <span className="text-zinc-300">Demo</span>
        </div>

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${isLive ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'} border`}>
                <span className={`h-1.5 w-1.5 rounded-full ${isLive ? 'bg-emerald-400' : 'bg-amber-400'} animate-pulse`} />
                <span className={`${isLive ? 'text-emerald-300' : 'text-amber-300'} text-xs font-medium`}>{isLive ? 'Live' : 'Demo'}</span>
              </span>
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-zinc-100 to-zinc-400 bg-clip-text text-transparent">
              Market Dashboard
            </h1>
            <p className="text-zinc-500 text-sm mt-1">
              Click any asset for detailed predictions & analysis
            </p>
          </div>

          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 px-4 py-2.5 bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30 hover:border-sky-500/50 rounded-xl text-sky-300 font-medium text-sm transition-all"
          >
            <Plus className="w-4 h-4" />
            Add Asset
          </button>
        </div>

        {/* Demo Notice */}
        <DemoNotice isLive={isLive} />

        {/* Crypto Section */}
        {cryptoAssets.length > 0 && (
          <section className="mb-8">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="w-4 h-4 text-sky-400" />
              <h2 className="text-lg font-semibold text-zinc-100">Cryptocurrencies</h2>
              <span className="text-xs text-zinc-500">({cryptoAssets.length})</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {cryptoAssets.map((asset) => (
                <AssetCard
                  key={asset.symbol}
                  asset={asset}
                  onRemove={() => handleRemoveAsset(asset.symbol)}
                />
              ))}
            </div>
          </section>
        )}

        {/* Global Indices Section */}
        {indexAssets.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Globe className="w-4 h-4 text-emerald-400" />
              <h2 className="text-lg font-semibold text-zinc-100">Global Indices</h2>
              <span className="text-xs text-zinc-500">({indexAssets.length})</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {indexAssets.map((asset) => (
                <AssetCard
                  key={asset.symbol}
                  asset={asset}
                  onRemove={() => handleRemoveAsset(asset.symbol)}
                />
              ))}
            </div>
          </section>
        )}
      </main>

      <SiteFooter />

      {/* Add Asset Modal */}
      {showAddModal && (
        <AddAssetModal
          availableAssets={disabledAssets}
          onAdd={handleAddAsset}
          onClose={() => setShowAddModal(false)}
        />
      )}
    </div>
  );
}
