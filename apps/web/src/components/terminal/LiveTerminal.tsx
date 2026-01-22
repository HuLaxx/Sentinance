"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, MessageSquare, Sparkles, TrendingDown, TrendingUp } from "lucide-react";
import { z } from "zod";
import { clamp, ema, mulberry32 } from "../../lib/math";
import Chat from "./Chat";
import AssetDetailModal from "../AssetDetailModal";

// Zod Schemas
const PriceSchema = z.object({
  symbol: z.string(),
  price: z.number(),
  priceChangePercent: z.number().optional(),
  volume: z.number().optional()
});

const PriceUpdateSchema = z.object({
  type: z.enum(["price_update", "initial"]),
  prices: z.array(PriceSchema)
});

const AlertSchema = z.object({
  type: z.literal("alert_triggered"),
  alert: z.object({
    id: z.string(),
    symbol: z.string(),
    message: z.string()
  })
});

const WSMessageSchema = z.union([PriceUpdateSchema, AlertSchema, z.object({ type: z.literal("pong") })]);

type MarketPoint = { t: number; p: number; v: number };

type Ticker = {
  btc: number;
  eth: number;
  sol: number;
  xrp: number;
  sp500: number;
  nifty: number;
  ftse: number;
  nikkei: number;
  latency: number;
};

const useMarketData = () => {
  const [data, setData] = useState<MarketPoint[]>(() => {
    // Initial data to populate chart before live feed starts
    const r = mulberry32(42);
    let p = 64250;
    const out: MarketPoint[] = [];
    for (let i = 0; i < 60; i += 1) {
      p = ema(p, p + (r() - 0.5) * 120, 0.35);
      out.push({ t: i, p, v: 20 + r() * 80 });
    }
    return out;
  });

  const [ticker, setTicker] = useState<Ticker>({
    btc: 90000,
    eth: 3000,
    sol: 130,
    xrp: 2.0,
    sp500: 6800,
    nifty: 25000,
    ftse: 10100,
    nikkei: 52000,
    latency: 12,
  });

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/prices';
    let ws: WebSocket | null = null;
    let mockInterval: NodeJS.Timeout | null = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 3;

    const startMockDataFeed = () => {
      console.log("📊 Starting simulated data feed (API server unavailable)");
      const r = mulberry32(Date.now());

      mockInterval = setInterval(() => {
        setData((prev) => {
          const last = prev[prev.length - 1];
          const drift = (r() - 0.5) * 80;
          const newPrice = ema(last.p, last.p + drift, 0.35);
          const newPoint = {
            t: last.t + 1,
            p: newPrice,
            v: 20 + r() * 80
          };
          const newData = [...prev, newPoint];
          if (newData.length > 60) newData.shift();
          return newData;
        });

        setTicker((prev) => ({
          btc: prev.btc + (r() - 0.5) * 50,
          eth: prev.eth + (r() - 0.5) * 8,
          sol: prev.sol + (r() - 0.5) * 2,
          latency: Math.floor(10 + r() * 15),
        }));
      }, 2000);
    };

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
          try {
            const raw = JSON.parse(event.data);
            const result = WSMessageSchema.safeParse(raw);

            if (!result.success) {
              // Ignore invalid messages (ping/pong or malformed)
              return;
            }

            const msg = result.data;

            if (msg.type === "price_update" || msg.type === "initial") {
              const btcData = msg.prices.find(p => p.symbol === "BTCUSDT");
              const ethData = msg.prices.find(p => p.symbol === "ETHUSDT");
              const solData = msg.prices.find(p => p.symbol === "SOLUSDT");
              const xrpData = msg.prices.find(p => p.symbol === "XRPUSDT");
              const sp500Data = msg.prices.find(p => p.symbol === "^GSPC");
              const niftyData = msg.prices.find(p => p.symbol === "^NSEI");
              const ftseData = msg.prices.find(p => p.symbol === "^FTSE");
              const nikkeiData = msg.prices.find(p => p.symbol === "^N225");

              if (btcData) {
                setData((prev) => {
                  const last = prev[prev.length - 1];
                  const newPoint = {
                    t: last.t + 1,
                    p: btcData.price,
                    v: btcData.volume || 20 + Math.random() * 80
                  };
                  const newData = [...prev, newPoint];
                  if (newData.length > 60) newData.shift();
                  return newData;
                });
              }

              setTicker((prev) => ({
                btc: btcData?.price || prev.btc,
                eth: ethData?.price || prev.eth,
                sol: solData?.price || prev.sol,
                xrp: xrpData?.price || prev.xrp,
                sp500: sp500Data?.price || prev.sp500,
                nifty: niftyData?.price || prev.nifty,
                ftse: ftseData?.price || prev.ftse,
                nikkei: nikkeiData?.price || prev.nikkei,
                latency: 15,
              }));
            }
          } catch (e) {
            console.error("WS Parse Error", e);
          }
        };

        ws.onopen = () => {
          console.log("✅ Connected to live price feed!");
          reconnectAttempts = 0;
          // Clear mock interval if it was running
          if (mockInterval) {
            clearInterval(mockInterval);
            mockInterval = null;
          }
        };

        ws.onerror = () => {
          // WebSocket error events don't contain useful info in browsers
          console.warn(`⚠️ WebSocket connection issue (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts}) - URL: ${wsUrl}`);
        };

        ws.onclose = (event) => {
          if (event.wasClean) {
            console.log("WebSocket closed cleanly");
          } else {
            reconnectAttempts++;
            if (reconnectAttempts < maxReconnectAttempts) {
              console.log(`🔄 Reconnecting in 3s... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
              setTimeout(connect, 3000);
            } else {
              console.warn("❌ WebSocket connection failed after max attempts. Using simulated data.");
              startMockDataFeed();
            }
          }
        };
      } catch (e) {
        console.error("Failed to create WebSocket:", e);
        startMockDataFeed();
      }
    };

    connect();

    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
      if (mockInterval) clearInterval(mockInterval);
    };
  }, []);

  return { data, ticker };
};

const assistantMessages = [
  { role: "assistant", text: "Morning brief is ready. Liquidity gaps widened 6% overnight." },
  { role: "user", text: "Any manipulation risk on BTC?" },
  { role: "assistant", text: "Low risk. Whale outflows cooled and order book depth normalized." },
];

const sparklinePath = (points: number[], width = 100, height = 32) => {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;

  return points
    .map((point, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - ((point - min) / range) * height;
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
};

export default function LiveTerminal() {
  const { data, ticker } = useMarketData();
  const [selectedAsset, setSelectedAsset] = useState<{
    symbol: string;
    name: string;
    price: number;
    assetType: "crypto" | "index";
  } | null>(null);

  const { min, max, pathD, fillD } = useMemo(() => {
    const prices = data.map((d) => d.p);
    const lo = Math.min(...prices);
    const hi = Math.max(...prices);
    const pad = (hi - lo) * 0.12 || 1;
    const min = lo - pad;
    const max = hi + pad;

    const W = 100;
    const H = 100;

    const pts = data.map((d, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H - ((d.p - min) / (max - min)) * H;
      return [x, y] as const;
    });

    const d0 = `M ${pts[0][0]} ${pts[0][1]} ` + pts.slice(1).map(([x, y]) => `L ${x} ${y}`).join(" ");
    const df = `${d0} L ${W} ${H} L 0 ${H} Z`;

    return { min, max, pathD: d0, fillD: df };
  }, [data]);

  const volumeB = useMemo(() => {
    const avg = data.reduce((sum, point) => sum + point.v, 0) / data.length;
    const scaled = 32 + avg / 5;
    return scaled.toFixed(1);
  }, [data]);

  const orderBook = useMemo(() => {
    const r = mulberry32(99);
    const base = ticker.btc || 64200;
    return Array.from({ length: 22 }, (_, i) => {
      const side = i % 2 === 0 ? "BID" : "ASK";
      const px = base + (i - 11) * 12 + (r() - 0.5) * 8;
      const amt = 0.04 + r() * 0.22;
      return { id: i, side, px, amt };
    });
  }, [ticker.btc]);

  const bars = useMemo(() => {
    const r = mulberry32(7);
    return Array.from({ length: 18 }, (_, i) => {
      const base = 0.25 + i / 28;
      const height = clamp(base + r() * 0.3, 0.22, 0.9);
      return { i, height };
    });
  }, []);

  const signalCards = useMemo(() => {
    const build = (seed: number, label: string, value: string, delta: string, tone: "up" | "down") => {
      const r = mulberry32(seed);
      const points = Array.from({ length: 18 }, () => 0.3 + r() * 0.6);
      return {
        label,
        value,
        delta,
        tone,
        path: sparklinePath(points),
      };
    };

    return [
      build(11, "Liquidity", "Stable", "+2.4%", "up"),
      build(22, "Momentum", "Neutral", "-0.8%", "down"),
      build(33, "On-chain flow", "Positive", "+1.6%", "up"),
    ];
  }, []);

  const forecast = useMemo(() => {
    const last = data[data.length - 1]?.p ?? 0;
    const prev = data[data.length - 2]?.p ?? last;
    const drift = last - prev;
    const forecast30m = last + drift * 6;
    const forecast2h = last + drift * 24;
    const volatility = Math.max(0, Math.min(1, Math.abs(drift) / (last || 1))) * 100;
    const confidence = Math.max(68, Math.min(92, 92 - volatility));

    return {
      forecast30m,
      forecast2h,
      confidence: Math.round(confidence),
      trendUp: forecast30m >= last,
    };
  }, [data]);

  return (
    <div className="pt-28 pb-10 px-4 md:px-8 min-h-screen max-w-[1600px] mx-auto flex flex-col gap-6">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="h-2 w-2 rounded-full bg-sky-400 animate-pulse" />
            <span className="text-sky-300 text-xs font-bold uppercase tracking-wider">
              Live Feed - {ticker.latency}ms
            </span>
          </div>
          <h2 className="font-display text-3xl font-bold tracking-tight bg-gradient-to-r from-zinc-100 to-zinc-300 bg-clip-text text-transparent">Trading Terminal</h2>
          <p className="text-zinc-400 text-sm mt-2">
            Real-time market data streaming via WebSocket. Monitor prices, signals, and AI analysis.
          </p>
        </div>

        <div className="flex gap-3 overflow-x-auto w-full md:w-auto pb-2 md:pb-0">
          {[
            { l: "BTC", v: ticker.btc, prefix: "$", symbol: "BTCUSDT", name: "Bitcoin" },
            { l: "ETH", v: ticker.eth, prefix: "$", symbol: "ETHUSDT", name: "Ethereum" },
            { l: "SOL", v: ticker.sol, prefix: "$", symbol: "SOLUSDT", name: "Solana" },
            { l: "XRP", v: ticker.xrp, prefix: "$", symbol: "XRPUSDT", name: "XRP" },
          ].map((t) => (
            <div
              key={t.l}
              onClick={() => setSelectedAsset({ symbol: t.symbol, name: t.name, price: t.v, assetType: "crypto" })}
              className="glass rounded-2xl px-5 py-3 min-w-[140px] cursor-pointer hover:bg-white/10 hover:border-indigo-500/50 border border-transparent transition-all"
            >
              <div className="text-[10px] text-slate-500 font-mono mb-1 flex justify-between">
                <span>{t.l}</span>
                <span className="text-white/50">CRYPTO</span>
              </div>
              <div className="text-lg font-bold font-mono text-white">{t.prefix}{t.v.toFixed(2)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Market Indices Row */}
      <div className="flex gap-3 overflow-x-auto pb-2">
        {[
          { l: "S&P 500", v: ticker.sp500, symbol: "^GSPC", name: "S&P 500" },
          { l: "NIFTY 50", v: ticker.nifty, symbol: "^NSEI", name: "Nifty 50" },
          { l: "FTSE 100", v: ticker.ftse, symbol: "^FTSE", name: "FTSE 100" },
          { l: "NIKKEI", v: ticker.nikkei, symbol: "^N225", name: "Nikkei 225" },
        ].map((t) => (
          <div
            key={t.l}
            onClick={() => setSelectedAsset({ symbol: t.symbol, name: t.name, price: t.v, assetType: "index" })}
            className="glass rounded-2xl px-5 py-3 min-w-[160px] border border-emerald-500/20 cursor-pointer hover:bg-emerald-500/10 hover:border-emerald-500/50 transition-all"
          >
            <div className="text-[10px] text-emerald-400 font-mono mb-1 flex justify-between">
              <span>{t.l}</span>
              <span className="text-white/50">INDEX</span>
            </div>
            <div className="text-lg font-bold font-mono text-white">{t.v.toLocaleString()}</div>
          </div>
        ))}
      </div>

      {/* Asset Detail Modal */}
      {selectedAsset && (
        <AssetDetailModal
          symbol={selectedAsset.symbol}
          name={selectedAsset.name}
          price={selectedAsset.price}
          assetType={selectedAsset.assetType}
          onClose={() => setSelectedAsset(null)}
        />
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {signalCards.map((card) => (
          <div key={card.label} className="glass rounded-2xl px-5 py-4 flex items-center justify-between">
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono">{card.label}</div>
              <div className="mt-2 text-lg font-semibold">{card.value}</div>
              <div className={`mt-1 text-xs ${card.tone === "up" ? "text-emerald-300" : "text-rose-300"}`}>
                {card.delta}
              </div>
            </div>
            <svg viewBox="0 0 100 32" className="h-10 w-24">
              <path
                d={card.path}
                fill="none"
                stroke={card.tone === "up" ? "#34d399" : "#fb7185"}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </div>
        ))}
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-0">
        <div className="lg:col-span-9 glass rounded-[var(--radius)] p-1 overflow-hidden min-h-[420px]">
          <div className="relative h-full rounded-[22px] bg-[#070707] overflow-hidden">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px)",
                backgroundSize: "44px 44px",
              }}
            />

            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
              <defs>
                <linearGradient id="f" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                </linearGradient>
              </defs>

              <path d={pathD} fill="none" stroke="#3b82f6" strokeWidth="0.9" vectorEffect="non-scaling-stroke" />
              <path d={fillD} fill="url(#f)" />
            </svg>

            <div className="absolute top-6 left-6">
              <div className="text-4xl md:text-5xl font-mono font-bold tracking-tighter text-white">
                ${ticker.btc.toFixed(2)}
              </div>
              <div className="mt-2 flex gap-6 text-xs font-mono text-slate-500">
                <span>
                  RANGE: <span className="text-white/90">{min.toFixed(0)}-{max.toFixed(0)}</span>
                </span>
                <span>
                  VOL: <span className="text-white/90">{volumeB}B</span>
                </span>
              </div>
            </div>

            <div className="absolute bottom-6 right-6 hidden md:flex items-end gap-1">
              {bars.map((bar) => (
                <div key={bar.i} className="w-2 rounded-full bg-white/10" style={{ height: `${bar.height * 60}px` }} />
              ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 flex flex-col gap-6 min-h-0">
          <Chat />
          <div className="glass rounded-[var(--radius)] p-5 min-h-[260px]">
            <div className="flex items-center justify-between mb-4">
              <div className="text-xs font-bold uppercase tracking-widest text-slate-500">Order Flow</div>
              <Activity size={16} className="text-white/60" />
            </div>

            <div className="h-full overflow-auto pr-1 font-mono text-[11px] space-y-1">
              {orderBook.map((row) => (
                <div
                  key={row.id}
                  className="flex items-center justify-between rounded-lg px-2 py-1 hover:bg-white/5 transition"
                >
                  <span className={row.side === "BID" ? "text-emerald-300" : "text-rose-300"}>{row.side}</span>
                  <span className="text-white/85">{row.amt.toFixed(4)}</span>
                  <span className="text-slate-500">{row.px.toFixed(0)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass rounded-[var(--radius)] p-6 relative overflow-hidden border-blue-500/20">
            <div className="absolute -bottom-24 -right-20 h-56 w-56 rounded-full bg-blue-900/20 blur-3xl" />
            <div className="relative z-10">
              <div className="flex items-center gap-2 text-[10px] font-bold text-sky-200 uppercase tracking-widest mb-2">
                <Sparkles size={12} />
                AI Prediction
              </div>
              <div className="flex items-center gap-2">
                <div className="font-display text-3xl font-bold">
                  {forecast.trendUp ? "Upside" : "Caution"}
                </div>
                {forecast.trendUp ? (
                  <TrendingUp size={18} className="text-emerald-300" />
                ) : (
                  <TrendingDown size={18} className="text-rose-300" />
                )}
              </div>
              <div className="mt-6">
                <div className="flex justify-between text-xs text-sky-200/90 font-medium">
                  <span>Confidence</span>
                  <span>{forecast.confidence}%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-black/25 overflow-hidden border border-white/10">
                  <div className="h-full bg-sky-500" style={{ width: `${forecast.confidence}%` }} />
                </div>
                <div className="mt-4 text-xs text-slate-400/90 space-y-1">
                  <div className="flex items-center justify-between">
                    <span>Forecast (30m)</span>
                    <span className="text-white/90">${forecast.forecast30m.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Forecast (2h)</span>
                    <span className="text-white/90">${forecast.forecast2h.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="glass rounded-[var(--radius)] p-6 flex flex-col gap-4 min-h-[260px]">
            <div className="flex items-center justify-between">
              <div className="text-xs font-bold uppercase tracking-widest text-slate-500">Operator Assistant</div>
              <MessageSquare size={16} className="text-white/60" />
            </div>
            <div className="flex-1 space-y-3 text-xs text-slate-300/90">
              {assistantMessages.map((msg, index) => (
                <div
                  key={`${msg.role}-${index}`}
                  className={
                    "rounded-2xl px-3 py-2 leading-relaxed " +
                    (msg.role === "assistant" ? "bg-white/5 border border-white/10" : "bg-blue-900/20 border border-blue-500/30")
                  }
                >
                  <span className="block text-[10px] uppercase tracking-widest text-slate-500 mb-1">
                    {msg.role === "assistant" ? "Sentinance AI" : "Operator"}
                  </span>
                  {msg.text}
                </div>
              ))}
            </div>
            <form
              onSubmit={(event) => event.preventDefault()}
              className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.02] px-3 py-2"
            >
              <input
                type="text"
                placeholder="Ask for a risk summary..."
                className="flex-1 bg-transparent text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none"
              />
              <button
                type="submit"
                className="text-[10px] uppercase tracking-widest text-sky-200 hover:text-white transition"
              >
                Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
