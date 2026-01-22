"use client";

import { useState, useEffect, FormEvent } from "react";
import {
    X,
    TrendingUp,
    TrendingDown,
    Activity,
    BarChart3,
    Send,
    Bot,
    User,
    Loader2,
} from "lucide-react";

type AssetDetailModalProps = {
    symbol: string;
    name: string;
    price: number;
    assetType: "crypto" | "index";
    onClose: () => void;
};

type Indicators = {
    symbol: string;
    rsi_14: number | null;
    macd: { value: number; signal: number; histogram: number } | null;
    moving_averages: {
        sma_20: number | null;
        ema_12: number | null;
    } | null;
    bollinger_bands: {
        upper: number | null;
        middle: number | null;
        lower: number | null;
    } | null;
};

type Prediction = {
    symbol: string;
    current_price: number;
    predicted_price: number;
    predicted_change_percent: number;
    direction: "up" | "down" | "neutral";
    confidence: number;
    horizon: string;
};

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

export default function AssetDetailModal({
    symbol,
    name,
    price,
    assetType,
    onClose,
}: AssetDetailModalProps) {
    const [indicators, setIndicators] = useState<Indicators | null>(null);
    const [predictions, setPredictions] = useState<Record<string, Prediction>>({});
    const [loading, setLoading] = useState(true);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [chatLoading, setChatLoading] = useState(false);

    // Fetch indicators and predictions
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                // Fetch indicators
                const indRes = await fetch(`/api/indicators/${symbol}`);
                if (indRes.ok) {
                    const indData = await indRes.json();
                    setIndicators(indData);
                }

                // Fetch predictions for different horizons
                const horizons = ["1h", "24h", "7d"];
                for (const h of horizons) {
                    const predRes = await fetch(`/api/predict/${symbol}?horizon=${h}`);
                    if (predRes.ok) {
                        const predData = await predRes.json();
                        setPredictions((prev) => ({ ...prev, [h]: predData }));
                    }
                }
            } catch (e) {
                console.error("Failed to fetch asset data:", e);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [symbol]);

    // Handle chat submission
    const handleChatSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!input.trim() || chatLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: input,
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setChatLoading(true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [
                        {
                            role: "system",
                            content: `You are analyzing ${name} (${symbol}). Current price: $${price.toLocaleString()}. RSI: ${indicators?.rsi_14 || "N/A"}. Provide concise, actionable insights.`,
                        },
                        ...messages,
                        userMessage,
                    ],
                }),
            });

            if (res.ok) {
                const text = await res.text();
                setMessages((prev) => [
                    ...prev,
                    { id: (Date.now() + 1).toString(), role: "assistant", content: text },
                ]);
            }
        } catch {
            setMessages((prev) => [
                ...prev,
                { id: (Date.now() + 1).toString(), role: "assistant", content: "Connection error." },
            ]);
        } finally {
            setChatLoading(false);
        }
    };

    const getRsiColor = (rsi: number) => {
        if (rsi >= 70) return "text-rose-400";
        if (rsi <= 30) return "text-emerald-400";
        return "text-zinc-300";
    };

    const getRsiLabel = (rsi: number) => {
        if (rsi >= 70) return "Overbought";
        if (rsi <= 30) return "Oversold";
        return "Neutral";
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

            {/* Modal */}
            <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl">
                {/* Header */}
                <div className="sticky top-0 z-10 flex items-center justify-between p-6 border-b border-zinc-800 bg-zinc-900/95 backdrop-blur">
                    <div>
                        <div className="flex items-center gap-3">
                            <span
                                className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded ${assetType === "crypto" ? "bg-indigo-500/20 text-indigo-300" : "bg-emerald-500/20 text-emerald-300"
                                    }`}
                            >
                                {assetType}
                            </span>
                            <h2 className="text-2xl font-bold">{name}</h2>
                            <span className="text-zinc-500 font-mono text-sm">{symbol}</span>
                        </div>
                        <div className="mt-2 text-3xl font-mono font-bold">
                            ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-zinc-800 rounded-full transition"
                    >
                        <X size={24} />
                    </button>
                </div>

                {loading ? (
                    <div className="flex items-center justify-center py-20">
                        <Loader2 className="animate-spin text-zinc-500" size={32} />
                    </div>
                ) : (
                    <div className="p-6 space-y-6">
                        {/* Technical Indicators */}
                        <div>
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Activity size={18} className="text-indigo-400" />
                                Technical Indicators
                            </h3>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {/* RSI */}
                                <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
                                    <div className="text-xs text-zinc-500 uppercase mb-1">RSI (14)</div>
                                    <div className={`text-2xl font-bold font-mono ${getRsiColor(indicators?.rsi_14 || 50)}`}>
                                        {indicators?.rsi_14?.toFixed(1) || "—"}
                                    </div>
                                    <div className="text-xs text-zinc-400 mt-1">
                                        {indicators?.rsi_14 ? getRsiLabel(indicators.rsi_14) : "—"}
                                    </div>
                                </div>

                                {/* MACD */}
                                <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
                                    <div className="text-xs text-zinc-500 uppercase mb-1">MACD</div>
                                    <div className="text-2xl font-bold font-mono">
                                        {indicators?.macd?.value?.toFixed(2) || "—"}
                                    </div>
                                    <div className="text-xs text-zinc-400 mt-1">
                                        Signal: {indicators?.macd?.signal?.toFixed(2) || "—"}
                                    </div>
                                </div>

                                {/* SMA */}
                                <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
                                    <div className="text-xs text-zinc-500 uppercase mb-1">SMA (20)</div>
                                    <div className="text-2xl font-bold font-mono">
                                        {indicators?.moving_averages?.sma_20?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || "—"}
                                    </div>
                                </div>

                                {/* Bollinger */}
                                <div className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50">
                                    <div className="text-xs text-zinc-500 uppercase mb-1">Bollinger</div>
                                    <div className="text-sm font-mono space-y-0.5">
                                        <div className="text-emerald-400">
                                            U: {indicators?.bollinger_bands?.upper?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || "—"}
                                        </div>
                                        <div className="text-zinc-300">
                                            M: {indicators?.bollinger_bands?.middle?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || "—"}
                                        </div>
                                        <div className="text-rose-400">
                                            L: {indicators?.bollinger_bands?.lower?.toLocaleString(undefined, { maximumFractionDigits: 0 }) || "—"}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Price Predictions */}
                        <div>
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <BarChart3 size={18} className="text-sky-400" />
                                AI Price Predictions
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {["1h", "24h", "7d"].map((horizon) => {
                                    const pred = predictions[horizon];
                                    const isUp = pred?.direction === "up";
                                    return (
                                        <div
                                            key={horizon}
                                            className={`rounded-xl p-4 border ${isUp ? "bg-emerald-500/10 border-emerald-500/30" : "bg-rose-500/10 border-rose-500/30"
                                                }`}
                                        >
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="text-sm font-medium text-zinc-300">{horizon} Forecast</span>
                                                {isUp ? (
                                                    <TrendingUp size={16} className="text-emerald-400" />
                                                ) : (
                                                    <TrendingDown size={16} className="text-rose-400" />
                                                )}
                                            </div>
                                            <div className="text-xl font-bold font-mono">
                                                ${pred?.predicted_price?.toLocaleString(undefined, { maximumFractionDigits: 2 }) || "—"}
                                            </div>
                                            <div className={`text-sm font-mono ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
                                                {pred?.predicted_change_percent !== undefined
                                                    ? `${isUp ? "+" : ""}${pred.predicted_change_percent.toFixed(2)}%`
                                                    : "—"}
                                            </div>
                                            <div className="mt-2 text-xs text-zinc-500">
                                                Confidence: {pred?.confidence ? `${(pred.confidence * 100).toFixed(0)}%` : "—"}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* AI Chat */}
                        <div>
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <Bot size={18} className="text-purple-400" />
                                Ask About {name}
                            </h3>
                            <div className="bg-zinc-800/30 rounded-xl border border-zinc-700/50 overflow-hidden">
                                <div className="h-48 overflow-y-auto p-4 space-y-3">
                                    {messages.length === 0 && (
                                        <div className="text-center text-zinc-500 text-sm py-8">
                                            Ask any question about {name}'s price action, technicals, or outlook.
                                        </div>
                                    )}
                                    {messages.map((m) => (
                                        <div
                                            key={m.id}
                                            className={`flex gap-2 ${m.role === "user" ? "flex-row-reverse" : ""}`}
                                        >
                                            <div
                                                className={`h-6 w-6 rounded-full flex items-center justify-center shrink-0 ${m.role === "user" ? "bg-zinc-700" : "bg-purple-500/20"
                                                    }`}
                                            >
                                                {m.role === "user" ? <User size={12} /> : <Bot size={12} className="text-purple-300" />}
                                            </div>
                                            <div
                                                className={`rounded-xl px-3 py-2 text-sm max-w-[80%] ${m.role === "user"
                                                        ? "bg-zinc-100 text-zinc-900"
                                                        : "bg-zinc-800 border border-zinc-700 text-zinc-200"
                                                    }`}
                                            >
                                                {m.content}
                                            </div>
                                        </div>
                                    ))}
                                    {chatLoading && (
                                        <div className="flex gap-2">
                                            <div className="h-6 w-6 rounded-full bg-purple-500/20 flex items-center justify-center">
                                                <Bot size={12} className="text-purple-300" />
                                            </div>
                                            <div className="bg-zinc-800 border border-zinc-700 rounded-xl px-3 py-2 flex gap-1">
                                                <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce" />
                                                <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce [animation-delay:75ms]" />
                                                <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce [animation-delay:150ms]" />
                                            </div>
                                        </div>
                                    )}
                                </div>
                                <form onSubmit={handleChatSubmit} className="border-t border-zinc-700 p-3">
                                    <div className="relative">
                                        <input
                                            className="w-full bg-zinc-800/50 border border-zinc-600 rounded-full pl-4 pr-12 py-2 text-sm focus:outline-none focus:border-purple-500/50"
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            placeholder={`Ask about ${name}...`}
                                        />
                                        <button
                                            type="submit"
                                            className="absolute right-1.5 top-1.5 h-7 w-7 rounded-full bg-purple-500 flex items-center justify-center hover:bg-purple-400 transition disabled:opacity-50"
                                            disabled={chatLoading || !input.trim()}
                                        >
                                            <Send size={12} className="text-white" />
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
