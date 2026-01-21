"use client";

import { useState, FormEvent } from "react";
import { Send, User, Bot, Sparkles } from "lucide-react";

type Message = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

export default function Chat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: "user",
            content: input,
        };

        setMessages((prev) => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ messages: [...messages, userMessage] }),
            });

            if (res.ok) {
                const text = await res.text();
                const assistantMessage: Message = {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content: text || "Analysis complete.",
                };
                setMessages((prev) => [...prev, assistantMessage]);
            } else {
                const errorMessage: Message = {
                    id: (Date.now() + 1).toString(),
                    role: "assistant",
                    content: "Unable to process request. Please try again.",
                };
                setMessages((prev) => [...prev, errorMessage]);
            }
        } catch {
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: "assistant",
                content: "Connection error. Please check your backend.",
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-[400px] glass rounded-[var(--radius)] overflow-hidden">
            <div className="p-4 border-b border-zinc-800 flex items-center gap-2 bg-zinc-900/50">
                <Sparkles size={16} className="text-indigo-400" />
                <span className="font-semibold text-sm">Sentinance Copilot</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center text-zinc-500 text-sm mt-16">
                        Ask about market conditions, specific assets, or risk analysis.
                    </div>
                )}

                {messages.map((m) => (
                    <div
                        key={m.id}
                        className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                    >
                        <div
                            className={`h-8 w-8 rounded-full flex items-center justify-center shrink-0 ${m.role === "user" ? "bg-zinc-800" : "bg-indigo-500/20"
                                }`}
                        >
                            {m.role === "user" ? <User size={14} /> : <Bot size={14} className="text-indigo-300" />}
                        </div>

                        <div
                            className={`rounded-2xl px-4 py-2 text-sm max-w-[80%] ${m.role === "user"
                                ? "bg-zinc-100 text-zinc-900 rounded-tr-sm"
                                : "bg-zinc-800/50 border border-zinc-700 rounded-tl-sm text-zinc-200"
                                }`}
                        >
                            {m.content}
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex gap-3">
                        <div className="h-8 w-8 rounded-full bg-indigo-500/20 flex items-center justify-center shrink-0">
                            <Bot size={14} className="text-indigo-300" />
                        </div>
                        <div className="bg-zinc-800/50 border border-zinc-700 rounded-2xl rounded-tl-sm px-4 py-2 flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce" />
                            <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce [animation-delay:75ms]" />
                            <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-bounce [animation-delay:150ms]" />
                        </div>
                    </div>
                )}
            </div>

            <form onSubmit={handleSubmit} className="p-4 border-t border-zinc-800 bg-zinc-900/30">
                <div className="relative">
                    <input
                        className="w-full bg-zinc-800/50 border border-zinc-700 rounded-full pl-4 pr-12 py-2.5 text-sm focus:outline-none focus:border-indigo-500/50 transition placeholder:text-zinc-500"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Analysis on BTC volume..."
                    />
                    <button
                        type="submit"
                        className="absolute right-1.5 top-1.5 h-8 w-8 rounded-full bg-indigo-500 flex items-center justify-center hover:bg-indigo-400 transition disabled:opacity-50"
                        disabled={isLoading || !input.trim()}
                    >
                        <Send size={14} className="text-white" />
                    </button>
                </div>
            </form>
        </div>
    );
}
