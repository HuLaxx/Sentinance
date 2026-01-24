'use client';

import { useEffect, useState } from 'react';

export function LoadingScreen() {
    const [isLoading, setIsLoading] = useState(true);
    const [isSliding, setIsSliding] = useState(false);

    useEffect(() => {
        // Start slide-up animation after content loads
        const loadTimer = setTimeout(() => {
            setIsSliding(true);
        }, 1500);

        // Remove from DOM after animation completes
        const removeTimer = setTimeout(() => {
            setIsLoading(false);
        }, 2300);

        return () => {
            clearTimeout(loadTimer);
            clearTimeout(removeTimer);
        };
    }, []);

    if (!isLoading) return null;

    return (
        <div
            className={`fixed inset-0 z-[100] flex flex-col items-center justify-center bg-zinc-950 transition-transform duration-700 ease-[cubic-bezier(0.76,0,0.24,1)]
        ${isSliding ? '-translate-y-full' : 'translate-y-0'}`}
        >
            {/* Background effects */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-r from-blue-500/10 via-cyan-500/10 to-blue-500/10 rounded-full blur-[150px] animate-pulse" />
                <div className="absolute top-1/4 left-1/4 w-[300px] h-[300px] bg-gradient-to-r from-purple-500/10 to-pink-500/10 rounded-full blur-[100px] animate-pulse" style={{ animationDelay: '0.3s' }} />
                <div className="absolute bottom-1/4 right-1/4 w-[250px] h-[250px] bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-full blur-[100px] animate-pulse" style={{ animationDelay: '0.6s' }} />
            </div>

            {/* Logo and content */}
            <div className="relative z-10 flex flex-col items-center">
                {/* Logo */}
                <div className="relative mb-8">
                    <div className="w-20 h-20 rounded-2xl overflow-hidden shadow-2xl shadow-blue-500/30 animate-pulse">
                        <img src="/icon.svg" alt="Sentinance" className="w-full h-full object-cover" />
                    </div>
                    {/* Glow ring */}
                    <div className="absolute -inset-2 rounded-3xl bg-gradient-to-r from-blue-500 via-cyan-500 to-blue-500 opacity-20 blur-xl animate-pulse" />
                </div>

                {/* Brand name */}
                <h1 className="text-3xl md:text-4xl font-bold mb-4 bg-gradient-to-r from-sky-300 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    Sentinance
                </h1>

                {/* Tagline */}
                <p className="text-zinc-500 text-sm mb-8 tracking-wide">
                    Real-Time Market Intelligence
                </p>

                {/* Loading indicator */}
                <div className="relative">
                    {/* Track */}
                    <div className="w-48 h-1 bg-zinc-800 rounded-full overflow-hidden">
                        {/* Progress bar */}
                        <div
                            className="h-full bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-500 rounded-full transition-all duration-1000 ease-out"
                            style={{
                                width: isSliding ? '100%' : '0%',
                                animation: 'loadProgress 1.5s ease-out forwards'
                            }}
                        />
                    </div>
                </div>

                {/* Pulse dots */}
                <div className="flex gap-2 mt-6">
                    {[0, 1, 2].map((i) => (
                        <div
                            key={i}
                            className="w-2 h-2 rounded-full bg-blue-400/60 animate-pulse"
                            style={{ animationDelay: `${i * 0.2}s` }}
                        />
                    ))}
                </div>
            </div>

            {/* Bottom gradient line */}
            <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />

            {/* Animation keyframes */}
            <style jsx>{`
        @keyframes loadProgress {
          0% { width: 0%; }
          50% { width: 70%; }
          100% { width: 100%; }
        }
      `}</style>
        </div>
    );
}
