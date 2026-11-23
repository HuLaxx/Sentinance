'use client';

import { useEffect, useState } from 'react';

/*
 * First-load sequence: a market trace draws the Sentinance pulse, the logo tile
 * forms around it, the wordmark rises with the HuLaX credit, then the curtain
 * lifts. CSS-timed so it covers the first paint even before hydration; client
 * navigation never replays it because the root layout persists.
 */

// The pulse from the app icon (512 grid), drawn left to right.
const PULSE = 'M42 256H170L234 384L320 128L362 256H469';
const TOTAL_MS = 2650;

export function LoadingScreen() {
  const [mounted, setMounted] = useState(true);

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const timer = window.setTimeout(() => {
      document.documentElement.dataset.sxLoaded = 'true';
      setMounted(false);
    }, reduce ? 750 : TOTAL_MS);
    return () => window.clearTimeout(timer);
  }, []);

  if (!mounted) return null;

  return (
    <div className="sx-loader" role="status" aria-label="Loading Sentinance">
      <div className="sx-loader__glow" aria-hidden="true" />
      <div className="sx-loader__center">
        <svg className="sx-loader__mark" viewBox="0 0 512 512" aria-hidden="true">
          <defs>
            <linearGradient id="sx-loader-tile" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stopColor="#6366f1" />
              <stop offset="1" stopColor="#06b6d4" />
            </linearGradient>
            <linearGradient id="sx-loader-trace" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stopColor="#6366f1" />
              <stop offset="0.55" stopColor="#38bdf8" />
              <stop offset="1" stopColor="#22d3ee" />
            </linearGradient>
          </defs>
          <rect className="sx-loader__tile" width="512" height="512" rx="128" fill="url(#sx-loader-tile)" />
          <path className="sx-loader__trace" d={PULSE} pathLength={1} />
          <path className="sx-loader__ink" d={PULSE} />
          <circle className="sx-loader__head" r="18" />
        </svg>
        <p className="sx-loader__word">Sentinance</p>
        <p className="sx-loader__line">Real-time market intelligence</p>
      </div>
      <p className="sx-loader__credit" aria-hidden="true">
        <span className="hulax-credit-label">A Project By</span> <span className="hulax-credit hulax-credit-gradient">HuLaX</span>
      </p>
      <div className="sx-loader__edge" aria-hidden="true" />
    </div>
  );
}
