'use client';

import { useEffect, useRef, useState } from 'react';

/*
 * Hero signal tape. A synthetic price line streams across the full width of the
 * hero. Every few seconds a spike appears; a thread drops from the moving spike to
 * a provenance card that fills in the path the value took through the system. The
 * feed is illustrative; the path is the one CI runs end to end.
 */

const STEPS = [
    { name: 'Exchange trade', detail: 'Binance aggregate-trade adapter, fixture in CI, live ingest opt-in' },
    { name: 'Kafka', detail: 'Versioned event schema, event time and numbers validated' },
    { name: 'PostgreSQL ledger', detail: 'Idempotent on the upstream event ID' },
    { name: 'Parquet lake', detail: 'Checked partition written for analysis' },
    { name: 'Forecast', detail: 'Ridge AR(1), compared with a last-close baseline' },
    { name: 'Agent note', detail: 'LangGraph retrieve, analyze, finalize, citing a pinned report' },
];

const DX = 6; // px between samples
const TICK_MS = 90;
const SPIKE_EVERY_MS = 7000;

type Sample = { v: number; spike?: number };
type Trace = { id: number; move: string; time: string };

export function SignalTape() {
    const boxRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const lineRef = useRef<SVGPathElement>(null);
    const areaRef = useRef<SVGPathElement>(null);
    const markerRef = useRef<SVGGElement>(null);
    const threadRef = useRef<SVGLineElement>(null);
    const cardRef = useRef<HTMLElement>(null);
    const priceRef = useRef<HTMLSpanElement>(null);
    const [trace, setTrace] = useState<Trace | null>(null);
    const [state, setState] = useState<'idle' | 'tracing'>('idle');

    useEffect(() => {
        const box = boxRef.current;
        const svg = svgRef.current;
        if (!box || !svg) return;
        const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        let w = box.clientWidth;
        let h = box.clientHeight;
        let card = { x: w, y: 0, visible: false };
        const measure = () => {
            w = box.clientWidth;
            h = box.clientHeight;
            svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
            const c = cardRef.current;
            if (c) {
                const br = box.getBoundingClientRect();
                const cr = c.getBoundingClientRect();
                const overlaps = cr.bottom > br.top && cr.top < br.bottom;
                card = { x: cr.left - br.left, y: cr.bottom - br.top, visible: overlaps && window.innerWidth > 980 };
            }
        };
        measure();
        const ro = new ResizeObserver(measure);
        ro.observe(box);

        // Seeded random walk with mean reversion, plus decaying spikes.
        let seed = 42;
        const rand = () => {
            seed = (seed * 16807) % 2147483647;
            return (seed - 1) / 2147483646 - 0.5;
        };
        const base = 64200;
        let level = base;
        let impulse = 0;
        let spikeCount = 0;
        const samples: Sample[] = [];
        const capacity = () => Math.ceil(w / DX) + 4;
        const step = (spike = false): Sample => {
            if (spike) impulse = base * (0.0055 + Math.abs(rand()) * 0.004);
            impulse *= 0.965;
            level += rand() * 70 - (level - base) * 0.02;
            return { v: level + impulse, spike: spike ? ++spikeCount : undefined };
        };
        while (samples.length < capacity()) samples.push(step());

        let lo = Math.min(...samples.map((s) => s.v));
        let hi = Math.max(...samples.map((s) => s.v));
        let offset = 0; // fraction of a tick scrolled
        let last = performance.now();
        let lastSpike = performance.now() - SPIKE_EVERY_MS + 2600;
        let raf = 0;
        let activeSpike = 0;
        let archived = true;

        const yOf = (v: number) => {
            const pad = h * 0.18;
            return h - pad - ((v - lo) / Math.max(1, hi - lo)) * (h - pad * 2);
        };

        const draw = () => {
            let d = '';
            let peak: { x: number; y: number } | null = null;
            samples.forEach((s, i) => {
                const x = (i - offset) * DX - DX * 2;
                const y = yOf(s.v);
                d += `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
                if (s.spike && s.spike === activeSpike) peak = { x, y };
            });
            lineRef.current?.setAttribute('d', d);
            areaRef.current?.setAttribute('d', `${d}L${w + DX} ${h}L${-DX * 2} ${h}Z`);
            if (priceRef.current) priceRef.current.textContent = samples[samples.length - 1].v.toLocaleString('en-GB', { maximumFractionDigits: 2, minimumFractionDigits: 2 });

            const marker = markerRef.current;
            const thread = threadRef.current;
            const p = peak as { x: number; y: number } | null;
            if (p && marker && thread) {
                marker.setAttribute('transform', `translate(${p.x} ${p.y - 4})`);
                marker.setAttribute('data-on', 'true');
                const showThread = card.visible && p.x > card.x - w * 0.55 && !archived;
                thread.setAttribute('x1', String(p.x));
                thread.setAttribute('y1', String(p.y - 14));
                thread.setAttribute('x2', String(Math.max(card.x + 24, Math.min(card.x + 140, p.x))));
                thread.setAttribute('y2', String(card.y));
                thread.style.opacity = showThread ? '1' : '0';
                if (!archived && card.visible && p.x < w * 0.38) {
                    archived = true;
                    setState('idle');
                }
            } else if (marker) {
                marker.setAttribute('data-on', 'false');
                if (thread) thread.style.opacity = '0';
            }
        };

        const tick = (now: number) => {
            const elapsed = now - last;
            last = now;
            offset += elapsed / TICK_MS;
            while (offset >= 1) {
                offset -= 1;
                samples.shift();
                let spike = false;
                if (now - lastSpike > SPIKE_EVERY_MS) {
                    lastSpike = now;
                    spike = true;
                }
                const s = step(spike);
                samples.push(s);
                if (spike && s.spike) {
                    activeSpike = s.spike;
                    archived = false;
                    const prev = samples[samples.length - 2]?.v ?? s.v;
                    const pct = ((s.v - prev) / prev) * 100;
                    const id = s.spike;
                    setState('idle');
                    setTrace({
                        id,
                        move: `${pct >= 0 ? '+' : '−'}${Math.abs(pct).toFixed(1)}%`,
                        time: new Date().toLocaleTimeString('en-GB', { hour12: false }),
                    });
                    window.setTimeout(() => setState('tracing'), 450);
                }
                while (samples.length < capacity()) samples.unshift(step());
            }
            // Ease the y-range so spikes rescale smoothly.
            const vs = samples.map((s) => s.v);
            lo += (Math.min(...vs) - lo) * 0.04;
            hi += (Math.max(...vs) - hi) * 0.04;
            draw();
            raf = requestAnimationFrame(tick);
        };

        if (reduce) {
            samples.splice(samples.length - Math.floor(samples.length * 0.3), 1, step(true));
            activeSpike = spikeCount;
            archived = false;
            lo = Math.min(...samples.map((s) => s.v));
            hi = Math.max(...samples.map((s) => s.v));
            draw();
            const frame = requestAnimationFrame(() => {
                setTrace({ id: activeSpike, move: '+1.9%', time: 'illustrative' });
                setState('tracing');
            });
            return () => {
                cancelAnimationFrame(frame);
                ro.disconnect();
            };
        }

        const io = new IntersectionObserver(([entry]) => {
            cancelAnimationFrame(raf);
            if (entry.isIntersecting) {
                last = performance.now();
                raf = requestAnimationFrame(tick);
            }
        });
        io.observe(box);
        return () => {
            io.disconnect();
            ro.disconnect();
            cancelAnimationFrame(raf);
        };
    }, []);

    return (
        <div className="sx-tape">
            <div className="sx-tape__canvas" ref={boxRef}>
                <svg ref={svgRef} aria-hidden="true" preserveAspectRatio="none">
                    <defs>
                        <linearGradient id="sx-line" x1="0" x2="1" y1="0" y2="0">
                            <stop offset="0" stopColor="#6366f1" stopOpacity="0.2" />
                            <stop offset="0.5" stopColor="#38bdf8" />
                            <stop offset="1" stopColor="#22d3ee" />
                        </linearGradient>
                        <linearGradient id="sx-area" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0" stopColor="#38bdf8" stopOpacity="0.18" />
                            <stop offset="1" stopColor="#38bdf8" stopOpacity="0" />
                        </linearGradient>
                        <filter id="sx-glow" x="-10%" y="-50%" width="120%" height="200%">
                            <feGaussianBlur stdDeviation="4" result="b" />
                            <feMerge>
                                <feMergeNode in="b" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>
                    <path ref={areaRef} fill="url(#sx-area)" />
                    <path ref={lineRef} fill="none" stroke="url(#sx-line)" strokeWidth="2" filter="url(#sx-glow)" strokeLinejoin="round" />
                    <line ref={threadRef} className="sx-thread" style={{ opacity: 0, transition: 'opacity 500ms ease' }} />
                    <g ref={markerRef} className="sx-marker" data-on="false">
                        <circle r="16" fill="rgba(251,191,36,0.12)" />
                        <circle r="5" fill="#fbbf24" />
                    </g>
                </svg>
                <div className="sx-tape__fade" />
                <p className="sx-tape__label">
                    Illustrative feed, BTCUSDT
                    <span className="sx-tape__price" ref={priceRef} />
                </p>
            </div>
            <aside
                className="sx-trace"
                ref={cardRef}
                data-state={trace ? state : 'idle'}
                aria-live="polite"
                aria-label="Where this price move came from"
            >
                <div className="sx-trace__head">
                    <p className="sx-trace__title">{trace ? 'Tracing the spike' : 'Waiting for a move'}</p>
                    {trace ? (
                        <span className="sx-trace__move">
                            {trace.move} at {trace.time}
                        </span>
                    ) : null}
                </div>
                <ol>
                    {STEPS.map((s, i) => (
                        <li key={s.name} style={{ ['--sx-i' as string]: i }}>
                            <strong>{s.name}</strong>
                            <span>{s.detail}</span>
                        </li>
                    ))}
                </ol>
                <p className="sx-trace__foot">Synthetic feed. The path is the one CI runs end to end.</p>
            </aside>
        </div>
    );
}
