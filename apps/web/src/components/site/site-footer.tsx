'use client';

import Link from 'next/link';
import { Github, Linkedin, Mail } from 'lucide-react';

const pageLinks = [
    { label: 'Home', href: '/' },
    { label: 'Demo', href: '/demo' },
    { label: 'Architecture', href: '/architecture' },
    { label: 'Tech Dash', href: '/techdash' },
];

const connectLinks = [
    { icon: Github, label: 'GitHub', href: 'https://github.com/HuLaxx/Sentinance' },
    { icon: Linkedin, label: 'LinkedIn', href: 'https://linkedin.com/in/rahul-khanke' },
    { icon: Mail, label: 'Gmail', href: 'mailto:rahulkhanke02@gmail.com' },
];

export function SiteFooter() {
    return (
        <footer className="relative mx-6 mb-6 rounded-2xl overflow-hidden mt-20 backdrop-blur-xl" suppressHydrationWarning>
            {/* Simple transparent glass effect */}
            <div className="absolute inset-0 bg-zinc-900/60" />

            {/* Glass border */}
            <div className="absolute inset-0 rounded-2xl border border-white/10" />

            <div className="relative px-8 md:px-16 py-16">
                <div className="max-w-7xl mx-auto">
                    {/* Main footer content */}
                    <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_auto] gap-16 md:gap-16 lg:gap-24">
                        {/* Brand Section */}
                        <div className="space-y-5">
                            <Link href="/" className="flex items-center gap-3 group">
                                <div className="w-11 h-11 rounded-xl overflow-hidden shadow-lg shadow-blue-500/20 
                                    group-hover:shadow-blue-500/40 transition-all duration-300 group-hover:scale-105">
                                    <img src="/icon.svg" alt="Sentinance" className="w-full h-full object-cover" />
                                </div>
                                <span className="text-xl font-bold bg-gradient-to-r from-zinc-100 to-zinc-300 
                                    bg-clip-text text-transparent group-hover:from-sky-300 group-hover:to-blue-400 transition-all duration-300">
                                    Sentinance
                                </span>
                            </Link>
                            <p className="text-sm text-zinc-400 leading-relaxed max-w-xs">
                                Real-time crypto market intelligence with agentic AI,
                                ML predictions, and enterprise-grade analytics.
                            </p>
                        </div>

                        {/* Pages Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-zinc-100 uppercase tracking-wider mb-5">
                                Pages
                            </h3>
                            <ul className="space-y-3">
                                {pageLinks.map((link) => (
                                    <li key={link.href}>
                                        <Link
                                            href={link.href}
                                            className="text-sm text-zinc-400 hover:text-sky-400 
                                                transition-all duration-200 hover:translate-x-1 inline-block">
                                            {link.label}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Connect Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-zinc-100 uppercase tracking-wider mb-5">
                                Connect
                            </h3>
                            <ul className="space-y-3">
                                {connectLinks.map((link) => (
                                    <li key={link.label}>
                                        <a
                                            href={link.href}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center gap-2 text-sm text-zinc-400 
                                                hover:text-sky-400 transition-all duration-200 group">
                                            <link.icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                                            {link.label}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Bottom Section */}
                <div className="mt-16 pt-8 border-t border-zinc-800/50 flex flex-col md:flex-row 
                        items-center justify-between gap-4">
                    <p className="text-sm text-zinc-500">
                        © {new Date().getFullYear()} Sentinance. All rights reserved.
                    </p>
                    <a
                        href="https://hulax.vercel.app"
                        target="_blank"
                        rel="noreferrer"
                        className="group inline-flex items-baseline gap-1.5 text-white/70">
                        <span className="hulax-credit-label text-xs text-white/60 transition-all group-hover:text-white/70 group-hover:drop-shadow-[0_0_8px_rgba(56,189,248,0.3)]">
                            A Project By
                        </span>
                        <span className="hulax-credit inline-block origin-left text-base text-white/80 transition-all 
                                group-hover:scale-110 group-hover:text-sky-400 group-hover:font-black 
                                group-hover:drop-shadow-[0_0_14px_rgba(56,189,248,0.45)] sm:text-lg">
                            HuLaX
                        </span>
                    </a>
                </div>
            </div>
        </footer>
    );
}
