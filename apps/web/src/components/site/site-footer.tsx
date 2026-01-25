'use client';

import Link from 'next/link';
import { Github, Linkedin, Mail } from 'lucide-react';

const pageLinks = [
    { label: 'Home', href: '/' },
    { label: 'Architecture', href: '/architecture' },
    { label: 'Tech Dash', href: '/techdash' },
    { label: 'Demo', href: '/demo' },
];

const connectLinks = [
    { icon: Github, label: 'GitHub', href: 'https://github.com/HuLaxx/Sentinance' },
    { icon: Linkedin, label: 'LinkedIn', href: 'https://linkedin.com/in/rahul-khanke' },
    { icon: Mail, label: 'Gmail', href: 'mailto:rahulkhanke02@gmail.com' },
];

export function SiteFooter() {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="relative z-10 m-4 md:m-8 mt-auto rounded-3xl overflow-hidden border border-white/5 bg-black/40 backdrop-blur-3xl shadow-2xl">
            <div className="px-8 lg:px-12 py-16 md:py-24">
                <div className="max-w-7xl mx-auto">
                    {/* Main Grid */}
                    <div className="grid md:grid-cols-4 gap-12 lg:gap-24">
                        {/* Brand Column - Spans 2 */}
                        <div className="md:col-span-2 space-y-6">
                            <Link href="/" className="flex items-center gap-3 group">
                                <div className="w-12 h-12 rounded-xl overflow-hidden shadow-lg shadow-blue-500/20 
                                    group-hover:shadow-blue-500/40 transition-all duration-300 group-hover:scale-105">
                                    <img src="/icon.svg" alt="Sentinance" className="w-full h-full object-cover" />
                                </div>
                                <span className="text-2xl font-bold bg-gradient-to-r from-sky-300 to-blue-500 
                                    bg-clip-text text-transparent group-hover:from-sky-200 group-hover:to-cyan-400 transition-all duration-300">
                                    Sentinance
                                </span>
                            </Link>
                            <p className="text-zinc-400 max-w-sm leading-relaxed text-sm">
                                Real-time crypto market intelligence with autonomous AI agents,
                                ML predictions, and enterprise-grade analytics. Built for traders
                                who demand institutional-quality insights.
                            </p>
                        </div>

                        {/* Pages Column */}
                        <div>
                            <h3 className="font-bold text-sm uppercase tracking-wider text-white mb-6">
                                Pages
                            </h3>
                            <ul className="space-y-4 text-sm text-zinc-400">
                                {pageLinks.map((link) => (
                                    <li key={link.href}>
                                        <Link
                                            href={link.href}
                                            className="hover:text-sky-400 transition-colors">
                                            {link.label}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Connect Column */}
                        <div>
                            <h3 className="font-bold text-sm uppercase tracking-wider text-white mb-6">
                                Connect
                            </h3>
                            <ul className="space-y-4 text-sm text-zinc-400">
                                {connectLinks.map((link) => (
                                    <li key={link.label}>
                                        <a
                                            href={link.href}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="hover:text-sky-400 transition-colors flex items-center gap-2 group">
                                            <link.icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                                            {link.label}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div className="max-w-7xl mx-auto pt-12 mt-12 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-6 text-sm text-zinc-500">
                    <p>© {currentYear} Sentinance. All rights reserved.</p>
                    <button
                        type="button"
                        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        className="text-white/60 hover:text-white transition-colors"
                    >
                        Back to top
                    </button>
                    <a
                        href="https://hulax.vercel.app"
                        target="_blank"
                        rel="noreferrer"
                        className="group inline-flex items-baseline gap-1.5 md:mr-6">
                        <span className="hulax-credit-label text-sm text-sky-100/60 transition-all duration-300 group-hover:text-sky-100 group-hover:drop-shadow-[0_0_8px_rgba(56,189,248,0.35)]">
                            A Project By
                        </span>
                        <span className="hulax-credit hulax-credit-gradient text-base sm:text-lg transition-all duration-300 origin-left group-hover:font-semibold">
                            HuLaX
                        </span>
                    </a>
                </div>
            </div>
        </footer>
    );
}
