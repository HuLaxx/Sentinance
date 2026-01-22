'use client';

import Link from 'next/link';
import { Github, Linkedin, Mail } from 'lucide-react';

const pageLinks = [
    { label: 'Home', href: '/' },
    { label: 'Live Terminal', href: '/live' },
    { label: 'Architecture', href: '/architecture' },
    { label: 'Tech Dash', href: '/techdash' },
    { label: 'Demo', href: '/demo' },
];

const connectLinks = [
    { icon: Github, label: 'GitHub', href: 'https://github.com/Hulaxx' },
    { icon: Linkedin, label: 'LinkedIn', href: 'https://linkedin.com/in/rahul-khanke' },
    { icon: Mail, label: 'Gmail', href: 'mailto:rahulkhanke02@gmail.com' },
];

export function SiteFooter() {
    return (
        <footer className="relative mx-4 mb-4 rounded-xl overflow-hidden mt-12">
            {/* Gradient background */}
            <div className="absolute inset-0 bg-gradient-to-r from-zinc-900 via-purple-950/20 to-zinc-900" />
            <div className="absolute inset-0 bg-gradient-to-t from-purple-900/10 via-transparent to-transparent" />

            <div className="relative px-6 md:px-12 py-12">
                <div className="max-w-7xl mx-auto">
                    {/* Main footer content */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-8">
                        {/* Brand Section */}
                        <div className="space-y-4">
                            <Link href="/" className="flex items-center gap-3 group">
                                <div className="w-10 h-10 rounded-xl overflow-hidden shadow-lg shadow-purple-500/20 
                                    group-hover:shadow-purple-500/40 transition-shadow">
                                    <img src="/icon.svg" alt="Sentinance" className="w-full h-full object-cover" />
                                </div>
                                <span className="text-xl font-bold bg-gradient-to-r from-zinc-100 to-zinc-300 
                                    bg-clip-text text-transparent group-hover:from-sky-300 group-hover:to-blue-400 transition-all">
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
                            <h3 className="text-sm font-semibold text-zinc-100 uppercase tracking-wider mb-4">
                                Pages
                            </h3>
                            <ul className="space-y-3">
                                {pageLinks.map((link) => (
                                    <li key={link.href}>
                                        <Link
                                            href={link.href}
                                            className="text-sm text-zinc-400 hover:text-zinc-100 
                                                transition-colors duration-200 hover:translate-x-1 inline-block"
                                        >
                                            {link.label}
                                        </Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        {/* Connect Section */}
                        <div>
                            <h3 className="text-sm font-semibold text-zinc-100 uppercase tracking-wider mb-4">
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
                                                hover:text-zinc-100 transition-colors duration-200 group"
                                        >
                                            <link.icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                                            {link.label}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    {/* Bottom Section */}
                    <div className="mt-12 pt-6 border-t border-zinc-800/50 flex flex-col md:flex-row 
                        items-center justify-between gap-4">
                        <p className="text-sm text-zinc-500">
                            © {new Date().getFullYear()} Sentinance. All rights reserved.
                        </p>
                        <p className="text-sm text-zinc-400">
                            <span className="italic">A Project By </span>
                            <a
                                href="https://github.com/Hulaxx"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="font-semibold text-zinc-300 hover:text-sky-400 transition-colors"
                            >
                                HuLaX
                            </a>
                        </p>
                    </div>
                </div>
            </div>
        </footer>
    );
}
