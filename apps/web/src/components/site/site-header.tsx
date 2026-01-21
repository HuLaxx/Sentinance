'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';
import { Menu, X, Github, Activity, ExternalLink } from 'lucide-react';

const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/architecture', label: 'Architecture' },
    { href: '/techdash', label: 'Tech Dash' },
    { href: '/live', label: 'Live Terminal', highlight: true },
];

export function SiteHeader() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);
    const pathname = usePathname();

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <header
            className={`fixed top-4 left-0 right-0 z-50 transition-all duration-300 px-4 flex justify-center`}
        >
            <div
                className={`w-full max-w-6xl transition-all duration-300 rounded-2xl border ${scrolled
                    ? 'bg-zinc-950/90 backdrop-blur-xl border-zinc-800 shadow-xl shadow-black/20 py-1'
                    : 'bg-transparent border-transparent py-2'
                    }`}
            >
                <div className="px-6 flex items-center justify-between h-14">
                    {/* Logo */}
                    <Link href="/" className="flex items-center gap-3 group">
                        <div className="w-9 h-9 rounded-xl overflow-hidden shadow-lg shadow-blue-900/20 group-hover:shadow-blue-900/40 group-hover:scale-105 transition-all">
                            <img src="/icon.svg" alt="Sentinance" className="w-full h-full object-cover" />
                        </div>
                        <div className="flex flex-col">
                            <span className="text-lg font-bold bg-gradient-to-r from-sky-300 to-blue-500 bg-clip-text text-transparent">
                                Sentinance
                            </span>
                        </div>
                    </Link>

                    {/* Desktop Nav */}
                    <nav className="hidden md:flex items-center gap-1">
                        {navLinks.map((link) => {
                            const isActive = pathname === link.href;

                            if (link.highlight) {
                                return (
                                    <Link
                                        key={link.href}
                                        href={link.href}
                                        className="ml-2 flex items-center gap-2 px-5 py-2 text-sm font-medium bg-gradient-to-r from-blue-600 to-cyan-500 text-white rounded-full hover:from-blue-500 hover:to-cyan-400 transition-all shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40"
                                    >
                                        {link.label}
                                        <ExternalLink className="w-3 h-3" />
                                    </Link>
                                );
                            }

                            return (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    className={`px-4 py-2 text-sm transition-all rounded-full hover:bg-slate-800/50 ${isActive
                                        ? 'text-sky-400 font-medium'
                                        : 'text-slate-400 hover:text-sky-200'
                                        }`}
                                >
                                    {link.label}
                                </Link>
                            );
                        })}
                    </nav>

                    {/* Actions */}
                    <div className="hidden md:flex items-center gap-3">
                        <a
                            href="https://github.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 hover:text-zinc-100 transition-colors rounded-full hover:bg-zinc-800/50"
                        >
                            <Github className="w-4 h-4" />
                            <span>GitHub</span>
                        </a>
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        className="md:hidden p-2 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 rounded-lg transition-colors"
                    >
                        {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                    </button>
                </div>

                {/* Mobile Menu */}
                {mobileMenuOpen && (
                    <div className="md:hidden px-4 pb-4 pt-2 border-t border-zinc-800">
                        <nav className="flex flex-col gap-1">
                            {navLinks.map((link) => (
                                <Link
                                    key={link.href}
                                    href={link.href}
                                    onClick={() => setMobileMenuOpen(false)}
                                    className={`px-4 py-3 text-sm rounded-lg transition-colors ${pathname === link.href
                                        ? 'text-sky-400 bg-blue-900/10'
                                        : 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/50'
                                        }`}
                                >
                                    {link.label}
                                </Link>
                            ))}
                            <a
                                href="https://github.com"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-2 px-4 py-3 text-sm text-zinc-400 hover:text-zinc-100"
                            >
                                <Github className="w-4 h-4" />
                                GitHub
                            </a>
                        </nav>
                    </div>
                )}
            </div>
        </header>
    );
}
