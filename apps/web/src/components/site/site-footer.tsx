'use client';

import Link from 'next/link';
import { Activity, Github, Twitter, Linkedin, Mail } from 'lucide-react';

const footerLinks = {
    product: [
        { label: 'Live Terminal', href: '/live' },
        { label: 'Architecture', href: '/architecture' },
        { label: 'Data Platform', href: '/data' },
        { label: 'Tech Dash', href: '/techdash' },
    ],
    resources: [
        { label: 'Demo Terminal', href: '/demo' },
        { label: 'Architecture Overview', href: '/architecture' },
        { label: 'Data Platform', href: '/data' },
        { label: 'Tech Dash', href: '/techdash' },
    ],
    company: [
        { label: 'Home', href: '/' },
        { label: 'Live Terminal', href: '/live' },
        { label: 'Demo Terminal', href: '/demo' },
        { label: 'Tech Dash', href: '/techdash' },
    ],
};

const socialLinks = [
    { icon: Github, href: 'https://github.com', label: 'GitHub' },
    { icon: Twitter, href: 'https://twitter.com', label: 'Twitter' },
    { icon: Linkedin, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: Mail, href: 'mailto:hello@sentinance.io', label: 'Email' },
];

export function SiteFooter() {
    return (
        <footer className="bg-zinc-900/50 border-t border-zinc-800">
            <div className="max-w-7xl mx-auto px-6 py-16">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-8">
                    {/* Brand */}
                    <div className="col-span-2">
                        <Link href="/" className="flex items-center gap-3 group mb-4">
                            <div className="w-10 h-10 rounded-xl overflow-hidden shadow-lg shadow-blue-900/20">
                                <img src="/icon.svg" alt="Sentinance" className="w-full h-full object-cover" />
                            </div>
                            <span className="text-xl font-bold bg-gradient-to-r from-sky-300 to-blue-500 bg-clip-text text-transparent">
                                Sentinance
                            </span>
                        </Link>
                        <p className="text-sm text-zinc-400 mb-6 max-w-xs">
                            Enterprise-grade crypto market intelligence with agentic AI and real-time analytics.
                        </p>
                        <div className="flex items-center gap-3">
                            {socialLinks.map((social) => (
                                <a
                                    key={social.label}
                                    href={social.href}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="w-9 h-9 rounded-lg bg-zinc-800/50 border border-zinc-700/50 flex items-center justify-center text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 hover:border-zinc-600 transition-all"
                                    aria-label={social.label}
                                >
                                    <social.icon className="w-4 h-4" />
                                </a>
                            ))}
                        </div>
                    </div>

                    {/* Product */}
                    <div>
                        <h3 className="text-sm font-semibold text-zinc-100 mb-4">Product</h3>
                        <ul className="space-y-3">
                            {footerLinks.product.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-zinc-400 hover:text-zinc-100 transition-colors"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Resources */}
                    <div>
                        <h3 className="text-sm font-semibold text-zinc-100 mb-4">Resources</h3>
                        <ul className="space-y-3">
                            {footerLinks.resources.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-zinc-400 hover:text-zinc-100 transition-colors"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Company */}
                    <div>
                        <h3 className="text-sm font-semibold text-zinc-100 mb-4">Company</h3>
                        <ul className="space-y-3">
                            {footerLinks.company.map((link) => (
                                <li key={link.href}>
                                    <Link
                                        href={link.href}
                                        className="text-sm text-zinc-400 hover:text-zinc-100 transition-colors"
                                    >
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* Bottom */}
                <div className="mt-12 pt-8 border-t border-zinc-800 flex flex-col md:flex-row items-center justify-between gap-4">
                    <p className="text-sm text-zinc-500">
                        © {new Date().getFullYear()} Sentinance. All rights reserved.
                    </p>
                    <div className="flex items-center gap-6">
                        <Link href="/privacy" className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors">
                            Privacy Policy
                        </Link>
                        <Link href="/terms" className="text-sm text-zinc-500 hover:text-zinc-300 transition-colors">
                            Terms of Service
                        </Link>
                    </div>
                </div>
            </div>
        </footer>
    );
}
