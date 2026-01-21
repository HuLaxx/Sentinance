'use client';

import LiveTerminal from '@/components/terminal/LiveTerminal';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';

export default function LivePage() {
    return (
        <div className="min-h-screen bg-zinc-950">
            <SiteHeader />
            <LiveTerminal />
            <SiteFooter />
        </div>
    );
}
