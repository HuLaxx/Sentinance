'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { Activity, ArrowUpRight, Menu, X } from 'lucide-react';

const pages = [
  ['/', 'Overview'], ['/architecture', 'Architecture'], ['/data', 'Data'],
  ['/techdash', 'Technology'], ['/demo', 'Terminal'],
];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return (
    <header className="atlas-header">
      <div className="atlas-bar">
        <Link className="atlas-brand" href="/" onClick={() => setOpen(false)}>
          <Activity aria-hidden="true" /> Sentinance
        </Link>
        <button className="atlas-toggle" aria-controls="site-nav" aria-expanded={open}
          aria-label={open ? 'Close navigation' : 'Open navigation'} onClick={() => setOpen(!open)}>
          {open ? <X /> : <Menu />}
        </button>
        <nav id="site-nav" aria-label="Primary" className={open ? 'is-open' : ''}>
          {pages.map(([href, label]) => <Link key={href} href={href} onClick={() => setOpen(false)}
            aria-current={pathname === href ? 'page' : undefined}>{label}{href === '/demo' && <ArrowUpRight size={15} />}</Link>)}
        </nav>
      </div>
    </header>
  );
}
