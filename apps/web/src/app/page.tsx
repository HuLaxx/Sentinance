import Image from 'next/image';
import Link from 'next/link';
import { ArrowRight, ArrowUpRight } from 'lucide-react';

const sections = [
  { href: '/architecture', title: 'Architecture', description: 'Follow a tick from the exchange feed to the browser.' },
  { href: '/data', title: 'Data catalogue', description: 'Inspect the sources, storage and checks behind a signal.' },
  { href: '/techdash', title: 'Technology', description: 'Explore the tools that power the research desk.' },
];

export default function HomePage() {
  return <main className="atlas-home">
    <section className="atlas-cover">
      <div>
        <p className="atlas-kicker">Crypto &amp; global indices</p>
        <h1>The Signal<br /><em>Atlas.</em></h1>
        <p className="atlas-deck">A market research desk for tracking assets, exploring forecasts and tracing signals back to their source.</p>
        <Link className="atlas-action" href="/demo">Open market terminal <ArrowUpRight size={20} /></Link>
      </div>
      <figure className="atlas-plate">
        <Image src="/signal-relief.webp" width={1100} height={800} priority alt="Layered copper traces forming a three-dimensional signal landscape" />
        <figcaption><span>Signal topography</span><span>Conceptual study</span></figcaption>
      </figure>
    </section>
    <section className="atlas-directory">
      <h2>Inside the system</h2>
      <div>{sections.map(({href, title, description}) => <Link key={href} href={href}>
        <h3>{title}</h3><p>{description}</p><ArrowRight size={18} />
      </Link>)}</div>
    </section>
  </main>;
}
