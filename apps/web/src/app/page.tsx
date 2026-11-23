import Link from 'next/link';
import { SiteHeader } from '@/components/site/site-header';
import { SiteFooter } from '@/components/site/site-footer';
import { SignalTape } from '@/components/home/signal-tape';
import { RetrievalChart } from '@/components/home/retrieval-chart';
import { AgentLog } from '@/components/home/agent-log';
import { InView } from '@/components/home/in-view';
import './home.css';

const delay = (ms: number) => ({ ['--sx-delay' as string]: `${ms}ms` });

const path = [
  { name: 'Source fixture', detail: 'Exchange-shaped trades' },
  { name: 'Kafka', detail: 'Real broker in CI' },
  { name: 'PostgreSQL', detail: 'Idempotent price ledger' },
  { name: 'Parquet', detail: 'Checked lake partition' },
  { name: 'Forecast', detail: 'Historical, hash-pinned' },
  { name: 'Citation', detail: 'Agent cites its source' },
  { name: 'Browser', detail: 'Read in real Chromium' },
];

export default function HomePage() {
  return (
    <div className="min-h-screen relative">
      <SiteHeader />
      <main className="sx">
        <section className="sx-hero">
          <div className="sx-wrap">
            <div className="sx-hero__copy">
            <h1 className="sx-hero__title sx-grad sx-rise" style={delay(0)}>
              Every price move leaves a paper trail.
            </h1>
            <p className="sx-hero__lede sx-rise" style={delay(140)}>
              Sentinance streams crypto and index prices through Kafka into a durable ledger, forecasts from history, and
              lets an AI research agent explain a move only when it can cite the evidence.
            </p>
            <div className="sx-actions sx-rise" style={delay(260)}>
              <Link href="/demo" className="sx-button sx-button--primary">
                Open market terminal
              </Link>
              <Link href="/data/evidence" className="sx-button sx-button--ghost">
                Inspect the evidence
              </Link>
            </div>
            </div>
          </div>
          <div className="sx-rise" style={delay(320)}>
            <SignalTape />
          </div>
        </section>

        <section className="sx-section" aria-labelledby="rag-title">
          <div className="sx-wrap sx-split">
            <div>
              <h2 className="sx-h2 sx-grad" id="rag-title">
                Retrieval that is measured, not assumed.
              </h2>
              <p className="sx-intro">
                Before the agent writes a word, it has to find the right news. CI scores dense retrieval in a real Qdrant
                container against a plain keyword baseline, and fails the build if recall or ranking drops below the gate.
              </p>
            </div>
            <RetrievalChart />
          </div>
        </section>

        <section className="sx-section" aria-labelledby="agent-title">
          <div className="sx-wrap sx-split">
            <AgentLog />
            <div>
              <h2 className="sx-h2 sx-grad" id="agent-title">
                The agent knows when to say no.
              </h2>
              <p className="sx-intro">
                Asked for a live quote, a trade or a secret, the LangGraph agent refuses before it opens a single file.
                These are the twelve prompts its golden test sends, including the injected ones. The one request it may
                answer is a comparison of historical forecasts, and only from a report whose SHA-256 it can verify.
              </p>
            </div>
          </div>
        </section>

        <section className="sx-section" aria-labelledby="path-title">
          <div className="sx-wrap">
            <h2 className="sx-h2 sx-grad" id="path-title">
              One event, verified from source to screen.
            </h2>
            <p className="sx-intro">
              This is the trace CI runs on every change, with real services rather than mocks. Tamper with the report and
              the page shows nothing instead of something wrong.
            </p>
            <InView as="ol" className="sx-path" aria-label="The verified path">
              {path.map((p, i) => (
                <li key={p.name} style={{ ['--sx-i' as string]: i }}>
                  <strong>{p.name}</strong>
                  {p.detail}
                </li>
              ))}
            </InView>
            <div className="sx-honest">
              <div>
                <h3>Proven in CI</h3>
                <p>Kafka to PostgreSQL storage, the Parquet lake, forecast evidence, the agent refusals and RAG retrieval, each against real services.</p>
              </div>
              <div>
                <h3>Fixture-driven</h3>
                <p>Exchange trades in CI come from fixtures. Live Binance ingest exists but is opt-in, and CI never calls a live exchange.</p>
              </div>
              <div>
                <h3>Not claimed</h3>
                <p>No trading advice, no profitable strategy and no live-price guarantee. Forecasts are historical and labelled as such.</p>
              </div>
            </div>
            <div className="sx-cta">
              <p>See the prices, then see where they came from.</p>
              <div className="sx-actions" style={{ marginTop: 0 }}>
                <Link href="/demo" className="sx-button sx-button--primary">
                  Open the terminal
                </Link>
                <Link href="/architecture" className="sx-button sx-button--ghost">
                  Read the architecture
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
