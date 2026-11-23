const stages = [
  ['Ingestion', 'Exchange connectors and news feeds supply the raw observations.', 'Binance WebSocket · Coinbase · Kraken · Yahoo Finance'],
  ['Streaming', 'The event backbone carries updates between services.', 'Apache Kafka · Redis Pub/Sub'],
  ['Processing', 'Services calculate indicators and assemble AI-assisted analysis.', 'FastAPI · LangGraph · Gemini'],
  ['Persistence', 'Different stores support history, fast lookups and semantic retrieval.', 'PostgreSQL · Redis · Qdrant'],
  ['Presentation', 'The browser presents prices, forecasts and the current feed status.', 'Next.js · React · Tailwind CSS'],
];

export default function ArchitecturePage() {
  return <main>
    <header className="atlas-chapter">
      <p className="atlas-kicker">System architecture</p>
      <h1>Anatomy of a signal</h1>
      <p>Trace the path from exchange feeds to the terminal. Expand a stage to inspect its components.</p>
    </header>
    <div className="atlas-index">
      <section>
        <h2>Signal path</h2>
        <div>{stages.map(([name, description, tools], index) =>
          <details className="atlas-stage" key={name} open={index === 0}>
            <summary><span>{String(index + 1).padStart(2, '0')}</span>{name}</summary>
            <p>{description}</p><small>{tools}</small>
          </details>)}</div>
      </section>
      <section><h2>Infrastructure</h2><div>
        <p>Docker packages the services. The repository also includes Kubernetes deployment configuration, monitoring and GitHub Actions workflows.</p>
        <a className="atlas-tool mt-5" href="https://github.com/HuLaxx/Sentinance/tree/main/infra">Inspect deployment configuration ↗</a>
      </div></section>
    </div>
  </main>;
}
