import { ExternalLink } from "lucide-react";

const techStack = {
  frontend: [
    { name: 'Next.js 16', url: 'https://nextjs.org' },
    { name: 'React 18', url: 'https://react.dev' },
    { name: 'TypeScript', url: 'https://typescriptlang.org' },
    { name: 'TailwindCSS', url: 'https://tailwindcss.com' },
    { name: 'Vercel AI SDK', url: 'https://sdk.vercel.ai' },
    { name: 'Zod', url: 'https://zod.dev' },
  ],
  backend: [
    { name: 'FastAPI', url: 'https://fastapi.tiangolo.com' },
    { name: 'Python 3.11', url: 'https://python.org' },
    { name: 'Pydantic', url: 'https://docs.pydantic.dev' },
    { name: 'SQLAlchemy', url: 'https://sqlalchemy.org' },
    { name: 'structlog', url: 'https://www.structlog.org' },
    { name: 'httpx', url: 'https://www.python-httpx.org' },
  ],
  data: [
    { name: 'PostgreSQL', url: 'https://postgresql.org' },
    { name: 'Redis', url: 'https://redis.io' },
    { name: 'Apache Kafka', url: 'https://kafka.apache.org' },
    { name: 'Multi-Exchange', url: 'https://binance.com' },
    { name: 'yfinance', url: 'https://github.com/ranaroussi/yfinance' },
    { name: 'BeautifulSoup', url: 'https://beautiful-soup-4.readthedocs.io' },
  ],
  ai_ml: [
    { name: 'LangGraph', url: 'https://langchain-ai.github.io/langgraph/' },
    { name: 'Gemini AI', url: 'https://ai.google.dev' },
    { name: 'Groq', url: 'https://groq.com' },
    { name: 'PyTorch', url: 'https://pytorch.org' },
    { name: 'Qdrant', url: 'https://qdrant.tech' },
    { name: 'SSE Streaming', url: 'https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events' },
  ],
  devops: [
    { name: 'Docker', url: 'https://docker.com' },
    { name: 'Kubernetes', url: 'https://kubernetes.io' },
    { name: 'Prometheus', url: 'https://prometheus.io' },
    { name: 'Grafana', url: 'https://grafana.com' },
    { name: 'GitHub Actions', url: 'https://github.com/features/actions' },
    { name: 'MLflow', url: 'https://mlflow.org' },
  ],
};

const categories = [
  ['Frontend', techStack.frontend], ['Backend', techStack.backend],
  ['Data layer', techStack.data], ['AI & machine learning', techStack.ai_ml],
  ['Operations', techStack.devops],
] as const;

export default function TechDashPage() {
  return <main>
    <header className="atlas-chapter">
      <p className="atlas-kicker">Technology directory</p>
      <h1>The research stack</h1>
      <p>Browse the tools behind the terminal and open their official documentation.</p>
    </header>
    <div className="atlas-index">
      {categories.map(([title, tools]) => <section key={title}>
        <h2>{title}</h2>
        <div className="atlas-tools">{tools.map(tool =>
          <a className="atlas-tool" key={tool.name} href={tool.url} target="_blank" rel="noopener noreferrer">
            {tool.name}<ExternalLink size={15} />
          </a>)}</div>
      </section>)}
    </div>
  </main>;
}
