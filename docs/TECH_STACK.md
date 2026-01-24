# Sentinance Technology Stack

> Complete inventory of all technologies, libraries, frameworks, and tools with integration status.

---

## 📊 Integration Status Legend

| Status | Meaning |
|--------|---------|
| ✅ **Live** | Fully integrated and working |
| 🔄 **Fallback** | Has fallback/mock when unavailable |
| ⚠️ **Optional** | Works without it, enhanced with it |
| 🔧 **Configured** | Config ready, not running |

---

## 🖥️ Frontend

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **Next.js** | 16.1.1 | React framework with App Router | ✅ Live |
| **React** | 18.x | UI library | ✅ Live |
| **TypeScript** | 5.x | Type safety | ✅ Live |
| **TailwindCSS** | 3.x | Styling | ✅ Live |
| **Lucide React** | latest | Icons | ✅ Live |
| **Zod** | 3.x | Schema validation | ✅ Live |
| **Vitest** | 2.x | Unit testing | 🔧 Configured |
| **Playwright** | latest | E2E testing | 🔧 Configured |
| **@testing-library/react** | 16.x | Component testing | 🔧 Configured |

---

## ⚙️ Backend

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **Python** | 3.10+ | Runtime | ✅ Live |
| **FastAPI** | 0.109+ | API framework | ✅ Live |
| **Uvicorn** | 0.27+ | ASGI server | ✅ Live |
| **Pydantic** | 2.x | Data validation | ✅ Live |
| **structlog** | latest | Structured logging | ✅ Live |
| **python-jose** | latest | JWT handling | ✅ Live |
| **passlib** | latest | Password hashing | ✅ Live |
| **httpx** | latest | Async HTTP client | ✅ Live |
| **python-dotenv** | latest | Env loading | ✅ Live |

---

## 🗄️ Databases & Storage

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **PostgreSQL** | 16 | Primary database | 🔧 Configured |
| **SQLAlchemy** | 2.0 | Async ORM | ✅ Live (mock) |
| **asyncpg** | latest | PostgreSQL driver | 🔧 Configured |
| **Alembic** | latest | DB migrations | 🔧 Configured |
| **Redis** | 7 | Caching, sessions | 🔄 Fallback (runs without) |
| **Qdrant** | latest | Vector database | 🔄 Fallback (mock) |

---

## 📡 Real-Time & Messaging

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **WebSocket** | native | Real-time streaming | ✅ Live |
| **Apache Kafka** | 3.6 (KRaft) | Event streaming | 🔧 Configured |
| **SSE** | native | Server-Sent Events | ✅ Live |

---

## 🤖 AI/ML Stack

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **Google Gemini** | 2.5-flash | Primary LLM | 🔄 Fallback to mock |
| **Groq** | llama-3.3-70b | Fallback LLM | 🔄 Fallback to mock |
| **LangGraph** | latest | Multi-agent orchestration | ✅ Live (mock agents) |
| **PyTorch** | 2.x | LSTM model | 🔄 Fallback (mock predictions) |
| **MLflow** | latest | Experiment tracking | 🔧 Configured |
| **SHAP** | latest | Model explainability | 🔧 Configured |
| **LIME** | latest | Model explainability | 🔧 Configured |
| **Feast** | latest | Feature store | 🔧 Configured |

---

## 📊 Data Engineering

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **dbt** | 1.7 | Data transformations | 🔧 Configured |
| **PySpark** | 3.5 | Feature engineering | 🔧 Configured |
| **Prefect** | 2.x | Orchestration | 🔧 Configured |
| **Great Expectations** | latest | Data quality | 🔧 Configured |
| **yfinance** | latest | Index data | ✅ Live |
| **Binance API** | v3 | Crypto prices | ✅ Live |

---

## 📈 Observability

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **Prometheus** | latest | Metrics collection | 🔧 Configured |
| **Grafana** | latest | Dashboards | 🔧 Configured |
| **OpenTelemetry** | latest | Tracing | 🔧 Configured |
| **structlog** | latest | JSON logging | ✅ Live |

---

## 🐳 Infrastructure

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| **Docker** | latest | Containerization | ✅ Live |
| **docker-compose** | latest | Local orchestration | ✅ Live |
| **Kubernetes** | 1.28+ | Production orchestration | 🔧 Configured |
| **GitHub Actions** | latest | CI/CD | 🔧 Configured |
| **Nginx** | latest | Ingress/proxy | 🔧 Configured |

---

## 🧪 Testing

| Technology | Type | Status |
|------------|------|--------|
| **pytest** | Unit/Integration | ✅ 126 tests passing |
| **pytest-asyncio** | Async tests | ✅ Live |
| **pytest-cov** | Coverage | ✅ 96% coverage |
| **respx** | HTTP mocking | ✅ Live |
| **Vitest** | Frontend unit | 🔧 Configured |
| **Playwright** | E2E browser | 🔧 Configured |

---

## 🔄 Fallback Behavior Summary

| Component | Primary | Fallback | Current |
|-----------|---------|----------|---------|
| **LLM** | Gemini API | Mock responses | 🔄 Mock |
| **Cache** | Redis | In-memory/none | ✅ Redis running |
| **Database** | PostgreSQL | Mock data | 🔄 Mock |
| **Vector DB** | Qdrant | Mock search | 🔄 Mock |
| **Predictions** | LSTM model | Mock predictions | 🔄 Mock |
| **Prices** | Binance + yfinance | Simulated | ✅ Live |
| **Kafka** | Kafka cluster | Direct writes | 🔄 Direct |

---

## 📦 Key Python Dependencies

```txt
# API Core
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.0.0
python-jose>=3.3.0
passlib>=1.7.4
httpx>=0.26.0
structlog>=24.1.0

# Database
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
redis>=5.0.0

# AI/ML
google-generativeai>=0.3.0
langchain>=0.1.0
langgraph>=0.0.20
torch>=2.0.0
numpy>=1.26.0
scipy>=1.12.0

# Data
yfinance>=0.2.0
pandas>=2.0.0

# Observability
prometheus-client>=0.19.0
opentelemetry-api>=1.22.0
```

---

## 📦 Key Frontend Dependencies

```json
{
  "next": "16.1.1",
  "react": "^18.0.0",
  "typescript": "^5.0.0",
  "tailwindcss": "^3.4.0",
  "lucide-react": "^0.300.0",
  "zod": "^3.22.0",
  "vitest": "^2.0.0",
  "@playwright/test": "^1.40.0"
}
```

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     SENTINANCE STACK                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  FRONTEND (Next.js 16 + React 18 + TypeScript)              │
│  ├── TailwindCSS, Lucide Icons                               │
│  └── WebSocket client for real-time                          │
│                         │                                     │
│                         ▼                                     │
│  API LAYER (FastAPI + Uvicorn)                               │
│  ├── JWT Auth (python-jose)                                  │
│  ├── Pydantic validation                                     │
│  └── WebSocket server                                         │
│           │                                                   │
│     ┌─────┴─────┬─────────────┬─────────────┐               │
│     ▼           ▼             ▼             ▼                │
│  ┌──────┐  ┌────────┐  ┌──────────┐  ┌──────────┐          │
│  │Redis │  │Postgres│  │  Qdrant  │  │  Kafka   │          │
│  │(opt) │  │ (opt)  │  │  (opt)   │  │  (opt)   │          │
│  └──────┘  └────────┘  └──────────┘  └──────────┘          │
│                                                               │
│  ML LAYER                                                    │
│  ├── PyTorch LSTM (fallback: mock)                          │
│  ├── Gemini/Groq LLM (fallback: mock)                       │
│  └── LangGraph agents                                        │
│                                                               │
│  DATA SOURCES                                                │
│  ├── Binance API ✅ Live                                     │
│  └── yfinance ✅ Live                                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```
