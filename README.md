<div align="center">
  <!-- Waving Header -->
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=280&section=header&text=SENTINANCE&fontSize=80&fontAlignY=35&animation=fadeIn&fontColor=ffffff&desc=Autonomous%20AI%20Crypto%20Intelligence&descSize=20&descAlignY=60&descAlign=50" alt="Sentinance Header" width="100%" />

  <!-- Typing SVG -->
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&pause=1000&color=38BDF8&center=true&vCenter=true&width=600&lines=Agentic+AI+Working+24%2F7;Real-Time+WebSocket+Streaming;Institutional-Grade+Analytics" alt="Typing SVG" />
  </a>

  <br />

  <!-- Badges -->
  <img src="https://img.shields.io/badge/Tests-126%20Passing-brightgreen?style=for-the-badge" alt="Tests" />
  <img src="https://img.shields.io/badge/Coverage-96%25-brightgreen?style=for-the-badge" alt="Coverage" />
  <img src="https://img.shields.io/badge/Build-Passing-success?style=for-the-badge" alt="Build" />
  <br />
  <img src="https://img.shields.io/badge/Next.js_16-black?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" />

  <br /><br />

  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-api-endpoints">API</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-architecture">Architecture</a>
</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 📊 Real-Time Streaming
Live WebSocket feeds for **10+ crypto assets** (BTC, ETH, SOL, XRP, etc.) and **4 global indices** with multi-exchange aggregation (Binance, Coinbase, Kraken).

</td>
<td width="50%">

### 🤖 Agentic AI
LangGraph multi-agent system with **Gemini + Groq** fallback for autonomous market analysis and verifiable reasoning.

</td>
</tr>
<tr>
<td width="50%">

### 📈 ML Predictions
**LSTM neural networks** for price forecasting with confidence intervals (4h, 24h, 7d) and <100ms inference latency.

</td>
<td width="50%">

### ⚡ Anomaly Detection
Real-time alerts for price spikes, volume surges, and manipulation patterns (pump-and-dump detection).

</td>
</tr>
<tr>
<td width="50%">

### 🔍 RAG Pipeline
**Qdrant** vector store + semantic search for intelligent market insights

</td>
<td width="50%">

### 🛡️ Production-Ready
Docker, Kubernetes, Prometheus, Grafana - enterprise-grade infrastructure

</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites

- Docker Desktop (recommended) OR
- Node.js 18+ & Python 3.10+

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/HuLaxx/Sentinance.git
cd sentinance
cp .env.example .env
```

### 2️⃣ Start with Docker (Recommended)

```bash
# Start all services
docker-compose up -d --build

# Or use dev compose file
docker-compose -f docker-compose.dev.yml up -d
```

### 3️⃣ Or Start Manually

**Start Backend (FastAPI)**
```bash
cd apps/api
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

**Start Frontend (Next.js)**
```bash
cd apps/web
npm install
npm run dev
```

**Start Redis (Optional - for caching)**
```bash
docker run -d --name sentinance-redis -p 6380:6379 redis:7-alpine
```

### 4️⃣ Access Services

| Service | URL | Description |
|---------|-----|-------------|
| 🌐 Demo Dashboard | http://localhost:3000/demo | Main demo interface |
| 🔌 API Docs | http://localhost:8000/docs | Swagger API documentation |
| ❤️ Health Check | http://localhost:8000/health | Service health status |
| 📊 Grafana | http://localhost:3001 | Monitoring dashboards |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws/prices` | Real-time price streaming |
| `GET` | `/api/prices` | All current prices (14 assets) |
| `GET` | `/api/prices/{symbol}` | Single asset price |
| `GET` | `/api/prices/{symbol}/history` | Price history |
| `POST` | `/api/chat` | AI chat with market context |
| `GET` | `/api/predict/{symbol}` | ML price prediction |
| `GET` | `/api/indicators/{symbol}` | Technical indicators (RSI, MACD, etc.) |
| `POST` | `/api/alerts` | Create price alert |
| `GET` | `/api/alerts/active` | List active alerts |
| `GET` | `/api/news` | Latest market news |
| `GET` | `/api/stats` | Market statistics |
| `GET` | `/api/stats/movers` | Top movers |

### Example API Calls

```bash
# Get all prices
curl http://localhost:8000/api/prices

# Get BTC prediction
curl http://localhost:8000/api/predict/BTCUSDT

# Get technical indicators
curl http://localhost:8000/api/indicators/BTCUSDT

# AI Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the outlook for Bitcoin?"}'
```

---

## 🏗️ Architecture

```
                         ┌──────────────────────────────────────┐
                         │           SENTINANCE                 │
                         │    Real-Time Market Intelligence     │
                         └──────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
     ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
     │   NEXT.JS 16    │        │    FASTAPI      │        │   LANGGRAPH     │
     │   Frontend      │◀──────▶│    Backend      │◀──────▶│   AI Agents     │
     │   TypeScript    │  REST  │    Python       │        │   Gemini AI     │
     └─────────────────┘  WS    └─────────────────┘        └─────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
     ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
     │   POSTGRESQL    │        │     REDIS       │        │    QDRANT       │
     │   Database      │        │   Cache/PubSub  │        │   Vector DB     │
     └─────────────────┘        └─────────────────┘        └─────────────────┘
```

---

## 🌍 Deployment (100% Free)

Deploy to production using **free tiers** of cloud services:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     VERCEL      │────▶│    RAILWAY      │────▶│    SUPABASE     │
│   (Frontend)    │     │   (Backend)     │     │  (PostgreSQL)   │
│      FREE       │     │      FREE       │     │      FREE       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Quick Deploy Steps:

1. **Vercel** → Deploy `apps/web` (Next.js frontend)
2. **Railway** → Deploy `apps/api` (FastAPI backend)
3. **Supabase** → Create PostgreSQL database
4. **Upstash** → Create Redis instance
5. **Connect** → Add environment variables

📖 **[Full Deployment Guide →](docs/VERCEL_DEPLOYMENT.md)**

---

## 🧪 Testing

```bash
# Backend tests (126+ passing)
cd apps/api
$env:JWT_SECRET="test-secret"           # PowerShell
# export JWT_SECRET="test-secret"       # Bash
pytest tests/ -v --cov=. --cov-report=html
```

| Category | Tests | Coverage |
|----------|-------|----------|
| Unit - Indicators | 29 | 96% |
| Unit - Predictor | 22 | 94% |
| Integration - API | 18 | 88% |
| Auth & Alerts | 20 | 85% |
| WebSocket & Chat | 37 | 72% |

**Frontend tests:**
```bash
cd apps/web
npm run test
```

---

## 📊 Tech Stack

<div align="center">
  <img src="https://skillicons.dev/icons?i=nextjs,react,ts,tailwind,python,fastapi,postgres,redis,docker,kubernetes,kafka,grafana,prometheus,pytorch,gcp&perline=15" />
</div>

<table>
<tr>
<td align="center" width="20%">

**Frontend**

Next.js 16<br>
React 18<br>
TailwindCSS<br>
TypeScript

</td>
<td align="center" width="20%">

**Backend**

FastAPI<br>
SQLAlchemy<br>
Pydantic v2<br>
AsyncIO

</td>
<td align="center" width="20%">

**AI/ML**

LangGraph<br>
Gemini + Groq<br>
PyTorch LSTM<br>
SHAP/LIME

</td>
<td align="center" width="20%">

**Data**

PostgreSQL<br>
Redis<br>
Kafka<br>
Qdrant<br>
BeautifulSoup

</td>
<td align="center" width="20%">

**DevOps**

Docker<br>
Kubernetes<br>
Prometheus<br>
Grafana

</td>
</tr>
</table>

> **[📚 Full Tech Stack Document →](docs/TECH_STACK.md)** — 30+ technologies with fallback status

---

## 📁 Project Structure

```
sentinance/
├── apps/
│   ├── api/                 # FastAPI backend (60+ modules)
│   │   ├── main.py          # Entry point
│   │   ├── agent.py         # LangGraph multi-agent system
│   │   ├── llm_wrapper.py   # Gemini/Groq with RAG
│   │   ├── predictor.py     # LSTM price predictions
│   │   ├── indicators.py    # Technical analysis (RSI, MACD)
│   │   ├── streaming_llm.py # SSE token streaming
│   │   └── tests/           # 126+ tests
│   ├── web/                 # Next.js 16 frontend
│   │   ├── src/app/demo/    # Demo dashboard
│   │   └── src/app/asset/   # Asset detail pages
│   ├── ml/                  # ML training & explainability
│   └── orchestration/       # Prefect data pipelines
├── docs/                    # Documentation
│   ├── TECH_STACK.md        # Technology inventory
│   ├── PORTFOLIO_CLAIMS.md  # Skills by role
│   ├── WORKFLOW_MAP.md      # Integration diagram
│   └── DEPLOYMENT.md        # Production deployment
├── infra/
│   ├── k8s/                 # Kubernetes manifests (12 files)
│   └── monitoring/          # Prometheus/Grafana configs
├── notebooks/               # Jupyter EDA notebooks
├── docker-compose.yml       # Local development
└── docker-compose.prod.yml  # Production setup
```

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [TECH_STACK.md](docs/TECH_STACK.md) | Full technology inventory with fallback status |
| [PORTFOLIO_CLAIMS.md](docs/PORTFOLIO_CLAIMS.md) | Skills breakdown by role |
| [WORKFLOW_MAP.md](docs/WORKFLOW_MAP.md) | Visual architecture & data flow |
| [API.md](docs/API.md) | API endpoint documentation |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |
| [VERCEL_DEPLOYMENT.md](docs/VERCEL_DEPLOYMENT.md) | Free-tier deployment steps |

---

## 📜 License

MIT © 2026 Sentinance

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0ea5e9&height=100&section=footer&text=Built%20by%20HuLaX&fontSize=24&fontColor=ffffff" width="100%" />
  
  <br />
  
  <a href="https://hulax.vercel.app">Portfolio</a> • 
  <a href="https://github.com/HuLaxx">GitHub</a> • 
  <a href="https://linkedin.com/in/rahul-khanke">LinkedIn</a> • 
  <a href="mailto:rahulkhanke02@gmail.com">Email</a>
  
  <br /><br />
  
  <a href="#top">⬆️ Back to Top</a>
</div>
