<div align="center">
  <!-- Waving Header -->
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=280&section=header&text=SENTINANCE&fontSize=80&fontAlignY=35&animation=fadeIn&fontColor=ffffff&desc=Autonomous%20AI%20Crypto%20Intelligence&descSize=20&descAlignY=60&descAlign=50" alt="Sentinance Header" width="100%" />

  <!-- Typing SVG for Cinematic Intro -->
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&pause=1000&color=38BDF8&center=true&vCenter=true&width=600&lines=Agentic+AI+Working+24%2F7;Real-Time+WebSocket+Streaming;Institutional-Grade+Analytics" alt="Typing SVG" />
  </a>

  <br />

  <!-- Badges -->
  <a href="https://github.com/HuLaxx/Sentinance">
    <img src="https://img.shields.io/github/stars/HuLaxx/Sentinance?style=for-the-badge&logo=github&color=0f172a" alt="Stars" />
  </a>
  <a href="https://github.com/HuLaxx/Sentinance/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/HuLaxx/Sentinance?style=for-the-badge&logo=github&color=0f172a" alt="Contributors" />
  </a>
  <br />
  <img src="https://img.shields.io/badge/Next.js_16-black?style=for-the-badge&logo=next.js&logoColor=white" />
  <img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" />

  <br /><br />
  
  <!-- Buttons -->
  <a href="http://localhost:3000/demo">
    <img src="https://img.shields.io/badge/🚀_Launch_Demo-0ea5e9?style=for-the-badge&height=40" alt="Launch Demo" />
  </a>
  <a href="docs/TECH_STACK.md">
    <img src="https://img.shields.io/badge/🛠️_Tech_Stack-blueviolet?style=for-the-badge&height=40" alt="Tech Stack" />
  </a>
  <a href="docs/WORKFLOW_MAP.md">
    <img src="https://img.shields.io/badge/🗺️_Architecture-10b981?style=for-the-badge&height=40" alt="Architecture" />
  </a>
</div>

<hr />

## 📖 About

**Sentinance** is a production-ready crypto market intelligence platform featuring:

- 🤖 **Autonomous AI Agents** — LangGraph multi-agent system that researches, analyzes, and debates market conditions 24/7
- 🔮 **ML Price Predictions** — LSTM neural networks with momentum models for short/medium/long term forecasting
- 📡 **Real-Time Streaming** — WebSocket + SSE for millisecond-latency price updates across 30+ global assets
- 🚨 **Anomaly Detection** — Automatic alerts for price spikes, volume surges, and market manipulation patterns
- 💬 **AI Chat Interface** — Natural language market analysis powered by Gemini/Groq with tool calling

---

## ⚡ Cinematic Features

<div align="center">
<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://capsule-render.vercel.app/api?type=rect&color=0ea5e9&height=50&text=🤖%20Autonomous%20Agents&fontSize=14&fontColor=ffffff" width="100%" />
      <br /><br />
      <b>LangGraph Multi-Agent</b><br />
      <sub>Researchers & analysts working 24/7</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://capsule-render.vercel.app/api?type=rect&color=7c3aed&height=50&text=🔮%20ML%20Predictions&fontSize=14&fontColor=ffffff" width="100%" />
      <br /><br />
      <b>LSTM + Momentum</b><br />
      <sub>Forecasts: 4h, 24h, 7d horizons</sub>
    </td>
    <td align="center" width="33%">
      <img src="https://capsule-render.vercel.app/api?type=rect&color=10b981&height=50&text=📡%20Live%20Streaming&fontSize=14&fontColor=ffffff" width="100%" />
      <br /><br />
      <b>WebSocket & SSE</b><br />
      <sub>Millisecond-latency global feeds</sub>
    </td>
  </tr>
</table>
</div>

---

## 🛠️ Tech Stack

<div align="center">
  <img src="https://skillicons.dev/icons?i=nextjs,react,ts,tailwind,python,fastapi,postgres,redis,docker,kubernetes,kafka,grafana,prometheus,pytorch,gcp&perline=15" />
</div>

<br />

| Category | Technologies |
|----------|--------------|
| **Frontend** | Next.js 16, React 18, TypeScript, TailwindCSS, Zod |
| **Backend** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **AI/ML** | LangGraph, Google Gemini, Groq, PyTorch LSTM, SHAP |
| **Data** | PostgreSQL, Redis, Qdrant, Kafka, dbt |
| **Infrastructure** | Docker, Kubernetes, GitHub Actions, Prometheus, Grafana |

> **[📚 View Full Tech Stack Document](docs/TECH_STACK.md)** — 30+ technologies with fallback status

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/HuLaxx/Sentinance.git
cd sentinance
cp .env.example .env
# Edit .env with your API keys (Gemini, Groq, etc.)
```

### 2️⃣ Run with Docker (Recommended)

```bash
docker-compose up -d --build
```

### 3️⃣ Or Run Locally

**Backend (FastAPI)**
```bash
cd apps/api
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

**Frontend (Next.js)**
```bash
cd apps/web
npm install
npm run dev
```

### 4️⃣ Access the App

| Service | URL |
|---------|-----|
| **Demo Dashboard** | http://localhost:3000/demo |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

---

## 📸 Portfolio Highlights

| Feature | Description | Status |
|---------|-------------|--------|
| **Auto-Agents** | 🧠 LangGraph agents that debate market moves | ✅ Live |
| **RAG Memory** | 📚 Historical learning via Qdrant vector search | ✅ Live |
| **Streaming AI** | ⚡ Token-by-token LLM streaming via SSE | ✅ Live |
| **Anomaly Alerts** | 🚨 Real-time price spike & volume surge detection | ✅ Live |
| **Drift Detection** | 📉 ML model distribution monitoring | ✅ Live |
| **A/B Testing** | 🔬 Model performance comparison framework | ✅ Live |
| **Feature Store** | 📦 Feast-based feature management | ✅ Configured |

---

## 📂 Project Structure

```
sentinance/
├── apps/
│   ├── api/                 # FastAPI backend (60+ modules)
│   │   ├── agent.py         # LangGraph multi-agent system
│   │   ├── llm_wrapper.py   # Gemini/Groq with fallback
│   │   ├── predictor.py     # LSTM price predictions
│   │   ├── indicators.py    # Technical analysis (RSI, MACD, etc.)
│   │   └── streaming_llm.py # SSE token streaming
│   ├── web/                 # Next.js 16 frontend
│   │   ├── src/app/demo/    # Demo dashboard
│   │   └── src/app/asset/   # Asset detail pages
│   ├── ml/                  # ML training & explainability
│   └── orchestration/       # Prefect data pipelines
├── docs/                    # Documentation
│   ├── TECH_STACK.md        # Technology inventory
│   ├── PORTFOLIO_CLAIMS.md  # Skills by role
│   └── WORKFLOW_MAP.md      # Integration diagram
├── infra/                   # Kubernetes, Docker, monitoring
└── notebooks/               # Jupyter EDA notebooks
```

---

## 🧪 Testing

```bash
# Backend tests (126+ passing)
cd apps/api
pytest -v --cov=. --cov-report=html

# Frontend tests
cd apps/web
npm run test
```

**Coverage:** 96%+ on core backend modules

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [TECH_STACK.md](docs/TECH_STACK.md) | Full technology inventory with fallback status |
| [PORTFOLIO_CLAIMS.md](docs/PORTFOLIO_CLAIMS.md) | Skills breakdown by role (Backend, ML, DevOps) |
| [WORKFLOW_MAP.md](docs/WORKFLOW_MAP.md) | Visual architecture & data flow |
| [API.md](docs/API.md) | API endpoint documentation |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0ea5e9&height=100&section=footer&text=Built%20by%20HuLaX&fontSize=24&fontColor=ffffff" width="100%" />
  
  <br />
  
  <a href="https://hulax.vercel.app">Portfolio</a> • 
  <a href="https://github.com/HuLaxx">GitHub</a> • 
  <a href="https://linkedin.com/in/rahul-khanke">LinkedIn</a> • 
  <a href="mailto:rahulkhanke02@gmail.com">Email</a>
</div>
