<h1 align="center">
  <br>
  <img src="apps/web/public/icon.svg" alt="Sentinance" width="120">
  <br>
  <br>
  <strong>SENTINANCE</strong>
  <br>
</h1>

<p align="center">
  <strong>Real-Time Crypto Market Intelligence Platform</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-16.1-black?style=for-the-badge&logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Gemini_AI-Powered-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/LangGraph-Agents-FF4081?style=for-the-badge" alt="LangGraph">
  <img src="https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Tests-126%20Passing-success?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/Coverage-96%25-brightgreen?style=for-the-badge" alt="Coverage">
  <img src="https://img.shields.io/badge/Build-Passing-success?style=for-the-badge" alt="Build">
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-api">API</a>
</p>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">
</div>

## ✨ Features

<table>
<tr>
<td width="50%">

### 📊 Real-Time Streaming
Live WebSocket feeds for **crypto** (BTC, ETH, SOL, XRP) and **global indices** (S&P 500, Nifty 50, FTSE 100, Nikkei 225)

</td>
<td width="50%">

### 🤖 Agentic AI
LangGraph multi-agent system with **Gemini AI** for autonomous market analysis

</td>
</tr>
<tr>
<td width="50%">

### 📈 ML Predictions
**LSTM neural networks** for price forecasting with confidence intervals

</td>
<td width="50%">

### ⚡ Anomaly Detection
Real-time alerts for price spikes, volume surges, and manipulation patterns

</td>
</tr>
<tr>
<td width="50%">

### 🔍 RAG Pipeline
**Qdrant** vector store + semantic search for intelligent insights

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

- Docker Desktop
- Node.js 18+
- Python 3.11+

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/HuLaxx/Sentinance.git
cd sentinance
cp .env.example .env
```

### 2️⃣ Start Infrastructure

```bash
docker compose -f docker-compose.dev.yml up -d
```

### 3️⃣ Start Backend

```bash
cd apps/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### 4️⃣ Start Frontend

```bash
cd apps/web
npm install
npm run dev
```

### 5️⃣ Open

| Service | URL |
|---------|-----|
| 🌐 Frontend | http://localhost:3000 |
| 🔌 API Docs | http://localhost:8001/docs |
| 📊 Grafana | http://localhost:3001 |

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

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws/prices` | Real-time price streaming |
| `GET` | `/api/prices` | All current prices |
| `GET` | `/api/prices/{symbol}` | Single asset price |
| `POST` | `/api/chat` | AI chat with market context |
| `GET` | `/api/predict/{symbol}` | ML price prediction |
| `GET` | `/api/indicators/{symbol}` | Technical indicators |
| `POST` | `/api/alerts` | Create price alert |
| `GET` | `/api/news` | Latest market news |

---

## 🧪 Testing

```bash
cd apps/api
$env:JWT_SECRET="your-secret-key"
pytest tests/ -v --cov=. --cov-report=html
```

**126 tests passing** with comprehensive coverage:

| Category | Tests | Coverage |
|----------|-------|----------|
| Unit - Indicators | 29 | 96% |
| Unit - Predictor | 22 | 94% |
| Integration - API | 18 | 88% |
| Auth & Alerts | 20 | 85% |
| WebSocket & Chat | 37 | 72% |

### Recent Fixes ✅
- MACD signal line now uses proper 9-period EMA
- ZeroDivisionError handling in predictor
- Kubernetes health probes (`/healthz`, `/ready`)

---

## 📊 Tech Stack

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
Pydantic<br>
AsyncIO

</td>
<td align="center" width="20%">

**AI/ML**

LangGraph<br>
Gemini AI<br>
PyTorch LSTM<br>
Qdrant

</td>
<td align="center" width="20%">

**Data**

PostgreSQL<br>
Redis<br>
Kafka<br>
WebSocket

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

---

## 📁 Project Structure

```
sentinance/
├── apps/
│   ├── api/          # FastAPI backend
│   ├── web/          # Next.js frontend
│   └── ml/           # ML models
├── infra/
│   ├── k8s/          # Kubernetes manifests
│   └── monitoring/   # Prometheus/Grafana
├── docs/             # Documentation
└── docker-compose.yml
```

---

## 📜 License

MIT © 2026 Sentinance

---

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" alt="line" width="100%">
  <br>
  <br>
  <strong>Built with ❤️ by <a href="https://github.com/Hulaxx">HuLaX</a></strong>
  <br>
  <br>
  <a href="#top">⬆️ Back to Top</a>
</div>
