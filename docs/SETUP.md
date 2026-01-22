# Sentinance - Local Setup Guide

> **Complete guide to run the full production-grade system locally**

---

## 📋 Prerequisites

### Required Software
- **Docker Desktop** (v4.0+) - [Download](https://www.docker.com/products/docker-desktop)
- **Node.js** (v18+) - [Download](https://nodejs.org/)
- **Python** (v3.11+) - [Download](https://www.python.org/)
- **Git** - [Download](https://git-scm.com/)

### System Requirements
- **RAM**: 8GB minimum (16GB recommended for full stack)
- **Disk**: 5GB free space
- **OS**: Windows 10/11, macOS, or Linux

---

## 🚀 Quick Start (5 minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sentinance.git
cd sentinance
```

### 2. Start Infrastructure
```bash
# Start all services (TimescaleDB, Redis, Kafka, Prometheus, Grafana, MLflow)
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Start Backend API
```bash
cd apps/api
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://sentinance:sentinance_prod_2024@localhost:5432/sentinance"
export REDIS_URL="redis://localhost:6379"
export GEMINI_API_KEY="your-gemini-api-key"  # Optional

# Run the API
uvicorn main:app --reload --port 8000
```

### 4. Start Frontend
```bash
cd apps/web
npm install
npm run dev
```

### 5. Access the Application
| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **API Docs** | http://localhost:8000/docs |
| **Grafana** | http://localhost:3001 (admin/admin) |
| **Prometheus** | http://localhost:9090 |
| **MLflow** | http://localhost:5000 |
| **Jaeger** | http://localhost:16686 |

---

## 🔧 Detailed Setup

### Environment Variables

Create `.env` file in root directory:

```bash
# Required
DATABASE_URL=postgresql://sentinance:sentinance_prod_2024@localhost:5432/sentinance
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-super-secret-jwt-key-at-least-32-characters

# Optional (for AI features)
GEMINI_API_KEY=your-gemini-api-key

# Optional (for Kafka)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Optional (for tracing)
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
```

### Database Setup

The database is automatically initialized when Docker starts. To verify:

```bash
# Connect to TimescaleDB
docker exec -it sentinance-timescaledb psql -U sentinance -d sentinance

# Check tables
\dt

# Check hypertable
SELECT * FROM timescaledb_information.hypertables;
```

### Training ML Models

```bash
cd apps/ml

# Train LSTM for Bitcoin
python train_lstm.py --symbol BTCUSDT --epochs 50

# Train all models
python train_lstm.py --all --epochs 30

# View in MLflow
open http://localhost:5000
```

---

## 🧪 Running Tests

```bash
cd apps/api

# Install test dependencies
pip install pytest pytest-cov pytest-asyncio httpx

# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_indicators.py -v
```

---

## 📊 Monitoring

### Grafana Dashboards

1. Open http://localhost:3001
2. Login with `admin` / `admin`
3. Datasources are pre-configured (Prometheus, TimescaleDB)
4. Import dashboards from `infra/local/grafana/provisioning/dashboards/`

### Prometheus Queries

```promql
# Request rate
rate(sentinance_requests_total[5m])

# P99 latency
histogram_quantile(0.99, rate(sentinance_request_latency_seconds_bucket[5m]))

# Error rate
rate(sentinance_errors_total[5m])
```

### Jaeger Tracing

1. Open http://localhost:16686
2. Select service: `sentinance-api`
3. Find traces for specific endpoints

---

## 🛠️ Development

### Hot Reload

Both backend and frontend support hot reload:

```bash
# Backend (auto-restarts on file changes)
uvicorn main:app --reload --port 8000

# Frontend (HMR enabled)
npm run dev
```

### Adding New Indicators

1. Edit `apps/api/indicators.py`
2. Add calculation function
3. Add to `TechnicalIndicators` dataclass
4. Add unit tests in `tests/unit/test_indicators.py`

### Adding New Predictions Models

1. Edit `apps/api/predictor.py`
2. Implement `predict_yourmodel()` function
3. Add to `generate_prediction()` match statement
4. Train and save model in `apps/ml/`

---

## 🐛 Troubleshooting

### Docker Issues

```bash
# Restart all services
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f api

# Reset everything
docker-compose -f docker-compose.prod.yml down -v
docker-compose -f docker-compose.prod.yml up -d
```

### Database Connection Issues

```bash
# Check if TimescaleDB is running
docker ps | grep timescaledb

# Test connection
psql -h localhost -U sentinance -d sentinance
```

### Redis Connection Issues

```bash
# Check Redis
docker exec -it sentinance-redis redis-cli ping
# Should return: PONG
```

---

## 📁 Project Structure

```
sentinance/
├── apps/
│   ├── api/              # FastAPI backend
│   │   ├── main.py       # Entry point
│   │   ├── indicators.py # Technical indicators
│   │   ├── predictor.py  # Price predictions
│   │   ├── agent.py      # LangGraph AI agent
│   │   └── tests/        # Test suite
│   ├── web/              # Next.js frontend
│   └── ml/               # ML training scripts
├── docs/
│   ├── adr/              # Architecture Decision Records
│   └── API.md            # API documentation
├── infra/
│   ├── local/            # Local dev configs
│   └── k8s/              # Kubernetes manifests
├── docker-compose.prod.yml  # Production stack
└── README.md
```

---

## 🚢 Deploying to Production

For production deployment, see:
- Kubernetes: `infra/k8s/`
- CI/CD: `.github/workflows/`
- Configuration: `docs/adr/`

---

**Questions?** Open an issue or check the ADRs in `docs/adr/` for architectural decisions.
