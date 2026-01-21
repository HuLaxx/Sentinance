# Production Setup Guide

## What You Need to Install/Setup

### Option A: Docker (Recommended - Easiest)

Run this single command to start PostgreSQL, Redis, and Kafka:

```powershell
cd "d:\Portfolio Projetcs\project 1\sentinance"
docker compose -f docker-compose.dev.yml up -d
```

This starts:
- PostgreSQL on port 5432
- Redis on port 6379
- Kafka on port 9092

---

### Option B: Manual Installation

#### 1. PostgreSQL

**Install:**
- Download from: https://www.postgresql.org/download/windows/
- Or use Chocolatey: `choco install postgresql`

**Create Database:**
```sql
CREATE DATABASE sentinance;
CREATE USER sentinance WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE sentinance TO sentinance;
```

**Connection String:**
```
DATABASE_URL=postgresql://sentinance:password@localhost:5432/sentinance
```

---

#### 2. Redis (Optional - for caching)

**Install:**
- Download from: https://github.com/microsoftarchive/redis/releases
- Or use Docker: `docker run -d -p 6379:6379 redis:alpine`

**Connection String:**
```
REDIS_URL=redis://localhost:6379
```

---

#### 3. Kafka (Optional - for event streaming)

**Install via Docker (easiest):**
```powershell
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
```

**Connection String:**
```
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

---

## Environment Variables

Create a `.env` file in `apps/api/` with:

```env
# Required
GEMINI_API_KEY=your_api_key_here

# Database (optional - uses in-memory if not set)
DATABASE_URL=postgresql://sentinance:password@localhost:5432/sentinance

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Kafka (optional)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# JWT
JWT_SECRET=your-secret-key-here-change-in-production
```

---

## Quick Start (After Setup)

```powershell
# Terminal 1: Start backend
cd apps/api
python main.py

# Terminal 2: Start frontend
cd apps/web
npm run dev
```

---

## Verification

1. Open http://localhost:8000/health - Should show "healthy"
2. Open http://localhost:3000 - Dashboard should load
3. Open http://localhost:3000/chat - AI should respond
