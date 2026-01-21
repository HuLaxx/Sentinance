# 🚀 Free Deployment Guide: Sentinance on Vercel

**Complete step-by-step guide to deploy Sentinance using free tiers**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        VERCEL (Free)                        │
│                    Next.js Frontend                         │
│                   https://sentinance.vercel.app             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAILWAY / RENDER (Free)                  │
│                    FastAPI Backend                          │
│                https://sentinance-api.up.railway.app        │
└───────────────────────────┬─────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │ Supabase │   │  Upstash │   │  Qdrant  │
      │ PostgreSQL│   │  Redis   │   │  Cloud   │
      │  (Free)  │   │  (Free)  │   │  (Free)  │
      └──────────┘   └──────────┘   └──────────┘
```

---

## Step 1: Prepare the Project

### 1.1 Create Vercel Config for Frontend

Create `apps/web/vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs"
}
```

### 1.2 Update Environment Variables in Frontend

Create `apps/web/.env.production`:

```bash
NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app
NEXT_PUBLIC_WS_URL=wss://your-backend.up.railway.app/ws/prices
```

---

## Step 2: Deploy Frontend to Vercel (FREE)

### 2.1 Sign Up for Vercel

1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub (free account)

### 2.2 Connect Your Repository

1. Click **"Add New Project"**
2. Connect your GitHub repository
3. Select the repository containing Sentinance

### 2.3 Configure Build Settings

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `apps/web` |
| **Build Command** | `npm run build` |
| **Output Directory** | `.next` |

### 2.4 Add Environment Variables

Click **"Environment Variables"** and add:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://your-backend.up.railway.app` (add after backend deployment) |
| `NEXT_PUBLIC_WS_URL` | `wss://your-backend.up.railway.app/ws/prices` |

### 2.5 Deploy

Click **"Deploy"** and wait for the build to complete.

Your frontend will be live at: `https://sentinance.vercel.app`

---

## Step 3: Deploy Backend to Railway (FREE)

### 3.1 Sign Up for Railway

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub (free $5/month credit)

### 3.2 Create New Project

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your Sentinance repository

### 3.3 Configure the Service

1. Set **Root Directory**: `apps/api`
2. Add a **Dockerfile** (if not exists):

Create `apps/api/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.4 Add Environment Variables in Railway

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | (from Supabase, Step 4) |
| `REDIS_URL` | (from Upstash, Step 5) |
| `JWT_SECRET` | `your-random-32-char-secret` |
| `GEMINI_API_KEY` | Your Google AI API key |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Same as above |
| `QDRANT_URL` | (from Qdrant Cloud, Step 6) |

### 3.5 Deploy

Railway will automatically deploy. Your backend will be at:
`https://sentinance-api.up.railway.app`

---

## Step 4: Set Up Supabase PostgreSQL (FREE)

### 4.1 Create Supabase Account

1. Go to [supabase.com](https://supabase.com)
2. Sign up (free tier: 500MB database)

### 4.2 Create New Project

1. Click **"New Project"**
2. Choose organization
3. Set project name: `sentinance`
4. Set database password (save this!)
5. Select region closest to you

### 4.3 Get Connection String

1. Go to **Settings** → **Database**
2. Copy the **Connection String (URI)**
3. Replace `[YOUR-PASSWORD]` with your actual password

Format:
```
postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
```

For async (add `+asyncpg`):
```
postgresql+asyncpg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
```

### 4.4 Run Migrations

```bash
# Set the DATABASE_URL
export DATABASE_URL="postgresql+asyncpg://postgres:PASSWORD@db.XXX.supabase.co:5432/postgres"

# Run migrations
cd apps/api
alembic upgrade head
```

---

## Step 5: Set Up Upstash Redis (FREE)

### 5.1 Create Upstash Account

1. Go to [upstash.com](https://upstash.com)
2. Sign up (free tier: 10,000 commands/day)

### 5.2 Create Redis Database

1. Click **"Create Database"**
2. Name: `sentinance-redis`
3. Region: Select closest to your Railway region
4. Type: **Regional** (lower latency)

### 5.3 Get Connection URL

1. Go to your database
2. Copy the **REST URL** or **Redis URL**

Format:
```
redis://default:PASSWORD@XXX.upstash.io:6379
```

---

## Step 6: Set Up Qdrant Cloud (FREE)

### 6.1 Create Qdrant Cloud Account

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Sign up (free tier: 1GB storage)

### 6.2 Create Cluster

1. Click **"Create Cluster"**
2. Name: `sentinance`
3. Cloud: Free tier

### 6.3 Get Connection URL

```
https://XXX.qdrant.io:6333
```

Add API key if required.

---

## Step 7: Get Free API Keys

### 7.1 Google Gemini API (FREE)

1. Go to [makersuite.google.com](https://makersuite.google.com)
2. Click **"Get API Key"**
3. Create new API key
4. Copy and save

Free tier: 60 requests/minute

### 7.2 Groq API (FREE - Optional Fallback)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up
3. Generate API key

Free tier: Very generous limits

---

## Step 8: Update Vercel Environment Variables

After backend is deployed, go back to Vercel:

1. Go to your project → **Settings** → **Environment Variables**
2. Update:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://sentinance-api.up.railway.app` |
| `NEXT_PUBLIC_WS_URL` | `wss://sentinance-api.up.railway.app/ws/prices` |

3. Click **Redeploy**

---

## Step 9: Verify Deployment

### 9.1 Test Backend Health

```bash
curl https://sentinance-api.up.railway.app/health
```

Expected:
```json
{"status": "ok", "service": "sentinance-api", "websocket_connections": 0}
```

### 9.2 Test Frontend

Visit: `https://sentinance.vercel.app`

- ✅ Page loads
- ✅ Prices stream in real-time
- ✅ AI Chat responds

---

## Free Tier Limits Summary

| Service | Free Limit |
|---------|-----------|
| **Vercel** | 100GB bandwidth/month, unlimited deploys |
| **Railway** | $5/month credit (~500 hours) |
| **Supabase** | 500MB storage, 2GB bandwidth |
| **Upstash** | 10,000 commands/day |
| **Qdrant** | 1GB storage |
| **Gemini** | 60 requests/minute |

---

## Alternative: Render (FREE)

If Railway doesn't work, use Render:

1. Go to [render.com](https://render.com)
2. Create **Web Service**
3. Connect GitHub repo
4. Set root: `apps/api`
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## Troubleshooting

### CORS Errors
Add your Vercel domain to `main.py` CORS origins:
```python
allow_origins=[
    "https://sentinance.vercel.app",
    "https://your-custom-domain.com",
]
```

### WebSocket Not Connecting
- Ensure backend supports WSS (HTTPS)
- Check if Railway/Render allows WebSocket connections

### Database Connection Errors
- Verify Supabase password is correct
- Check if IP is whitelisted (Supabase → Settings → Database → Network)

---

## 🎉 Congratulations!

Your Sentinance platform is now live and completely FREE!

- **Frontend**: https://sentinance.vercel.app
- **Backend API**: https://sentinance-api.up.railway.app
- **API Docs**: https://sentinance-api.up.railway.app/docs
