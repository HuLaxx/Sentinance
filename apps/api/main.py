"""
Sentinance API - Main Entry Point

This is the FastAPI application that serves as the back for Sentinance.
It provides REST endpoints for prices, predictions, and WebSocket for real-time streaming.
"""

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import structlog

# New Imports
from core.redis import init_redis, close_redis
from managers.connection_manager import manager
from services.price_stream import price_stream_task, CURRENT_PRICES, PRICE_HISTORY, MAX_HISTORY_POINTS, fetch_and_update_prices

# ============================================
# LOGGING SETUP
# ============================================
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
log = structlog.get_logger()


# ============================================
# LIFESPAN - Startup & Shutdown
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at application startup and shutdown."""
    log.info("Starting Sentinance API...")
    
    # 1. Initialize Redis
    try:
        await init_redis()
    except Exception:
        log.warning("redis_init_failed_running_in_degraded_mode")
    
    # 2. Start WebSocket Manager Listener (Redis Sub)
    await manager.start_listener()
    
    # 3. Start background price fetching & publishing
    stream_task = asyncio.create_task(price_stream_task())
    log.info("Price streaming service started")
    
    yield
    
    # Shutdown
    stream_task.cancel()
    await manager.stop_listener()
    await close_redis()
    log.info("Shutting down Sentinance API...")


# ============================================
# CREATE FASTAPI APP
# ============================================
app = FastAPI(
    title="Sentinance API",
    description="Real-Time Crypto Market Intelligence Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Include routers
from routers.auth_router import router as auth_router
app.include_router(auth_router)


# ============================================
# CORS MIDDLEWARE
# ============================================
import os

# Build allowed origins list from environment
_allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add production frontend URL if configured
_frontend_url = os.getenv("FRONTEND_URL", "")
if _frontend_url:
    _allowed_origins.append(_frontend_url)

# Add any additional comma-separated origins from env
_extra_origins = os.getenv("CORS_ORIGINS", "")
if _extra_origins:
    _allowed_origins.extend([o.strip() for o in _extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)


# ============================================
# WEBSOCKET ENDPOINT
# ============================================
@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price streaming.
    Now backed by Redis Pub/Sub for horizontal scaling.
    """
    await manager.connect(websocket)
    
    # Send initial prices immediately from local cache
    if CURRENT_PRICES:
        await websocket.send_json({
            "type": "initial",
            "prices": list(CURRENT_PRICES.values())
        })
    else:
        prices = await fetch_and_update_prices()
        await websocket.send_json({
            "type": "initial",
            "prices": prices
        })
    
    try:
        while True:
            # Keep connection alive, wait for client messages
            data = await websocket.receive_text()
            
            # Handle client commands
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================
@app.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "ok", 
        "service": "sentinance-api",
        "websocket_connections": len(manager.active_connections)
    }


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    return {"status": "ready"}


# Kubernetes-style health checks (aliases for container orchestration)
@app.get("/healthz")
async def healthz():
    """Kubernetes liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Kubernetes readiness probe endpoint."""
    return {"status": "ready"}


# ============================================
# PROMETHEUS METRICS
# ============================================
try:
    from metrics import get_metrics
    from fastapi.responses import Response
    
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint for monitoring."""
        content, content_type = get_metrics()
        return Response(content=content, media_type=content_type)
except ImportError:
    pass


# ============================================
# PRICE ENDPOINTS (REST)
# ============================================

@app.get("/api/prices/{symbol}")
async def get_price(symbol: str):
    """Get the latest price for a symbol."""
    symbol_upper = symbol.upper()
    if symbol_upper not in CURRENT_PRICES:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
    return CURRENT_PRICES[symbol_upper]


@app.get("/api/prices")
async def list_prices():
    """Get prices for all supported symbols."""
    return {"prices": list(CURRENT_PRICES.values())}


@app.get("/api/prices/{symbol}/history")
async def get_price_history(
    symbol: str, 
    limit: int = Query(default=50, ge=1, le=500, description="Number of history points to return")
):
    """Get historical prices for a symbol."""
    symbol_upper = symbol.upper()
    
    if symbol_upper not in PRICE_HISTORY:
        return {"symbol": symbol_upper, "history": [], "count": 0}
    
    history = PRICE_HISTORY[symbol_upper][-min(limit, MAX_HISTORY_POINTS):]
    return {
        "symbol": symbol_upper,
        "history": history,
        "count": len(history),
    }


# ============================================
# AI CHAT ENDPOINTS
# ============================================
from pydantic import BaseModel, Field
from typing import List
from ai_chat import get_ai_response, get_suggested_questions
from dependencies import get_current_user, get_current_user_optional

# Import LangGraph agent
try:
    from agent import run_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    log.warning("agent_not_available", message="LangGraph agent not loaded")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    use_agent: bool = True

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """AI Chat endpoint."""
    from datetime import datetime

    if request.use_agent and AGENT_AVAILABLE:
        try:
            agent_result = await run_agent(request.message)
            return {
                "role": "assistant",
                "content": agent_result.get("analysis", ""),
                "metadata": {
                    "model": "langgraph-agent",
                    "confidence": agent_result.get("confidence", 0),
                    "plan": agent_result.get("plan", []),
                    "timestamp": agent_result.get("timestamp") or datetime.utcnow().isoformat(),
                }
            }
        except Exception as e:
            log.error("agent_failed", error=str(e))

    error_msg = None
    try:
        from enhanced_ai import chat_with_ai

        ai_response = await chat_with_ai(
            message=request.message,
            current_prices=CURRENT_PRICES,
            price_history=PRICE_HISTORY
        )

        if not ai_response.get("error"):
            return {
                "role": "assistant",
                "content": ai_response["content"],
                "metadata": {
                    "model": ai_response.get("model", "gemini-2.5-flash"),
                    "confidence": 0.92,
                    "context_used": ai_response.get("context_used", False),
                    "data_sources": ai_response.get("data_sources", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            }

        error_msg = ai_response.get("content", "AI service error")
    except Exception as e:
        error_msg = f"AI Error: {str(e)}"
        log.error("enhanced_ai_failed", error=str(e))

    mock = get_ai_response(request.message, CURRENT_PRICES)
    return {
        "role": mock.get("role", "assistant"),
        "content": mock.get("content", ""),
        "metadata": mock.get("metadata", {"model": "mock"}),
    }

@app.get("/api/chat/suggestions")
async def chat_suggestions():
    return {"suggestions": get_suggested_questions()}


# ============================================
# ALERTS ENDPOINTS
# ============================================
from alerts_service import (
    get_alerts_service, 
    CreateAlertRequest
)

@app.post("/api/alerts")
async def create_alert(
    request: CreateAlertRequest,
    user=Depends(get_current_user_optional),
):
    service = get_alerts_service()
    user_id = user.sub if user else "anonymous"
    alert = service.create_alert(
        user_id=user_id,
        symbol=request.symbol,
        alert_type=request.alert_type,
        target_value=request.target_value,
        message=request.message
    )
    return alert


@app.get("/api/alerts")
async def list_alerts(user=Depends(get_current_user)):
    service = get_alerts_service()
    alerts = service.get_alerts(user_id=user.sub)
    return {"alerts": [a.model_dump() for a in alerts]}


@app.get("/api/alerts/active")
async def list_active_alerts(user=Depends(get_current_user)):
    service = get_alerts_service()
    alerts = service.get_active_alerts()
    alerts = [a for a in alerts if a.user_id == user.sub]
    return {"alerts": [a.model_dump() for a in alerts]}


@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: str, user=Depends(get_current_user)):
    service = get_alerts_service()
    user_alerts = service.get_alerts(user_id=user.sub)
    if not any(a.id == alert_id for a in user_alerts):
        raise HTTPException(status_code=403, detail="Not authorized")
    success = service.delete_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted", "id": alert_id}


# ============================================
# NEWS & STATS ENDPOINTS
# ============================================
try:
    from news_scraper import get_latest_news, get_news_by_topic
    
    @app.get("/api/news")
    async def list_news(limit: int = 20):
        news = await get_latest_news(limit=min(limit, 50))
        return {"news": news, "count": len(news)}
    
    @app.get("/api/news/{topic}")
    async def news_by_topic(topic: str, limit: int = 10):
        news = await get_news_by_topic(topic, limit=min(limit, 20))
        return {"topic": topic, "news": news, "count": len(news)}
except ImportError:
    pass

try:
    from market_stats import calculate_market_stats, get_top_movers
    
    @app.get("/api/stats")
    async def market_statistics():
        stats = calculate_market_stats(CURRENT_PRICES)
        return stats.to_dict()
    
    @app.get("/api/stats/movers")
    async def top_movers(limit: int = 5):
        movers = get_top_movers(CURRENT_PRICES, limit=min(limit, 10))
        return movers
except ImportError:
    pass


# ============================================
# TECHNICAL INDICATORS ENDPOINTS
# ============================================
try:
    from indicators import calculate_all_indicators
    from predictor import generate_prediction
    
    @app.get("/api/indicators/{symbol}")
    async def get_indicators(symbol: str):
        symbol_upper = symbol.upper()
        history = PRICE_HISTORY.get(symbol_upper, [])
        prices = [h["price"] for h in history]
        if not prices:
            prices = [CURRENT_PRICES.get(symbol_upper, {}).get("price", 0)]
        
        indicators = calculate_all_indicators(symbol_upper, prices)
        return indicators.to_dict()
    
    @app.get("/api/predict/{symbol}")
    async def get_prediction(symbol: str, horizon: str = "24h", model: str = "ensemble"):
        symbol_upper = symbol.upper()
        history = PRICE_HISTORY.get(symbol_upper, [])
        prices = [h["price"] for h in history]
        current = CURRENT_PRICES.get(symbol_upper, {}).get("price", 0)
        
        horizon_seconds = {
            "1h": 60 * 60,
            "24h": 24 * 60 * 60,
            "7d": 7 * 24 * 60 * 60,
        }
        if horizon not in horizon_seconds:
            raise HTTPException(status_code=400, detail="Invalid horizon")

        if not prices:
            prices = [current] if current else [0]
        
        prediction = generate_prediction(symbol_upper, prices, current, horizon, model)
        return prediction.to_dict()

except ImportError:
    pass


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
async def root():
    return {
        "name": "Sentinance API",
        "version": "0.1.0",
        "docs": "/docs",
        "websocket": "/ws/prices",
        "chat": "/api/chat",
        "alerts": "/api/alerts",
        "news": "/api/news",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
