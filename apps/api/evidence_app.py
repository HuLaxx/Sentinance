"""Local, isolated evidence-only ASGI application for fixture demonstrations.

No background provider calls, Redis connections, chat models or streaming.
Use loopback only; normal main.py includes the same route separately.
"""

from fastapi import FastAPI

from routers.forecast_evidence_router import router as forecast_evidence_router
from routers.market_event_router import router as market_evidence_router

app = FastAPI(
    title="Sentinance Local Stored Market Evidence",
    description="Read-only fixture/provider-labelled persisted price events",
)
app.include_router(market_evidence_router)
app.include_router(forecast_evidence_router)
