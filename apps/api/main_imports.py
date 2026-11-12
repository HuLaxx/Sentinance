"""
Sentinance API - Main Entry Point

This is the FastAPI application that serves as the backend for Sentinance.
It provides REST endpoints for prices, predictions, and WebSocket for real-time streaming.
"""

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

import structlog

# ============================================
# LOGGING SETUP
# ============================================
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger()
