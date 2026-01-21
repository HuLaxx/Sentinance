"""
Sentinance API - Main Entry Point

This is the FastAPI application that serves as the backend for Sentinance.
It provides REST endpoints for prices, predictions, and WebSocket for real-time streaming.
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import asyncio
import random
from contextlib import asynccontextmanager
from typing import Set
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog
from auth import verify_token

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
