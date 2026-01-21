"""
Prometheus Metrics Endpoint

Exposes application metrics for monitoring:
- Request counts
- Latency histograms
- Active connections
- Price update frequency
"""

import time
from typing import Callable
from functools import wraps
import structlog

log = structlog.get_logger()

# Check if prometheus is available
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_not_installed", message="Install with: pip install prometheus-client")


# ============================================
# METRICS DEFINITIONS
# ============================================

if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        'sentinance_http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    REQUEST_LATENCY = Histogram(
        'sentinance_http_request_duration_seconds',
        'HTTP request latency',
        ['method', 'endpoint']
    )
    
    # WebSocket metrics
    WEBSOCKET_CONNECTIONS = Gauge(
        'sentinance_websocket_connections_active',
        'Active WebSocket connections'
    )
    
    # Price metrics
    PRICE_UPDATES = Counter(
        'sentinance_price_updates_total',
        'Total price updates fetched',
        ['symbol']
    )
    
    PRICE_FETCH_LATENCY = Histogram(
        'sentinance_price_fetch_duration_seconds',
        'Price fetch latency from exchanges'
    )
    
    # AI metrics
    AI_REQUESTS = Counter(
        'sentinance_ai_requests_total',
        'Total AI chat requests',
        ['model']
    )
    
    AI_LATENCY = Histogram(
        'sentinance_ai_request_duration_seconds',
        'AI request latency'
    )
    
    # Alert metrics
    ALERTS_CREATED = Counter(
        'sentinance_alerts_created_total',
        'Total alerts created'
    )
    
    ALERTS_TRIGGERED = Counter(
        'sentinance_alerts_triggered_total',
        'Total alerts triggered',
        ['symbol', 'alert_type']
    )


# ============================================
# METRIC HELPERS
# ============================================

def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics."""
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_websocket_connect():
    """Record WebSocket connection."""
    if PROMETHEUS_AVAILABLE:
        WEBSOCKET_CONNECTIONS.inc()


def record_websocket_disconnect():
    """Record WebSocket disconnection."""
    if PROMETHEUS_AVAILABLE:
        WEBSOCKET_CONNECTIONS.dec()


def record_price_update(symbol: str, duration: float):
    """Record price update metrics."""
    if PROMETHEUS_AVAILABLE:
        PRICE_UPDATES.labels(symbol=symbol).inc()
        PRICE_FETCH_LATENCY.observe(duration)


def record_ai_request(model: str, duration: float):
    """Record AI request metrics."""
    if PROMETHEUS_AVAILABLE:
        AI_REQUESTS.labels(model=model).inc()
        AI_LATENCY.observe(duration)


def record_alert_created():
    """Record alert creation."""
    if PROMETHEUS_AVAILABLE:
        ALERTS_CREATED.inc()


def record_alert_triggered(symbol: str, alert_type: str):
    """Record alert trigger."""
    if PROMETHEUS_AVAILABLE:
        ALERTS_TRIGGERED.labels(symbol=symbol, alert_type=alert_type).inc()


# ============================================
# TIMING DECORATOR
# ============================================

def timed(metric_name: str = "default"):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                log.debug(f"{metric_name}_duration", duration=duration)
        return wrapper
    return decorator


# ============================================
# METRICS ENDPOINT
# ============================================

def get_metrics():
    """Generate Prometheus metrics output."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(), CONTENT_TYPE_LATEST
    return "Prometheus not available", "text/plain"
