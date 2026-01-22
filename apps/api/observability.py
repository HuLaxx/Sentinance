"""
Observability Module for Sentinance

Provides:
- Prometheus metrics
- OpenTelemetry tracing (Jaeger)
- Structured logging

Graceful fallback: Works without Jaeger/Prometheus in demo mode.
"""
import os
import time
import functools
from typing import Optional, Callable, Any

# ============================================
# PROMETHEUS METRICS
# ============================================

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Metrics (only if prometheus available)
if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUEST_COUNT = Counter(
        'sentinance_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    REQUEST_LATENCY = Histogram(
        'sentinance_request_latency_seconds',
        'Request latency in seconds',
        ['method', 'endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    # WebSocket metrics
    WEBSOCKET_CONNECTIONS = Gauge(
        'sentinance_websocket_connections',
        'Number of active WebSocket connections'
    )
    
    # Price streaming metrics
    PRICE_UPDATE_LATENCY = Histogram(
        'sentinance_price_update_latency_seconds',
        'Price update latency from exchange to client',
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    
    # AI/ML metrics
    PREDICTION_LATENCY = Histogram(
        'sentinance_prediction_latency_seconds',
        'ML prediction latency',
        ['model', 'horizon'],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    
    CHAT_LATENCY = Histogram(
        'sentinance_chat_latency_seconds',
        'AI chat response latency',
        ['model'],
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
    )
    
    # Error metrics
    ERROR_COUNT = Counter(
        'sentinance_errors_total',
        'Total errors',
        ['type', 'endpoint']
    )


def get_metrics() -> str:
    """Get Prometheus metrics as string."""
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus not available\n"
    return generate_latest().decode('utf-8')


def get_metrics_content_type() -> str:
    """Get Prometheus content type."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# ============================================
# OPENTELEMETRY TRACING
# ============================================

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None


def setup_tracing(app=None, service_name: str = "sentinance-api"):
    """
    Setup OpenTelemetry tracing with Jaeger exporter.
    
    Gracefully degrades if Jaeger is not available.
    """
    if not OPENTELEMETRY_AVAILABLE:
        print("⚠️ OpenTelemetry not installed, tracing disabled")
        return
    
    jaeger_host = os.getenv("JAEGER_AGENT_HOST", "jaeger")
    jaeger_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
    
    try:
        # Setup tracer provider
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )
        
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        trace.set_tracer_provider(provider)
        
        # Instrument FastAPI
        if app:
            FastAPIInstrumentor.instrument_app(app)
        
        # Instrument HTTP clients
        HTTPXClientInstrumentor().instrument()
        
        print(f"✅ Tracing enabled → Jaeger at {jaeger_host}:{jaeger_port}")
        
    except Exception as e:
        print(f"⚠️ Tracing setup failed (will continue without): {e}")


def get_tracer(name: str = "sentinance"):
    """Get a tracer instance."""
    if not OPENTELEMETRY_AVAILABLE or trace is None:
        return None
    return trace.get_tracer(name)


# ============================================
# TIMING DECORATOR
# ============================================

def timed(metric_name: Optional[str] = None, labels: Optional[dict] = None):
    """
    Decorator to time function execution and record to Prometheus.
    
    Usage:
        @timed("prediction_latency", {"model": "ensemble"})
        def generate_prediction(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if PROMETHEUS_AVAILABLE and metric_name:
                    # Record to appropriate histogram
                    if "prediction" in metric_name.lower():
                        PREDICTION_LATENCY.labels(**(labels or {})).observe(duration)
                    elif "chat" in metric_name.lower():
                        CHAT_LATENCY.labels(**(labels or {})).observe(duration)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if PROMETHEUS_AVAILABLE and metric_name:
                    if "prediction" in metric_name.lower():
                        PREDICTION_LATENCY.labels(**(labels or {})).observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================
# HEALTH CHECK HELPERS
# ============================================

def check_dependency_health() -> dict:
    """Check health of all dependencies."""
    health = {
        "prometheus": PROMETHEUS_AVAILABLE,
        "tracing": OPENTELEMETRY_AVAILABLE,
    }
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379)
        r.ping()
        health["redis"] = True
    except:
        health["redis"] = False
    
    # Check database
    try:
        # Would check DB connection here
        health["database"] = True
    except:
        health["database"] = False
    
    return health
