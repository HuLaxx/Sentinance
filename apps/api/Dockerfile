# ============================================
# SENTINANCE API - DOCKERFILE
# ============================================
# Multi-stage build for smaller image size
#
# Build:  docker build -t sentinance-api .
# Run:    docker run -p 8000:8000 sentinance-api

# -----------------
# Stage 1: Builder
# -----------------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements-railway.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


# -----------------
# Stage 2: Runtime
# -----------------
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, httpx; port = os.getenv('PORT', '8000'); httpx.get(f'http://localhost:{port}/health')" || exit 1

# Run the application (supports $PORT for Railway/Render)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
