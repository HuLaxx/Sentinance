# ============================================
# SENTINANCE - MAKEFILE
# ============================================
# Convenient commands for development
#
# Usage:
#   make dev        - Start local development servers
#   make docker-up  - Start Docker stack
#   make docker-down- Stop Docker stack
#   make test       - Run tests
#   make lint       - Run linters

.PHONY: help dev api web docker-up docker-down docker-logs test lint clean

# Default target
help:
	@echo "Sentinance Development Commands"
	@echo "================================"
	@echo "  make dev          - Start local development (API + Web)"
	@echo "  make api          - Start API server only"
	@echo "  make web          - Start Web server only"
	@echo "  make docker-up    - Start full Docker stack"
	@echo "  make docker-down  - Stop Docker stack"
	@echo "  make docker-logs  - View Docker logs"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linters"
	@echo "  make clean        - Clean build artifacts"

# ==========================================
# LOCAL DEVELOPMENT
# ==========================================

# Start both API and Web in separate terminals
dev:
	@echo "Starting development servers..."
	@echo "API: http://localhost:8000"
	@echo "Web: http://localhost:3000"
	@echo ""
	@echo "Run 'make api' and 'make web' in separate terminals"

# Start API server
api:
	cd apps/api && python main.py

# Start Web server
web:
	cd apps/web && npm run dev

# Install all dependencies
install:
	cd apps/api && pip install -r requirements.txt
	cd apps/web && npm install

# ==========================================
# DOCKER
# ==========================================

# Start Docker stack
docker-up:
	docker-compose -f docker-compose.dev.yml up -d
	@echo ""
	@echo "Services started:"
	@echo "  API:           http://localhost:8000"
	@echo "  Web:           http://localhost:3000"
	@echo "  Swagger:       http://localhost:8000/docs"
	@echo "  Redpanda UI:   http://localhost:8080"
	@echo "  PostgreSQL:    localhost:5432"
	@echo "  Redis:         localhost:6379"

# Stop Docker stack
docker-down:
	docker-compose -f docker-compose.dev.yml down

# View Docker logs
docker-logs:
	docker-compose -f docker-compose.dev.yml logs -f

# Rebuild Docker images
docker-build:
	docker-compose -f docker-compose.dev.yml build --no-cache

# Clean Docker volumes
docker-clean:
	docker-compose -f docker-compose.dev.yml down -v

# ==========================================
# INFRASTRUCTURE ONLY
# ==========================================

# Start only infrastructure (DB, Redis, Kafka)
infra-up:
	docker-compose -f infra/docker-compose.yml up -d

infra-down:
	docker-compose -f infra/docker-compose.yml down

# ==========================================
# TESTING
# ==========================================

# Run all tests
test:
	cd apps/api && pytest -v
	cd apps/web && npm run test

# Run API tests only
test-api:
	cd apps/api && pytest -v

# Run Web tests only
test-web:
	cd apps/web && npm run test

# ==========================================
# LINTING
# ==========================================

# Run all linters
lint:
	cd apps/api && ruff check . && mypy .
	cd apps/web && npm run lint

# Fix linting issues
lint-fix:
	cd apps/api && ruff check --fix .
	cd apps/web && npm run lint -- --fix

# ==========================================
# CLEANING
# ==========================================

# Clean build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".next" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts"
