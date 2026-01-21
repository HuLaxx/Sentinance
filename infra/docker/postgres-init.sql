-- ============================================
-- SENTINANCE DATABASE INITIALIZATION
-- ============================================
-- This script runs when PostgreSQL container starts
-- Creates initial tables and extensions

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- Text search

-- ============================================
-- PRICES TABLE
-- ============================================
-- Stores real-time price data from exchanges
-- Partitioned by day for efficient queries

CREATE TABLE IF NOT EXISTS prices (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,           -- e.g., 'BTCUSDT'
    price DOUBLE PRECISION NOT NULL,       -- High precision for crypto
    volume DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    change_24h DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Index for fast lookups
    CONSTRAINT prices_positive CHECK (price > 0)
);

-- Index for common query: latest price by symbol
CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts 
ON prices (symbol, timestamp DESC);

-- ============================================
-- USERS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================
-- ALERTS TABLE
-- ============================================
-- User-defined price alerts
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(20) NOT NULL,        -- price_above, price_below, percent_change
    target_value DECIMAL(18, 8) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    message TEXT,
    triggered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_user_symbol
ON alerts (user_id, symbol);

CREATE INDEX IF NOT EXISTS idx_alerts_status
ON alerts (status);


-- ============================================
-- AUDIT_LOGS TABLE
-- ============================================
-- Immutable record of all AI decisions (compliance)
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(36),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(36),
    old_values TEXT,
    new_values TEXT,
    ip_address VARCHAR(45),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_user 
ON audit_logs (user_id);

CREATE INDEX IF NOT EXISTS idx_audit_entity
ON audit_logs (entity_type, entity_id);


-- ============================================
-- PREDICTIONS TABLE
-- ============================================
-- Store ML model predictions for analysis
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    prediction_type VARCHAR(50) NOT NULL,   -- price_direction, sentiment
    prediction_value DECIMAL(18, 8) NOT NULL,
    confidence DECIMAL(5, 4),               -- 0.0000 to 1.0000
    horizon VARCHAR(20),                    -- 1h, 4h, 24h
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol 
ON predictions (symbol, created_at DESC);


-- ============================================
-- Grant permissions
-- ============================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sentinance;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sentinance;

-- Log success
DO $$
BEGIN
    RAISE NOTICE 'Sentinance database initialized successfully!';
END $$;
