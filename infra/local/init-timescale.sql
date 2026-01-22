-- TimescaleDB Initialization Script for Sentinance
-- Creates hypertables for time-series price data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================
-- PRICE HISTORY HYPERTABLE
-- ============================================
CREATE TABLE IF NOT EXISTS price_history (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    open DOUBLE PRECISION,
    source TEXT DEFAULT 'binance'
);

-- Convert to hypertable (chunk by time, 1 day chunks)
SELECT create_hypertable('price_history', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_price_history_symbol ON price_history (symbol, time DESC);

-- ============================================
-- CONTINUOUS AGGREGATES (Pre-computed rollups)
-- ============================================

-- Hourly OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS price_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    first(price, time) as open,
    max(price) as high,
    min(price) as low,
    last(price, time) as close,
    sum(volume) as volume,
    count(*) as trades
FROM price_history
GROUP BY bucket, symbol
WITH NO DATA;

-- Daily OHLCV
CREATE MATERIALIZED VIEW IF NOT EXISTS price_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    symbol,
    first(price, time) as open,
    max(price) as high,
    min(price) as low,
    last(price, time) as close,
    sum(volume) as volume,
    count(*) as trades
FROM price_history
GROUP BY bucket, symbol
WITH NO DATA;

-- ============================================
-- ALERTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    target_value DOUBLE PRECISION NOT NULL,
    message TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    triggered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts (user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts (symbol, is_active);

-- ============================================
-- USERS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_premium BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

-- ============================================
-- RETENTION POLICY (Optional: Keep 1 year of tick data)
-- ============================================
-- SELECT add_retention_policy('price_history', INTERVAL '1 year');

-- ============================================
-- REFRESH POLICIES FOR CONTINUOUS AGGREGATES
-- ============================================
SELECT add_continuous_aggregate_policy('price_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

SELECT add_continuous_aggregate_policy('price_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sentinance;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sentinance;
