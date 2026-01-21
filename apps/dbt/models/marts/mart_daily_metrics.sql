-- mart_daily_metrics.sql - Daily aggregated metrics

{{
    config(
        materialized='table'
    )
}}

with prices as (
    select * from {{ ref('stg_prices') }}
),

-- First get OHLCV using proper GROUP BY aggregations
daily_ohlcv as (
    select
        symbol,
        date_trunc('day', timestamp) as date,
        
        -- For open/close, we need subqueries or DISTINCT ON
        min(timestamp) as first_timestamp,
        max(timestamp) as last_timestamp,
        
        -- Standard aggregations
        max(high_price) as high,
        min(low_price) as low,
        sum(volume) as volume,
        count(*) as num_trades,
        avg(close_price) as avg_price,
        stddev(close_price) as price_volatility

    from prices
    group by symbol, date_trunc('day', timestamp)
),

-- Get open prices (price at first timestamp of day)
open_prices as (
    select distinct on (symbol, date_trunc('day', timestamp))
        symbol,
        date_trunc('day', timestamp) as date,
        open_price as open
    from prices
    order by symbol, date_trunc('day', timestamp), timestamp asc
),

-- Get close prices (price at last timestamp of day)
close_prices as (
    select distinct on (symbol, date_trunc('day', timestamp))
        symbol,
        date_trunc('day', timestamp) as date,
        close_price as close
    from prices
    order by symbol, date_trunc('day', timestamp), timestamp desc
),

-- Combine OHLCV
daily_agg as (
    select
        d.symbol,
        d.date,
        o.open,
        d.high,
        d.low,
        c.close,
        d.volume,
        d.num_trades,
        d.avg_price,
        d.price_volatility,
        d.high - d.low as daily_range,
        (d.high - d.low) / nullif(d.low, 0) * 100 as range_percent
    from daily_ohlcv d
    left join open_prices o on d.symbol = o.symbol and d.date = o.date
    left join close_prices c on d.symbol = c.symbol and d.date = c.date
),

with_returns as (
    select
        *,
        (close - lag(close) over (partition by symbol order by date)) / 
            nullif(lag(close) over (partition by symbol order by date), 0) * 100 as daily_return,
        (close - first_value(close) over (
            partition by symbol 
            order by date 
            rows between 6 preceding and current row
        )) / nullif(first_value(close) over (
            partition by symbol 
            order by date 
            rows between 6 preceding and current row
        ), 0) * 100 as weekly_return
    from daily_agg
)

select * from with_returns
