-- mart_technical_indicators.sql - Technical indicators computed via dbt

{{
    config(
        materialized='table'
    )
}}

with prices as (
    select * from {{ ref('stg_prices') }}
),

with_sma as (
    select
        symbol,
        timestamp,
        close_price,
        volume,
        
        -- SMA calculations
        avg(close_price) over (
            partition by symbol 
            order by timestamp 
            rows between 19 preceding and current row
        ) as sma_20,
        
        avg(close_price) over (
            partition by symbol 
            order by timestamp 
            rows between 49 preceding and current row
        ) as sma_50,
        
        avg(close_price) over (
            partition by symbol 
            order by timestamp 
            rows between 199 preceding and current row
        ) as sma_200,
        
        -- Price change for RSI
        close_price - lag(close_price) over (
            partition by symbol order by timestamp
        ) as price_change

    from prices
),

with_rsi_components as (
    select
        *,
        case when price_change > 0 then price_change else 0 end as gain,
        case when price_change < 0 then abs(price_change) else 0 end as loss
    from with_sma
),

with_rsi as (
    select
        symbol,
        timestamp,
        close_price,
        volume,
        sma_20,
        sma_50,
        sma_200,
        
        -- RSI 14
        case 
            when avg(loss) over (
                partition by symbol 
                order by timestamp 
                rows between 13 preceding and current row
            ) = 0 then 100
            else 100 - (100 / (1 + (
                avg(gain) over (
                    partition by symbol 
                    order by timestamp 
                    rows between 13 preceding and current row
                ) / 
                nullif(avg(loss) over (
                    partition by symbol 
                    order by timestamp 
                    rows between 13 preceding and current row
                ), 0)
            )))
        end as rsi_14,
        
        -- Bollinger Bands
        sma_20 as bb_middle,
        sma_20 + (2 * stddev(close_price) over (
            partition by symbol 
            order by timestamp 
            rows between 19 preceding and current row
        )) as bb_upper,
        sma_20 - (2 * stddev(close_price) over (
            partition by symbol 
            order by timestamp 
            rows between 19 preceding and current row
        )) as bb_lower

    from with_rsi_components
),

final as (
    select
        symbol,
        timestamp,
        close_price,
        volume,
        sma_20,
        sma_50,
        sma_200,
        rsi_14,
        bb_upper,
        bb_middle,
        bb_lower,
        
        -- Trend signals
        case 
            when sma_20 > sma_50 and sma_50 > sma_200 then 'bullish'
            when sma_20 < sma_50 and sma_50 < sma_200 then 'bearish'
            else 'neutral'
        end as trend,
        
        -- RSI signals
        case 
            when rsi_14 > 70 then 'overbought'
            when rsi_14 < 30 then 'oversold'
            else 'neutral'
        end as rsi_signal,
        
        -- BB signals
        case 
            when close_price > bb_upper then 'above_band'
            when close_price < bb_lower then 'below_band'
            else 'within_band'
        end as bb_signal

    from with_rsi
)

select * from final
