-- dbt models for Sentinance data transformations
-- stg_prices.sql - Staging layer for raw price data

{{
    config(
        materialized='incremental',
        unique_key=['symbol', 'timestamp'],
        incremental_strategy='merge'
    )
}}

with source as (
    select * from {{ source('raw', 'prices') }}
),

cleaned as (
    select
        symbol,
        timestamp,
        open::float as open_price,
        high::float as high_price,
        low::float as low_price,
        close::float as close_price,
        volume::float as volume,
        -- Data quality checks
        case when close > 0 then true else false end as is_valid_price,
        -- Metadata
        current_timestamp as _loaded_at
    from source
    where symbol is not null
      and timestamp is not null
      and close > 0
)

select * from cleaned

{% if is_incremental() %}
where timestamp > (select max(timestamp) from {{ this }})
{% endif %}
