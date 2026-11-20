with expected as (
    select source, symbol, cast(minute_utc as date) as day_utc, snapshot_id,
           sum(trade_count) as trades, sum(base_volume) as volume
    from {{ ref('stg_snapshot_minutes') }}
    group by source, symbol, day_utc, snapshot_id
)
select e.source from expected e
full join {{ ref('mart_snapshot_daily') }} m
using (source, symbol, day_utc, snapshot_id)
where e.trades is distinct from m.trade_count
   or e.volume is distinct from m.base_volume
