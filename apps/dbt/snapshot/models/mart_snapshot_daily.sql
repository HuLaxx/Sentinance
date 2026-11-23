select source, symbol, cast(minute_utc as date) as day_utc,
       first(open order by minute_utc) as open,
       max(high) as high, min(low) as low,
       last(close order by minute_utc) as close,
       sum(base_volume) as base_volume, sum(trade_count) as trade_count,
       count(*) as observed_minutes,
       snapshot_id, manifest_sha256
from {{ ref('stg_snapshot_minutes') }}
group by source, symbol, cast(minute_utc as date), snapshot_id, manifest_sha256
