with history as (
    select *,
        count(*) over prior_window as bars_in_window,
        avg(close) over prior_window as rolling_mean,
        min(minute_utc) over prior_window as first_minute
    from {{ ref('stg_snapshot_minutes') }}
    window prior_window as (
        partition by source, symbol, snapshot_id order by minute_utc
        rows between 5 preceding and 1 preceding
    )
)
select source, symbol, minute_utc, close, snapshot_id, manifest_sha256,
       case when bars_in_window = 5
                 and date_diff('minute', first_minute, minute_utc) = 5
            then rolling_mean end as previous_five_minute_mean
from history
