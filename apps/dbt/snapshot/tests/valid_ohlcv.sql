select * from {{ ref('stg_snapshot_minutes') }}
where least(open, high, low, close) <= 0 or high < greatest(open, close)
   or low > least(open, close) or high < low
   or base_volume < 0 or trade_count <= 0
   or open is null or high is null or low is null or close is null
   or base_volume is null or trade_count is null
