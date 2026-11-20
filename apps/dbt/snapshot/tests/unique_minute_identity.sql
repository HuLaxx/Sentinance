select source, symbol, minute_utc, snapshot_id
from {{ ref('stg_snapshot_minutes') }}
group by source, symbol, minute_utc, snapshot_id
having count(*) <> 1
