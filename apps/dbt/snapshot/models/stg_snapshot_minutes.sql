select source, symbol, minute_utc, open, high, low, close,
       base_volume, trade_count, vwap, snapshot_id, manifest_sha256
from {{ source('snapshot', 'minute_bars') }}
