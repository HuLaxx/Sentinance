"""Manually export an immutable local Parquet snapshot from the durable ledger.

Requires a populated PostgreSQL ledger and an explicit local export directory.
No S3, cloud deployment, external provider traffic or destructive DB writes.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from database import close_db
from price_lake import export_ledger_snapshot


async def main(destination: Path) -> None:
    if os.getenv("P1_LOCAL_LAKE_EXPORT_ENABLED") != "1":
        raise RuntimeError(
            "Local Parquet export disabled; set P1_LOCAL_LAKE_EXPORT_ENABLED=1"
        )
    try:
        manifest = await export_ledger_snapshot(destination)
        print(json.dumps(manifest, sort_keys=True, indent=2))
    finally:
        await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    asyncio.run(main(options.output_dir))
