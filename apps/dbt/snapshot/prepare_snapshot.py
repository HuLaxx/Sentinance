"""Import an independently pinned immutable snapshot into an isolated DuckDB."""

import argparse
import hashlib
from pathlib import Path

import duckdb
from price_lake import verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    digest = hashlib.sha256((args.snapshot / "manifest.json").read_bytes()).hexdigest()
    if digest != args.manifest_sha256:
        raise ValueError("Manifest pin mismatch")
    manifest = verify_snapshot(args.snapshot)
    if args.database.exists():
        raise FileExistsError("Use a new isolated database for each snapshot")
    with duckdb.connect(str(args.database)) as con:
        con.execute("SET TimeZone='UTC'")
        con.execute("create schema raw")
        con.execute(
            """create table raw.minute_bars as
            select *, ? as snapshot_id, ? as manifest_sha256 from read_parquet(?)""",
            [
                manifest["snapshot_id"],
                digest,
                str(args.snapshot / "curated" / "minute_bars.parquet"),
            ],
        )
        count = con.execute("select count(*) from raw.minute_bars").fetchone()[0]
        if count != manifest["curated_minute_count"]:
            raise ValueError("Imported minute count mismatch")


if __name__ == "__main__":
    main()
