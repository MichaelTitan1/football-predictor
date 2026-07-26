"""Run the single canonical Fovra V1 data update.

Usage:
    python -m src.data_pipeline.update_canonical
    python -m src.data_pipeline.update_canonical --offline
"""
from __future__ import annotations

import argparse
import json
import logging

from .canonical_data import connect, initialize, record_source_error, summary, upsert_records
from .football_data_provider import FootballDataProvider

logger = logging.getLogger(__name__)


def run(offline: bool = False, db_path: str = "data/processed/fovra_data.sqlite3") -> dict:
    conn = connect(db_path)
    initialize(conn)
    provider = FootballDataProvider(include_remote=not offline)
    try:
        snapshot = provider.fetch()
        upserted = upsert_records(
            conn,
            snapshot.leagues,
            snapshot.teams,
            snapshot.matches,
            snapshot.provider,
        )
        result = summary(conn)
        result.update(
            {
                "provider": snapshot.provider,
                "fetched_at": snapshot.fetched_at,
                "records_seen": len(snapshot.matches),
                "records_upserted": upserted,
                "offline": offline,
            }
        )
        return result
    except Exception as exc:
        record_source_error(conn, provider.name, str(exc))
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Fovra's canonical football data store")
    parser.add_argument("--offline", action="store_true", help="Use only existing data/raw CSVs; never fetch the network")
    parser.add_argument("--db", default="data/processed/fovra_data.sqlite3")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(offline=args.offline, db_path=args.db), indent=2))


if __name__ == "__main__":
    main()
