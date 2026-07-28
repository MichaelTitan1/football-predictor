"""Run the single canonical Fovra V1 data update.

Production:
    python -m src.data_pipeline.update_canonical

Local-only validation/testing:
    python -m src.data_pipeline.update_canonical --sqlite-local --offline

The production source of truth is the existing Supabase PostgreSQL project.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

from .canonical_data import connect, initialize, upsert_records
from .football_data_provider import FootballDataProvider
from .supabase_store import SupabaseStore

logger = logging.getLogger(__name__)


def run(*, offline: bool = False, sqlite_local: bool = False, db_path: str = "data/processed/fovra_data.sqlite3") -> dict:
    logger.info("FOVRA: starting canonical ingestion")

    provider = FootballDataProvider(include_remote=not offline)

    logger.info("FOVRA: fetching Football-Data provider")
    snapshot = provider.fetch()

    logger.info(
        "FOVRA: fetched %s matches, %s leagues, %s teams",
        len(snapshot.matches),
        len(snapshot.leagues),
        len(snapshot.teams),
    )

    newest = max((m.kickoff_utc for m in snapshot.matches), default=None)
    if sqlite_local:
        conn=connect(db_path); initialize(conn)
        try:
            upserted=upsert_records(conn,snapshot.leagues,snapshot.teams,snapshot.matches,snapshot.provider)
            return {"provider":snapshot.provider,"fetched_at":snapshot.fetched_at,"records_seen":len(snapshot.matches),"records_upserted":upserted,"storage":"sqlite-local-only","offline":offline}
        finally: conn.close()
    logger.info("FOVRA: connecting to Supabase")

    store = SupabaseStore()

    logger.info("FOVRA: recording ingestion start")
    run_id = store.record_ingestion_start(snapshot.provider)

       try:
        logger.info("FOVRA: starting Supabase canonical upsert")

        upserted = store.upsert_snapshot(
            snapshot.leagues,
            snapshot.teams,
            snapshot.matches,
            snapshot.provider,
            snapshot.fetched_at,
        )

        logger.info("FOVRA: Supabase upsert complete: %s records", upserted)

        logger.info("FOVRA: resolving finished predictions")

        resolved = store.resolve_predictions(snapshot.matches)

        logger.info("FOVRA: resolved %s predictions", resolved)
        store._request("PATCH","data_sources",params={"provider_key":f"eq.{snapshot.provider}"},payload={"last_attempt_at":datetime.now(timezone.utc).isoformat(),"last_success_at":datetime.now(timezone.utc).isoformat(),"last_data_at":snapshot.fetched_at,"last_success_rows":len(snapshot.matches),"last_error":None})
        store.record_ingestion_finish(run_id,status="succeeded",records_seen=len(snapshot.matches),records_upserted=upserted,newest_match_at=newest)
        return {"provider":snapshot.provider,"fetched_at":snapshot.fetched_at,"records_seen":len(snapshot.matches),"records_upserted":upserted,"prediction_results_resolved":resolved,"newest_match_at":newest,"storage":"supabase-postgresql","offline":offline}
    except Exception as exc:
        try:
            store.record_ingestion_finish(run_id,status="failed",records_seen=len(snapshot.matches),records_upserted=0,newest_match_at=newest,error_message=str(exc))
            store._request("PATCH","data_sources",params={"provider_key":f"eq.{snapshot.provider}"},payload={"last_attempt_at":datetime.now(timezone.utc).isoformat(),"last_error":str(exc)[:2000]})
        except Exception: logger.exception("Could not record ingestion failure metadata")
        raise

def main()->None:
    parser=argparse.ArgumentParser(description="Update Fovra's canonical football data store")
    parser.add_argument("--offline",action="store_true",help="Use existing data/raw CSVs; never fetch remote data")
    parser.add_argument("--sqlite-local",action="store_true",help="Use SQLite only for local testing; never use this in production")
    parser.add_argument("--db",default="data/processed/fovra_data.sqlite3")
    args=parser.parse_args(); logging.basicConfig(level=logging.INFO); print(json.dumps(run(offline=args.offline,sqlite_local=args.sqlite_local,db_path=args.db),indent=2))

if __name__=="__main__": main()
