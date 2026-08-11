from __future__ import annotations

import json
import os
import sys

import psycopg

PROVIDER = "xgabora-club-football-2000-2025"
EXPECTED = 168120


def run() -> dict:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM matches WHERE source_provider=%s AND status='finished'", (PROVIDER,))
            matches = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM provider_records WHERE provider_key=%s AND record_type='match'", (PROVIDER,))
            provider_records = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM matches WHERE source_provider=%s AND (home_score IS NULL OR away_score IS NULL)", (PROVIDER,))
            incomplete = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM matches WHERE source_provider=%s AND kickoff_at > now()", (PROVIDER,))
            future = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM (SELECT canonical_key FROM matches WHERE source_provider=%s GROUP BY canonical_key HAVING count(*) > 1) d", (PROVIDER,))
            duplicate_keys = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM leagues")
            leagues = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM teams")
            teams = cur.fetchone()[0]
    result = {
        "provider": PROVIDER,
        "expected_completed_matches": EXPECTED,
        "neon_finished_matches": matches,
        "neon_provider_records": provider_records,
        "incomplete_matches": incomplete,
        "future_dated_matches": future,
        "duplicate_canonical_keys": duplicate_keys,
        "leagues": leagues,
        "teams": teams,
        "status": "PASS" if matches == EXPECTED and provider_records == EXPECTED and incomplete == 0 and future == 0 and duplicate_keys == 0 else "FAIL",
    }
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        sys.exit(1)
