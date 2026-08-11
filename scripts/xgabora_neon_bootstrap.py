from __future__ import annotations

"""Bootstrap Fovra Neon from xgabora's curated 2000-2025 match dataset.

Loads only the audited 2010/11-2024/25 completed-match window. The complete
xgabora row is preserved in provider_records as source evidence. The existing
Fovra matches schema is respected: raw payload is NOT required on matches.
"""

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen

import psycopg

SOURCE_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
PROVIDER = "xgabora-club-football-2000-2025"
START_DATE = datetime(2010, 7, 1).date()
END_DATE = datetime(2025, 7, 1).date()
EXPECTED_COMPLETED_MATCHES = 168120
BATCH_SIZE = int(os.getenv("FOVRA_NEON_BATCH_SIZE", "500"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean(value: str | None):
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def integer(value):
    value = clean(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def kickoff_utc(match_date: str, match_time: str | None) -> datetime:
    d = datetime.strptime(match_date, "%Y-%m-%d").date()
    t = datetime.strptime(clean(match_time) or "00:00:00", "%H:%M:%S").time()
    return datetime.combine(d, t, tzinfo=timezone(timedelta(hours=1))).astimezone(timezone.utc)


def canonical_key(division: str, season: str, match_date: str, home: str, away: str) -> str:
    raw = "|".join(("match-v2", division, season or "", match_date, home, away))
    return sha256_text(raw)


def season_for_date(match_date: str) -> str:
    d = datetime.strptime(match_date, "%Y-%m-%d").date()
    year = d.year if d.month >= 7 else d.year - 1
    return f"{year}-{year + 1}"


def download_rows() -> list[dict[str, str]]:
    request = Request(SOURCE_URL, headers={"User-Agent": "Fovra/1.0"})
    with urlopen(request, timeout=180) as response:
        raw = response.read()
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def prepare(rows: list[dict[str, str]]):
    prepared = []
    leagues: dict[str, tuple[str, str | None]] = {}
    teams: dict[str, tuple[str, str]] = {}
    seen: set[str] = set()
    quarantined = 0

    for row in rows:
        division = clean(row.get("Division"))
        match_date = clean(row.get("MatchDate"))
        home = clean(row.get("HomeTeam"))
        away = clean(row.get("AwayTeam"))
        fthome = integer(row.get("FTHome"))
        ftaway = integer(row.get("FTAway"))
        result = clean(row.get("FTResult"))
        if not division or not match_date or not home or not away or fthome is None or ftaway is None or result not in {"H", "D", "A"}:
            quarantined += 1
            continue
        try:
            d = datetime.strptime(match_date, "%Y-%m-%d").date()
        except ValueError:
            quarantined += 1
            continue
        if not (START_DATE <= d < END_DATE):
            continue

        season = season_for_date(match_date)
        key = canonical_key(division, season, match_date, home, away)
        if key in seen:
            raise RuntimeError(f"Duplicate canonical match identity in xgabora source: {key}")
        seen.add(key)
        home_key = f"{division}:{sha256_text(home.lower())}"
        away_key = f"{division}:{sha256_text(away.lower())}"
        leagues[division] = (division, None)
        teams[home_key] = (division, home)
        teams[away_key] = (division, away)
        payload = {k: clean(v) for k, v in row.items()}
        prepared.append((key, division, season, kickoff_utc(match_date, row.get("MatchTime")), home_key, away_key, fthome, ftaway, payload))

    return prepared, leagues, teams, quarantined


def batches(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def run() -> dict:
    rows = download_rows()
    prepared, leagues, teams, quarantined = prepare(rows)
    if len(prepared) != EXPECTED_COMPLETED_MATCHES:
        raise RuntimeError(f"Refusing Neon bootstrap: expected audited {EXPECTED_COMPLETED_MATCHES} completed matches, got {len(prepared)}")

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")

    now = datetime.now(timezone.utc)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO provider_sources(provider_key,display_name,source_type,updated_at) VALUES(%s,%s,%s,%s) ON CONFLICT(provider_key) DO UPDATE SET display_name=excluded.display_name,source_type=excluded.source_type,updated_at=excluded.updated_at", (PROVIDER, "xgabora Club Football Match Data 2000-2025", "curated-historical", now))
            cur.execute("INSERT INTO ingestion_runs(provider_key,status,started_at,records_seen) VALUES(%s,'running',%s,%s) RETURNING id", (PROVIDER, now, len(prepared)))
            run_id = cur.fetchone()[0]

            for batch in batches(list(leagues.items()), BATCH_SIZE):
                cur.executemany("INSERT INTO leagues(canonical_key,name,country,source_provider,source_updated_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(canonical_key) DO UPDATE SET name=excluded.name,source_provider=excluded.source_provider,source_updated_at=excluded.source_updated_at,updated_at=excluded.updated_at", [(k, v[0], v[1], PROVIDER, now, now) for k, v in batch])
            for batch in batches(list(teams.items()), BATCH_SIZE):
                cur.executemany("INSERT INTO teams(canonical_key,league_canonical_key,name,source_provider,source_updated_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(canonical_key) DO UPDATE SET name=excluded.name,league_canonical_key=excluded.league_canonical_key,source_provider=excluded.source_provider,source_updated_at=excluded.source_updated_at,updated_at=excluded.updated_at", [(k, v[0], v[1], PROVIDER, now, now) for k, v in batch])

            for batch in batches(prepared, BATCH_SIZE):
                match_params = []
                provider_params = []
                for key, division, season, kickoff, home_key, away_key, hs, aws, payload in batch:
                    source_id = key
                    # IMPORTANT: raw xgabora evidence belongs in provider_records.
                    # Existing Fovra matches tables may not have a payload column.
                    match_params.append((key, division, season, kickoff, "finished", home_key, away_key, hs, aws, PROVIDER, source_id, now))
                    provider_params.append((PROVIDER, "match", source_id, key, json.dumps(payload), now, now))
                cur.executemany("INSERT INTO matches(canonical_key,league_canonical_key,season,kickoff_at,status,home_team_canonical_key,away_team_canonical_key,home_score,away_score,source_provider,source_match_id,source_updated_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(canonical_key) DO UPDATE SET status='finished',home_score=excluded.home_score,away_score=excluded.away_score,source_provider=excluded.source_provider,source_match_id=excluded.source_match_id,source_updated_at=excluded.source_updated_at,updated_at=excluded.updated_at", [(a,b,c,d,e,f,g,h,i,j,k,l,now) for a,b,c,d,e,f,g,h,i,j,k,l in match_params])
                cur.executemany("INSERT INTO provider_records(provider_key,record_type,source_id,canonical_key,payload,source_updated_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(provider_key,record_type,source_id) DO UPDATE SET canonical_key=excluded.canonical_key,payload=excluded.payload,source_updated_at=excluded.source_updated_at,updated_at=excluded.updated_at", provider_params)

            cur.execute("SELECT count(*) FROM matches WHERE source_provider=%s AND status='finished'", (PROVIDER,))
            match_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM provider_records WHERE provider_key=%s AND record_type='match'", (PROVIDER,))
            provider_count = cur.fetchone()[0]
            if match_count != len(prepared) or provider_count != len(prepared):
                raise RuntimeError(f"Neon verification failed before commit: matches={match_count}, provider_records={provider_count}, expected={len(prepared)}")
            cur.execute("UPDATE ingestion_runs SET status='succeeded',finished_at=%s,records_upserted=%s WHERE id=%s", (datetime.now(timezone.utc), len(prepared), run_id))
        conn.commit()

    return {"provider": PROVIDER, "source_rows": len(rows), "valid_completed_matches": len(prepared), "leagues": len(leagues), "teams": len(teams), "quarantined_incomplete": quarantined, "neon_matches": match_count, "neon_provider_records": provider_count, "status": "PASS"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
