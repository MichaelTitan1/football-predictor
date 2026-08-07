from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .data_downloader import LEAGUE_CONFIG, RAW_DIR, START_YEAR, current_season_start
from .football_data_provider import FootballDataProvider
from .neon_store import NeonStore

logger = logging.getLogger(__name__)

SINGLE_LEAGUES = tuple(k for k, v in LEAGUE_CONFIG.items() if v.get("source_type") == "single")
SEASON_LEAGUES = tuple(k for k, v in LEAGUE_CONFIG.items() if v.get("source_type") != "single")
REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
# Football-Data's 16 worldwide all-seasons files use a different schema:
# Home/Away/HG/AG/Res instead of HomeTeam/AwayTeam/FTHG/FTAG/FTR.
ALIASES = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"}
SINGLE_URLS = {code: f"https://www.football-data.co.uk/new/{code}.csv" for code in SINGLE_LEAGUES}
TIMEOUT = int(os.getenv("FOVRA_SINGLE_FEED_TIMEOUT", "60"))
RETRIES = int(os.getenv("FOVRA_SINGLE_FEED_RETRIES", "5"))
STAGE_BATCH = int(os.getenv("FOVRA_FAST_STAGE_BATCH", "20000"))
SOURCE_STATE = RAW_DIR / "football_data_source_state.json"


def _normalise_frame(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    rename = {}
    for source, target in ALIASES.items():
        if target not in df.columns and source in df.columns:
            rename[source] = target
    if rename:
        df = df.rename(columns=rename)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{code} CSV missing required match columns after normalization: {missing}; columns={list(df.columns)}"
        )
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[df["Date"].dt.year.ge(START_YEAR)].copy()
    df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna()].copy()
    df["League"] = code
    df["Tier"] = LEAGUE_CONFIG[code]["tier"]
    df["LeagueStrength"] = LEAGUE_CONFIG[code]["strength"]
    return df


def _read_csv_bytes(payload: bytes, code: str) -> pd.DataFrame:
    last = None
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return _normalise_frame(pd.read_csv(io.BytesIO(payload), encoding=encoding), code)
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to parse Football-Data CSV for {code}: {last}")


def _get(session: requests.Session, url: str) -> bytes:
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT, allow_redirects=True)
            response.raise_for_status()
            if not response.content:
                raise RuntimeError("empty response")
            return response.content
        except Exception as exc:
            last = exc
            if attempt < RETRIES:
                time.sleep(min(2 ** (attempt - 1), 16))
    raise RuntimeError(f"download failed for {url}: {last}")


def _load_state() -> dict[str, Any]:
    if not SOURCE_STATE.exists():
        return {}
    try:
        return json.loads(SOURCE_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, Any]) -> None:
    SOURCE_STATE.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_STATE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _write_single_if_changed(code: str, payload: bytes, frame: pd.DataFrame, url: str) -> bool:
    state = _load_state()
    key = f"{code}:new"
    digest = hashlib.sha256(payload).hexdigest()
    target = RAW_DIR / f"{code}_new.csv"
    if target.exists() and state.get(key, {}).get("sha256") == digest:
        return False
    temp = target.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(target)
    state[key] = {
        "url": url,
        "sha256": digest,
        "row_count": len(frame),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_state(state)
    return True


def _season_url(code: str, year: int) -> str:
    info = LEAGUE_CONFIG[code]
    season_code = f"{year % 100:02d}{(year + 1) % 100:02d}"
    return f"https://www.football-data.co.uk/mmz4281/{season_code}/{info['code']}.csv"


def refresh_sources() -> dict[str, Any]:
    """Refresh all 38 leagues without re-downloading historical single feeds."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Fovra/1.0 (football prediction data refresh)",
        "Accept": "text/csv,*/*;q=0.8",
    })
    current = current_season_start()
    bootstrap = not SOURCE_STATE.exists()
    changed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    # The 16 extra leagues are explicitly all-seasons files. Download each
    # exact /new/CODE.csv feed once, then filter locally to Date >= 2010.
    for code in SINGLE_LEAGUES:
        try:
            url = SINGLE_URLS[code]
            payload = _get(session, url)
            frame = _read_csv_bytes(payload, code)
            if _write_single_if_changed(code, payload, frame, url):
                changed.append(code)
            else:
                skipped.append(code)
            logger.info("Extra league %s: %d rows retained from %s", code, len(frame), url)
        except Exception as exc:
            failed.append(code)
            logger.error("Extra league %s failed: %s", code, exc)

    # The 22 main leagues are season-by-season. Bootstrap is the only time
    # historical seasons are downloaded. Normal Monday/Wednesday runs touch
    # only the current season, leaving the cached historical files alone.
    for code in SEASON_LEAGUES:
        years = range(START_YEAR, current + 1) if bootstrap else (current,)
        for year in years:
            path = RAW_DIR / f"{code}_{year}.csv"
            if path.exists() and not bootstrap and year != current:
                continue
            url = _season_url(code, year)
            try:
                payload = _get(session, url)
                frame = _read_csv_bytes(payload, code)
                digest = hashlib.sha256(payload).hexdigest()
                state = _load_state()
                key = f"{code}:{year}"
                if path.exists() and state.get(key, {}).get("sha256") == digest:
                    skipped.append(key)
                    continue
                temp = path.with_suffix(".tmp")
                frame.to_csv(temp, index=False)
                temp.replace(path)
                state[key] = {
                    "url": url,
                    "sha256": digest,
                    "row_count": len(frame),
                    "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                _save_state(state)
                changed.append(key)
            except Exception as exc:
                # A not-yet-published current season must not erase the last
                # good local season. The local snapshot will still be used.
                logger.warning("Season feed %s %s unavailable/failed: %s", code, year, exc)

    if failed:
        raise RuntimeError("Football-Data extra leagues failed: " + ", ".join(failed))
    return {
        "bootstrap": bootstrap,
        "changed": len(changed),
        "skipped": len(skipped),
        "single_leagues": len(SINGLE_LEAGUES),
        "season_leagues": len(SEASON_LEAGUES),
    }


def _rows(snapshot: Any):
    now = datetime.now(timezone.utc)
    league_rows = [(l.key, l.name, l.country, snapshot.provider, snapshot.fetched_at, now) for l in snapshot.leagues]
    team_rows = [(f"{t.league_key}:{t.key}", t.league_key, t.name, snapshot.provider, snapshot.fetched_at, now) for t in snapshot.teams]
    match_rows = []
    provider_rows = []
    for m in snapshot.matches:
        match_key = m.match_key
        source_id = m.source_id or match_key
        home_key = f"{m.league_key}:{m.home_team}"
        away_key = f"{m.league_key}:{m.away_team}"
        match_rows.append((match_key, m.league_key, m.season, m.kickoff_utc, m.status, home_key, away_key, m.home_score, m.away_score, snapshot.provider, source_id, snapshot.fetched_at, now))
        payload = json.dumps({
            "league_key": m.league_key,
            "season": m.season,
            "kickoff_utc": m.kickoff_utc,
            "status": m.status,
            "home_team": home_key,
            "away_team": away_key,
            "home_score": m.home_score,
            "away_score": m.away_score,
        }, separators=(",", ":"))
        provider_rows.append((snapshot.provider, "match", source_id, match_key, payload, snapshot.fetched_at, now))
    return league_rows, team_rows, match_rows, provider_rows


def bulk_upsert(snapshot: Any) -> int:
    """COPY all staging data, then do set-based PostgreSQL merges."""
    store = NeonStore()
    store.initialize_schema()
    conn = store.connection
    league_rows, team_rows, match_rows, provider_rows = _rows(snapshot)

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO leagues(canonical_key,name,country,source_provider,source_updated_at,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT(canonical_key) DO UPDATE SET
                 name=EXCLUDED.name,country=EXCLUDED.country,source_provider=EXCLUDED.source_provider,
                 source_updated_at=EXCLUDED.source_updated_at,updated_at=EXCLUDED.updated_at""",
            league_rows,
        )
        cur.executemany(
            """INSERT INTO teams(canonical_key,league_canonical_key,name,source_provider,source_updated_at,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT(canonical_key) DO UPDATE SET
                 league_canonical_key=EXCLUDED.league_canonical_key,name=EXCLUDED.name,
                 source_provider=EXCLUDED.source_provider,source_updated_at=EXCLUDED.source_updated_at,
                 updated_at=EXCLUDED.updated_at""",
            team_rows,
        )
        cur.execute("""CREATE TEMP TABLE fovra_match_stage (
            canonical_key text, league_canonical_key text, season text, kickoff_at timestamptz,
            status text, home_team_canonical_key text, away_team_canonical_key text,
            home_score integer, away_score integer, source_provider text, source_match_id text,
            source_updated_at timestamptz, updated_at timestamptz) ON COMMIT DROP""")
        cur.execute("""CREATE TEMP TABLE fovra_provider_stage (
            provider_key text, record_type text, source_id text, canonical_key text,
            payload jsonb, source_updated_at timestamptz, updated_at timestamptz) ON COMMIT DROP""")

    for start in range(0, len(match_rows), STAGE_BATCH):
        match_chunk = match_rows[start:start + STAGE_BATCH]
        provider_chunk = provider_rows[start:start + STAGE_BATCH]
        with conn.cursor() as cur:
            with cur.copy("""COPY fovra_match_stage
                (canonical_key,league_canonical_key,season,kickoff_at,status,home_team_canonical_key,
                 away_team_canonical_key,home_score,away_score,source_provider,source_match_id,
                 source_updated_at,updated_at) FROM STDIN""") as copy:
                for row in match_chunk:
                    copy.write_row(row)
            with cur.copy("""COPY fovra_provider_stage
                (provider_key,record_type,source_id,canonical_key,payload,source_updated_at,updated_at)
                FROM STDIN""") as copy:
                for row in provider_chunk:
                    copy.write_row(row)
        logger.info("Copied %d/%d matches into Neon staging", min(start + len(match_chunk), len(match_rows)), len(match_rows))

    with conn.cursor() as cur:
        cur.execute("""INSERT INTO matches(
            canonical_key,league_canonical_key,season,kickoff_at,status,home_team_canonical_key,
            away_team_canonical_key,home_score,away_score,source_provider,source_match_id,source_updated_at,updated_at)
            SELECT canonical_key,league_canonical_key,season,kickoff_at,status,home_team_canonical_key,
                   away_team_canonical_key,home_score,away_score,source_provider,source_match_id,source_updated_at,updated_at
            FROM fovra_match_stage
            ON CONFLICT(canonical_key) DO UPDATE SET
              league_canonical_key=EXCLUDED.league_canonical_key,season=EXCLUDED.season,
              kickoff_at=EXCLUDED.kickoff_at,status=EXCLUDED.status,
              home_team_canonical_key=EXCLUDED.home_team_canonical_key,
              away_team_canonical_key=EXCLUDED.away_team_canonical_key,
              home_score=EXCLUDED.home_score,away_score=EXCLUDED.away_score,
              source_provider=EXCLUDED.source_provider,source_match_id=EXCLUDED.source_match_id,
              source_updated_at=EXCLUDED.source_updated_at,updated_at=EXCLUDED.updated_at
            WHERE (matches.season,matches.kickoff_at,matches.status,matches.home_team_canonical_key,
                   matches.away_team_canonical_key,matches.home_score,matches.away_score,
                   matches.source_provider,matches.source_match_id)
              IS DISTINCT FROM
                  (EXCLUDED.season,EXCLUDED.kickoff_at,EXCLUDED.status,EXCLUDED.home_team_canonical_key,
                   EXCLUDED.away_team_canonical_key,EXCLUDED.home_score,EXCLUDED.away_score,
                   EXCLUDED.source_provider,EXCLUDED.source_match_id)""")
        changed = cur.rowcount
        cur.execute("""INSERT INTO provider_records(provider_key,record_type,source_id,canonical_key,payload,source_updated_at,updated_at)
            SELECT provider_key,record_type,source_id,canonical_key,payload,source_updated_at,updated_at
            FROM fovra_provider_stage
            ON CONFLICT(provider_key,record_type,source_id) DO UPDATE SET
              canonical_key=EXCLUDED.canonical_key,payload=EXCLUDED.payload,
              source_updated_at=EXCLUDED.source_updated_at,updated_at=EXCLUDED.updated_at
            WHERE provider_records.canonical_key IS DISTINCT FROM EXCLUDED.canonical_key
               OR provider_records.payload IS DISTINCT FROM EXCLUDED.payload
               OR provider_records.source_updated_at IS DISTINCT FROM EXCLUDED.source_updated_at""")
        conn.commit()
    return int(changed if changed >= 0 else len(match_rows))


def run() -> dict[str, Any]:
    refresh = refresh_sources()
    provider = FootballDataProvider(raw_dir=RAW_DIR, include_remote=False)
    snapshot = provider.fetch()
    expected = set(LEAGUE_CONFIG)
    actual = {m.league_key for m in snapshot.matches}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError("Football-Data local snapshot is missing configured leagues: " + ", ".join(missing))

    store = NeonStore()
    store.initialize_schema()
    run_id = store.record_ingestion_start(snapshot.provider)
    try:
        upserted = bulk_upsert(snapshot)
        now = datetime.now(timezone.utc).isoformat()
        store.upsert("provider_sources", [{"provider_key": snapshot.provider, "display_name": snapshot.provider, "source_type": "match-data", "updated_at": now}], "provider_key")
        store.upsert("data_sources", [{"provider_key": snapshot.provider, "display_name": snapshot.provider, "last_attempt_at": now, "last_success_at": now, "last_data_at": snapshot.fetched_at, "last_success_rows": len(snapshot.matches), "last_error": None, "freshness_policy": "Monday/Wednesday Football-Data refresh; incremental local cache"}], "provider_key")
        store.record_ingestion_finish(run_id, status="succeeded", records_seen=len(snapshot.matches), records_upserted=upserted, newest_match_at=max((m.kickoff_utc for m in snapshot.matches), default=None))
        return {"provider": snapshot.provider, "records_seen": len(snapshot.matches), "records_upserted": upserted, "leagues_seen": len(actual), "refresh": refresh}
    except Exception as exc:
        store.record_ingestion_finish(run_id, status="failed", records_seen=len(snapshot.matches), records_upserted=0, error_message=str(exc))
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=2, default=str))
