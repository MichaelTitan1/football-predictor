"""Fast, deterministic Football-Data -> Neon ingestion for Fovra.

Football-Data has two layouts in this project:
22 main divisions are season-by-season files; 16 extra divisions are exposed
through country-specific pages whose current CSV links are all-seasons feeds.
The extra feeds are downloaded once and filtered locally to Date >= 2010.

The Neon write path uses PostgreSQL COPY into a temporary staging table and a
single INSERT ... ON CONFLICT merge. This avoids one SQL round-trip per match.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from .canonical_data import MatchRecord
from .data_downloader import (
    ALLOWED_LEAGUES,
    LEAGUE_CONFIG,
    RAW_DIR,
    START_YEAR,
    current_season_start,
    download_all_leagues,
    download_season_data,
)
from .football_data_provider import FootballDataProvider
from .neon_store import NeonStore

logger = logging.getLogger(__name__)

SINGLE_LEAGUES = tuple(key for key, info in LEAGUE_CONFIG.items() if info.get("source_type") == "single")
SEASON_LEAGUES = tuple(key for key, info in LEAGUE_CONFIG.items() if info.get("source_type") != "single")
SINGLE_START_YEAR = START_YEAR
SINGLE_TIMEOUT = int(os.getenv("FOVRA_SINGLE_FEED_TIMEOUT", "60"))
SINGLE_RETRIES = int(os.getenv("FOVRA_SINGLE_FEED_RETRIES", "5"))
MATCH_STAGE_BATCH = int(os.getenv("FOVRA_FAST_STAGE_BATCH", "10000"))

# Football-Data exposes the 16 extra leagues through country pages. The
# apparent /new/{CODE}.csv endpoint can return the country data page rather
# than the CSV itself, so discover the actual CSV href instead of guessing a
# storage path that can change.
SINGLE_PAGE_SLUGS = {
    "ARG": "argentina",
    "AUT": "austria",
    "BRA": "brazil",
    "CHN": "china",
    "DNK": "denmark",
    "FIN": "finland",
    "IRL": "ireland",
    "JPN": "japan",
    "MEX": "mexico",
    "NOR": "norway",
    "POL": "poland",
    "ROU": "romania",
    "RUS": "russia",
    "SWE": "sweden",
    "SWZ": "switzerland",
    "USA": "usa",
}
REQUIRED_MATCH_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


class _CsvLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value and ".csv" in value.lower():
                self.hrefs.append(value.strip())


def _fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state_path() -> Path:
    return RAW_DIR / "football_data_source_state.json"


def _load_state() -> dict[str, dict[str, Any]]:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict[str, dict[str, Any]]) -> None:
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def _payload_is_match_csv(payload: bytes) -> bool:
    try:
        frame = _normalise_columns(pd.read_csv(io.BytesIO(payload), nrows=0))
        return REQUIRED_MATCH_COLUMNS.issubset(set(frame.columns))
    except Exception:
        return False


def _csv_links_from_html(payload: bytes, base_url: str) -> list[str]:
    try:
        parser = _CsvLinkParser()
        parser.feed(payload.decode("utf-8", errors="replace"))
    except Exception:
        return []
    links: list[str] = []
    for href in parser.hrefs:
        absolute = urljoin(base_url, href)
        if absolute not in links:
            links.append(absolute)
    return links


def _http_get_with_retries(session: requests.Session, url: str) -> tuple[bytes | None, int | None, Exception | None]:
    last_error: Exception | None = None
    for attempt in range(1, SINGLE_RETRIES + 1):
        try:
            response = session.get(url, timeout=SINGLE_TIMEOUT, allow_redirects=True)
            if response.status_code == 404:
                return None, 404, RuntimeError(f"Football-Data returned HTTP 404 for {url}")
            response.raise_for_status()
            if not response.content:
                return None, response.status_code, RuntimeError(f"Football-Data returned an empty response for {url}")
            return response.content, response.status_code, None
        except Exception as exc:
            last_error = exc
            if attempt < SINGLE_RETRIES:
                delay = min(2 ** (attempt - 1), 16)
                logger.warning("%s attempt %d/%d failed: %s; retrying in %ss", url, attempt, SINGLE_RETRIES, exc, delay)
                time.sleep(delay)
    return None, None, last_error


def _fetch_single_feed(code: str) -> tuple[bytes, str]:
    """Fetch a valid all-seasons extra-league CSV and return payload + real URL."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Fovra/1.0 (football prediction data refresh)",
        "Accept": "text/csv,text/html;q=0.9,*/*;q=0.8",
    })

    page_slug = SINGLE_PAGE_SLUGS.get(code)
    queue = [f"https://www.football-data.co.uk/new/{code}.csv"]
    if page_slug:
        queue.append(f"https://www.football-data.co.uk/{page_slug}.php")

    attempted: set[str] = set()
    last_error: Exception | None = None
    while queue:
        url = queue.pop(0)
        if url in attempted:
            continue
        attempted.add(url)

        payload, _status, error = _http_get_with_retries(session, url)
        if payload is None:
            last_error = error or RuntimeError(f"No response from {url}")
            continue
        if _payload_is_match_csv(payload):
            return payload, url

        for csv_url in _csv_links_from_html(payload, url):
            if csv_url not in attempted:
                queue.append(csv_url)
        last_error = RuntimeError(f"Response from {url} was not a Football-Data match CSV")

    raise RuntimeError(f"Unable to locate a valid Football-Data CSV for extra league {code}: {last_error}")


def refresh_single_leagues() -> dict[str, Any]:
    """Download each all-seasons extra league once and keep only Date >= 2010."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_state()
    refreshed: list[str] = []
    unchanged: list[str] = []
    failed: list[str] = []
    rows_by_league: dict[str, int] = {}

    for code in SINGLE_LEAGUES:
        url_key = f"{code}:new"
        try:
            payload, source_url = _fetch_single_feed(code)
            digest = _fingerprint(payload)
            target = RAW_DIR / f"{code}_new.csv"
            previous = state.get(url_key, {}).get("sha256")

            if target.exists() and previous == digest:
                unchanged.append(code)
                try:
                    rows_by_league[code] = len(pd.read_csv(target, usecols=["Date"]))
                except Exception:
                    pass
                continue

            frame = _normalise_columns(pd.read_csv(io.BytesIO(payload)))
            missing = sorted(REQUIRED_MATCH_COLUMNS - set(frame.columns))
            if missing:
                raise RuntimeError(f"{code} CSV missing required columns: {missing}")

            parsed = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
            keep = parsed.dt.year.ge(SINGLE_START_YEAR).fillna(False)
            frame = frame.loc[keep].copy()
            frame["Date"] = parsed.loc[frame.index]
            frame["League"] = code
            frame["Tier"] = LEAGUE_CONFIG[code]["tier"]
            frame["LeagueStrength"] = LEAGUE_CONFIG[code]["strength"]
            frame.to_csv(target, index=False)
            state[url_key] = {
                "url": source_url,
                "sha256": digest,
                "row_count": int(len(frame)),
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "filter": f"Date >= {SINGLE_START_YEAR}-01-01",
            }
            rows_by_league[code] = int(len(frame))
            refreshed.append(code)
            logger.info("Refreshed %s all-seasons feed from %s: %d rows kept (Date >= %d)", code, source_url, len(frame), SINGLE_START_YEAR)
        except Exception as exc:
            failed.append(code)
            logger.error("Extra league %s failed: %s", code, exc)

    _save_state(state)
    if failed:
        raise RuntimeError("Football-Data extra leagues failed: " + ", ".join(failed))
    return {"refreshed": refreshed, "unchanged": unchanged, "rows": rows_by_league}


def refresh_season_leagues(*, bootstrap: bool) -> dict[str, Any]:
    """Bootstrap 22 season feeds once, then refresh only current/new seasons."""
    if bootstrap:
        logger.info("Bootstrapping the 22 season-based leagues from %d through %d", START_YEAR, current_season_start())
        result = download_all_leagues(START_YEAR, current_season_start())
        return {"bootstrap": True, "activity": sum(len(v) for v in result.values())}

    current = current_season_start()
    activity = 0
    for code in SEASON_LEAGUES:
        try:
            path = RAW_DIR / f"{code}_{current}.csv"
            if path.exists():
                if download_season_data(code, current, force_refresh=True, ignore_unavailable=True):
                    activity += 1
            else:
                if download_season_data(code, current, force_refresh=False, ignore_unavailable=True):
                    activity += 1
        except Exception as exc:
            logger.warning("Current-season refresh failed for %s: %s", code, exc)
    return {"bootstrap": False, "activity": activity}


def _match_rows(matches: list[MatchRecord], provider: str, fetched_at: str) -> list[tuple[Any, ...]]:
    return [
        (
            m.match_key, m.league_key, m.season, m.kickoff_utc, m.status,
            f"{m.league_key}:{m.home_team}", f"{m.league_key}:{m.away_team}",
            m.home_score, m.away_score, provider, m.source_id or m.match_key,
            fetched_at, datetime.now(timezone.utc),
        )
        for m in matches
    ]


def _bulk_upsert(store: NeonStore, snapshot: Any) -> int:
    """Stage and merge the complete snapshot in PostgreSQL using COPY."""
    rows = _match_rows(list(snapshot.matches), snapshot.provider, snapshot.fetched_at)
    now = datetime.now(timezone.utc)
    conn = store.connection

    league_rows = [(l.key, l.name, l.country, snapshot.provider, snapshot.fetched_at, now) for l in snapshot.leagues]
    team_rows = [(f"{t.league_key}:{t.key}", t.league_key, t.name, snapshot.provider, snapshot.fetched_at, now) for t in snapshot.teams]

    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO leagues (canonical_key,name,country,source_provider,source_updated_at,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (canonical_key) DO UPDATE SET
                 name=EXCLUDED.name,country=EXCLUDED.country,source_provider=EXCLUDED.source_provider,
                 source_updated_at=EXCLUDED.source_updated_at,updated_at=EXCLUDED.updated_at""", league_rows,
        )
        cur.executemany(
            """INSERT INTO teams (canonical_key,league_canonical_key,name,source_provider,source_updated_at,updated_at)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (canonical_key) DO UPDATE SET
                 name=EXCLUDED.name,source_provider=EXCLUDED.source_provider,
                 source_updated_at=EXCLUDED.source_updated_at,updated_at=EXCLUDED.updated_at""", team_rows,
        )
        cur.execute("DROP TABLE IF EXISTS fovra_match_stage")
        cur.execute("""CREATE TEMP TABLE fovra_match_stage AS
            SELECT canonical_key,league_canonical_key,season,kickoff_at,status,
                   home_team_canonical_key,away_team_canonical_key,home_score,away_score,
                   source_provider,source_match_id,source_updated_at,updated_at
            FROM matches WITH NO DATA""")

    for start in range(0, len(rows), MATCH_STAGE_BATCH):
        chunk = rows[start:start + MATCH_STAGE_BATCH]
        with conn.cursor() as cur:
            with cur.copy("COPY fovra_match_stage (canonical_key,league_canonical_key,season,kickoff_at,status,home_team_canonical_key,away_team_canonical_key,home_score,away_score,source_provider,source_match_id,source_updated_at,updated_at) FROM STDIN") as copy:
                for row in chunk:
                    copy.write_row(row)
        logger.info("Staged matches %d/%d", min(start + len(chunk), len(rows)), len(rows))

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO matches (
                canonical_key,league_canonical_key,season,kickoff_at,status,
                home_team_canonical_key,away_team_canonical_key,home_score,away_score,
                source_provider,source_match_id,source_updated_at,updated_at
            )
            SELECT canonical_key,league_canonical_key,season,kickoff_at,status,
                   home_team_canonical_key,away_team_canonical_key,home_score,away_score,
                   source_provider,source_match_id,source_updated_at,updated_at
            FROM fovra_match_stage
            ON CONFLICT (canonical_key) DO UPDATE SET
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
                   EXCLUDED.source_provider,EXCLUDED.source_match_id)"""
        )
        changed = cur.rowcount
        cur.execute("DELETE FROM fovra_match_stage WHERE canonical_key IS NULL")
        cur.execute(
            """INSERT INTO provider_records (provider_key,record_type,source_id,canonical_key,payload,source_updated_at,updated_at)
            SELECT source_provider,'match',source_match_id,canonical_key,
                   jsonb_build_object('league_key',league_canonical_key,'season',season,'kickoff_utc',kickoff_at,
                     'status',status,'home_team',home_team_canonical_key,'away_team',away_team_canonical_key,
                     'home_score',home_score,'away_score',away_score),source_updated_at,updated_at
            FROM fovra_match_stage
            ON CONFLICT (provider_key,record_type,source_id) DO UPDATE SET
              canonical_key=EXCLUDED.canonical_key,payload=EXCLUDED.payload,
              source_updated_at=EXCLUDED.source_updated_at,updated_at=EXCLUDED.updated_at
            WHERE provider_records.payload IS DISTINCT FROM EXCLUDED.payload
               OR provider_records.canonical_key IS DISTINCT FROM EXCLUDED.canonical_key
               OR provider_records.source_updated_at IS DISTINCT FROM EXCLUDED.source_updated_at"""
        )
        conn.commit()
    return int(changed if changed >= 0 else len(rows))


def run() -> dict[str, Any]:
    cache_marker = RAW_DIR / "football_data_source_state.json"
    bootstrap = not cache_marker.exists()
    season_result = refresh_season_leagues(bootstrap=bootstrap)
    single_result = refresh_single_leagues()
    provider = FootballDataProvider(raw_dir=RAW_DIR, include_remote=False)
    snapshot = provider.fetch()
    expected = set(ALLOWED_LEAGUES)
    actual = {m.league_key for m in snapshot.matches}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"Football-Data local snapshot is missing configured leagues: {', '.join(missing)}")

    store = NeonStore()
    store.initialize_schema()
    run_id = store.record_ingestion_start(snapshot.provider)
    try:
        upserted = _bulk_upsert(store, snapshot)
        resolved = store.resolve_predictions(snapshot.matches)
        store.upsert("provider_sources", [{"provider_key": snapshot.provider, "display_name": snapshot.provider, "source_type": "match-data", "updated_at": datetime.now(timezone.utc).isoformat()}], "provider_key")
        store.upsert("data_sources", [{
            "provider_key": snapshot.provider,
            "display_name": snapshot.provider,
            "last_attempt_at": datetime.now(timezone.utc).isoformat(),
            "last_success_at": datetime.now(timezone.utc).isoformat(),
            "last_data_at": snapshot.fetched_at,
            "last_success_rows": len(snapshot.matches),
            "last_error": None,
            "freshness_policy": "Monday/Wednesday Football-Data refresh; incremental local cache",
        }], "provider_key")
        store.record_ingestion_finish(run_id, status="succeeded", records_seen=len(snapshot.matches), records_upserted=upserted, newest_match_at=max((m.kickoff_utc for m in snapshot.matches), default=None))
        return {
            "provider": snapshot.provider,
            "records_seen": len(snapshot.matches),
            "records_upserted": upserted,
            "leagues_seen": len(actual),
            "season_refresh": season_result,
            "single_refresh": single_result,
            "predictions_resolved": resolved,
        }
    except Exception as exc:
        store.record_ingestion_finish(run_id, status="failed", records_seen=len(snapshot.matches), records_upserted=0, error_message=str(exc))
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=2, default=str))
