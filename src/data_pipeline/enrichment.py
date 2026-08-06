"""Operational enrichments from ClubElo and MET Norway.

ClubElo is a global daily club-strength snapshot rather than a league feed.
We ingest every club rating ClubElo publishes for the selected day. Fovra has
38 Football-Data leagues, but ClubElo does not cover every one of them.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from api.repository import FovraRepository
from .neon_store import NeonStore


def _team_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


CLUBELO_CACHE_PATH = Path(os.getenv("CLUBELO_CACHE_PATH", "data/raw/clubelo_latest.csv"))
CLUBELO_META_PATH = Path(os.getenv("CLUBELO_META_PATH", "data/raw/clubelo_latest.json"))
CLUBELO_TIMEOUT = int(os.getenv("CLUBELO_TIMEOUT_SECONDS", "60"))
CLUBELO_RETRIES = int(os.getenv("CLUBELO_RETRIES", "4"))
CLUBELO_LOOKBACK_DAYS = int(os.getenv("CLUBELO_LOOKBACK_DAYS", "7"))
CLUBELO_BASE_URLS = [
    value.rstrip("/")
    for value in os.getenv("CLUBELO_BASE_URLS", "https://api.clubelo.com,http://api.clubelo.com").split(",")
    if value.strip()
]


def _read_clubelo_csv(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    required = {"Club", "Country", "Elo", "From", "To"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ClubElo response missing columns: {sorted(missing)}")
    frame = frame[frame["Club"].notna()].copy()
    if frame.empty:
        raise ValueError("ClubElo response contained no club rows")
    return frame


def _fetch_clubelo_snapshot(target: date) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Fetch ClubElo with retries, protocol fallback, and recent-date fallback."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Fovra/1.0 (football prediction data refresh)"})
    last_error: str | None = None

    for days_back in range(CLUBELO_LOOKBACK_DAYS + 1):
        snapshot_date = target - timedelta(days=days_back)
        date_text = snapshot_date.isoformat()
        for base_url in CLUBELO_BASE_URLS:
            url = f"{base_url}/{date_text}"
            for attempt in range(1, CLUBELO_RETRIES + 1):
                try:
                    response = session.get(url, timeout=CLUBELO_TIMEOUT)
                    if response.status_code == 404:
                        last_error = f"HTTP 404 for {url}"
                        break
                    response.raise_for_status()
                    return _read_clubelo_csv(response.content), date_text, None
                except Exception as exc:
                    last_error = f"{url}: {exc}"
                    if attempt < CLUBELO_RETRIES:
                        time.sleep(min(2 ** (attempt - 1), 8))
    return None, None, last_error


def _load_cached_clubelo() -> tuple[pd.DataFrame | None, str | None]:
    if not CLUBELO_CACHE_PATH.exists():
        return None, None
    try:
        frame = _read_clubelo_csv(CLUBELO_CACHE_PATH.read_bytes())
        cached_date = None
        if CLUBELO_META_PATH.exists():
            cached_date = json.loads(CLUBELO_META_PATH.read_text(encoding="utf-8")).get("source_date")
        return frame, cached_date
    except Exception:
        return None, None


def _save_clubelo_cache(frame: pd.DataFrame, source_date: str, fetched_at: str) -> None:
    CLUBELO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CLUBELO_CACHE_PATH.with_suffix(".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(CLUBELO_CACHE_PATH)
    CLUBELO_META_PATH.write_text(
        json.dumps({"source_date": source_date, "fetched_at": fetched_at, "rows": len(frame)}, indent=2),
        encoding="utf-8",
    )


def _coverage(store: NeonStore, clubelo_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report direct team-name coverage without making missing ClubElo data fatal."""
    try:
        teams = store.select("teams", columns="canonical_key,name,league_canonical_key")
        strength_slugs = {_team_slug(str(row["team_name"])) for row in clubelo_rows if row.get("team_name")}
        covered_leagues = {
            str(row["league_canonical_key"])
            for row in teams
            if _team_slug(str(row.get("name", ""))) in strength_slugs
        }
        configured_leagues = {str(row["league_canonical_key"]) for row in teams}
        return {
            "configured_leagues_seen": len(configured_leagues),
            "leagues_with_direct_clubelo_name_match": len(covered_leagues),
        }
    except Exception:
        return {}


def refresh_clubelo(store: NeonStore | None = None) -> dict[str, Any]:
    """Refresh ClubElo without making a temporary source outage break Fovra.

    Fresh data is preferred. If ClubElo is temporarily unreachable, the last
    successful local snapshot is retained. If no local snapshot exists, the
    existing Neon team_strength table is retained as the final fallback.
    """
    store = store or NeonStore()
    store.initialize_schema()
    now = datetime.now(timezone.utc).isoformat()
    run_id = store.record_ingestion_start("clubelo")

    frame, source_date, live_error = _fetch_clubelo_snapshot(datetime.now(timezone.utc).date())
    fresh = frame is not None
    if frame is not None and source_date:
        _save_clubelo_cache(frame, source_date, now)
    else:
        frame, source_date = _load_cached_clubelo()

    if frame is None:
        existing = store.select(
            "team_strength",
            columns="team_slug,team_name,elo,rank,country,level,from_date,to_date,source_provider,updated_at",
        )
        if existing:
            store.upsert("provider_sources", [{
                "provider_key": "clubelo",
                "display_name": "ClubElo",
                "source_type": "team-strength",
                "updated_at": now,
            }], "provider_key")
            # Do not overwrite last_success_at when the live source is down.
            store.upsert("data_sources", [{
                "provider_key": "clubelo",
                "display_name": "ClubElo",
                "last_attempt_at": now,
                "last_data_at": source_date,
                "last_success_rows": len(existing),
                "last_error": live_error,
                "freshness_policy": "daily snapshot; retain last known data on source outage",
            }], "provider_key")
            store.record_ingestion_finish(run_id, status="succeeded", records_seen=len(existing), records_upserted=0, error_message=live_error)
            return {
                "provider": "clubelo",
                "rows": len(existing),
                "fresh": False,
                "source_date": source_date,
                "skipped": "ClubElo temporarily unavailable; retained existing Neon team strength",
                "error": live_error,
            }
        store.record_ingestion_finish(run_id, status="succeeded", records_seen=0, records_upserted=0, error_message=live_error)
        return {
            "provider": "clubelo",
            "rows": 0,
            "fresh": False,
            "skipped": "ClubElo unavailable and no previous snapshot exists yet",
            "error": live_error,
        }

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        club = row.get("Club")
        if not club or str(club) == "nan":
            continue
        elo = pd.to_numeric(pd.Series([row.get("Elo")]), errors="coerce").iloc[0]
        rank = pd.to_numeric(pd.Series([row.get("Rank")]), errors="coerce").iloc[0]
        level = row.get("Level")
        rows.append({
            "team_slug": _team_slug(str(club)),
            "team_name": str(club),
            "elo": float(elo) if pd.notna(elo) else None,
            "rank": int(rank) if pd.notna(rank) else None,
            "country": row.get("Country"),
            "level": str(level) if pd.notna(level) else None,
            "from_date": str(row.get("From")) if pd.notna(row.get("From")) else None,
            "to_date": str(row.get("To")) if pd.notna(row.get("To")) else None,
            "source_provider": "clubelo-soccerdata",
            "updated_at": now,
        })

    store.upsert("team_strength", rows, "team_slug")
    store.upsert("provider_sources", [{
        "provider_key": "clubelo",
        "display_name": "ClubElo",
        "source_type": "team-strength",
        "updated_at": now,
    }], "provider_key")
    store.upsert("data_sources", [{
        "provider_key": "clubelo",
        "display_name": "ClubElo",
        "last_attempt_at": now,
        "last_success_at": now,
        "last_data_at": source_date,
        "last_success_rows": len(rows),
        "last_error": None,
        "freshness_policy": "daily snapshot; retain last known data on source outage",
    }], "provider_key")
    store.record_ingestion_finish(run_id, status="succeeded", records_seen=len(rows), records_upserted=len(rows), newest_match_at=None)
    return {
        "provider": "clubelo",
        "rows": len(rows),
        "fresh": fresh,
        "source_date": source_date,
        "updated_at": now,
        "coverage": _coverage(store, rows),
    }


def refresh_weather(store: NeonStore | None = None, limit: int = 200) -> dict[str, Any]:
    store = store or NeonStore()
    repo = FovraRepository(store)
    now = datetime.now(timezone.utc)
    fixtures = repo._select("matches", params={
        "kickoff_at": f"gte.{now.isoformat()}",
        "and": f"(kickoff_at.lt.{(now + timedelta(hours=24)).isoformat()})",
        "status": "eq.scheduled",
        "order": "kickoff_at.asc",
        "limit": str(limit),
    })
    rows = []
    user_agent = os.getenv("MET_NORWAY_USER_AGENT", "Fovra/1.0 contact@example.com")
    for match in fixtures:
        lat, lon = match.get("venue_latitude"), match.get("venue_longitude")
        if lat is None or lon is None:
            continue
        payload = requests.get(
            "https://api.met.no/weatherapi/locationforecast/2.0/compact",
            params={"lat": lat, "lon": lon},
            headers={"User-Agent": user_agent},
            timeout=30,
        ).json()
        timeseries = payload.get("properties", {}).get("timeseries", [])
        if not timeseries:
            continue
        target = datetime.fromisoformat(str(match["kickoff_at"]).replace("Z", "+00:00"))
        nearest = min(timeseries, key=lambda x: abs(datetime.fromisoformat(x["time"].replace("Z", "+00:00")) - target))
        details = nearest.get("data", {}).get("instant", {}).get("details", {})
        summary = (nearest.get("data", {}).get("next_1_hours") or nearest.get("data", {}).get("next_6_hours") or {}).get("summary", {})
        rows.append({
            "match_canonical_key": match["canonical_key"],
            "forecast_at": nearest.get("time"),
            "temperature_c": details.get("air_temperature"),
            "rain_mm": details.get("precipitation_amount", 0),
            "wind_mps": details.get("wind_speed"),
            "condition": summary.get("symbol_code"),
            "updated_at": now.isoformat(),
        })
    store.upsert("match_weather", rows, "match_canonical_key")
    return {"provider": "met-norway", "fixtures_checked": len(fixtures), "rows": len(rows)}
