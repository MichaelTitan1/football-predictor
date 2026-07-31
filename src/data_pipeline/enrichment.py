"""Operational enrichments from ClubElo and MET Norway."""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from api.repository import FovraRepository
from .supabase_store import SupabaseStore


def _team_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def refresh_clubelo(store: SupabaseStore | None = None) -> dict[str, Any]:
    store = store or SupabaseStore()
    try:
        import soccerdata as sd  # type: ignore
    except ImportError:
        return {"provider": "clubelo", "rows": 0, "skipped": "soccerdata is not installed"}

    now = datetime.now(timezone.utc).isoformat()
    try:
        frame = sd.ClubElo().read_by_date(now[:10])
    except Exception as exc:
        return {"provider": "clubelo", "rows": 0, "skipped": f"ClubElo unavailable through soccerdata: {exc}"}

    rows = []
    for _, row in frame.reset_index().iterrows():
        club = row.get("Club") or row.get("club") or row.get("team") or row.get("Team")
        if not club or str(club) == "nan":
            continue
        elo = row.get("Elo") or row.get("elo")
        rank = row.get("Rank") or row.get("rank")
        level = row.get("Level") or row.get("level")
        rows.append({
            "team_slug": _team_slug(str(club)),
            "team_name": str(club),
            "elo": float(elo) if elo == elo and elo is not None else None,
            "rank": int(rank) if rank == rank and rank is not None else None,
            "country": row.get("Country") or row.get("country"),
            "level": level if level == level else None,
            "from_date": row.get("From") or row.get("from"),
            "to_date": row.get("To") or row.get("to"),
            "updated_at": now,
        })
    store.upsert("team_strength", rows, "team_slug")
    return {"provider": "clubelo", "rows": len(rows), "updated_at": now}


def refresh_weather(store: SupabaseStore | None = None, limit: int = 200) -> dict[str, Any]:
    store = store or SupabaseStore()
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
