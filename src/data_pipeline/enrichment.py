"""Operational enrichments from ClubElo and MET Norway."""
from __future__ import annotations

import csv
import io
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
    response = requests.get("http://api.clubelo.com/", timeout=30)
    response.raise_for_status()
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in csv.DictReader(io.StringIO(response.text)):
        club = row.get("Club")
        if not club:
            continue
        rows.append({
            "team_slug": _team_slug(club),
            "team_name": club,
            "elo": float(row["Elo"]) if row.get("Elo") else None,
            "rank": int(row["Rank"]) if row.get("Rank") else None,
            "country": row.get("Country"),
            "level": row.get("Level"),
            "from_date": row.get("From"),
            "to_date": row.get("To"),
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
