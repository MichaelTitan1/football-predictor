"""API-Football operational provider for fixtures, results, and standings only."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord
from .league_config import api_football_id_map, load_enabled_leagues
from .providers import ProviderSnapshot

logger = logging.getLogger(__name__)
API_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
REQUEST_TIMEOUT = 30

class APIFootballProvider:
    name = "api-football"

    def __init__(self, api_key: str | None = None, season: int | None = None):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "")
        if not self.api_key:
            raise RuntimeError("API_FOOTBALL_KEY is required for operational football updates")
        now = datetime.now(timezone.utc)
        self.season = season or (now.year if now.month >= 7 else now.year - 1)
        self._league_by_api_id = api_football_id_map()
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}
        self.match_metadata: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _team_key(name: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not value:
            raise ValueError("empty team name")
        return value

    def _get(self, path: str, **params: Any) -> Any:
        cache_key = (path, tuple(sorted(params.items())))
        if cache_key in self._cache:
            return self._cache[cache_key]
        response = requests.get(
            f"{API_BASE_URL}/{path.lstrip('/')}",
            headers={"x-apisports-key": self.api_key},
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"API-Football error for {path}: {payload['errors']}")
        data = payload.get("response", [])
        self._cache[cache_key] = data
        return data

    @staticmethod
    def _status(short: str) -> str:
        if short in {"FT", "AET", "PEN"}:
            return "finished"
        if short in {"PST", "TBD"}:
            return "postponed"
        if short in {"CANC", "ABD", "AWD", "WO"}:
            return "cancelled"
        return "scheduled"

    @staticmethod
    def _season_label(season_start: int) -> str:
        return f"{season_start}-{season_start + 1}"

    def fetch(self, mode: str = "fixtures") -> ProviderSnapshot:
        leagues: dict[str, LeagueRecord] = {}
        teams: dict[tuple[str, str], TeamRecord] = {}
        matches: dict[str, MatchRecord] = {}
        today = datetime.now(timezone.utc).date()
        date_params: dict[str, str] = {}
        if mode == "fixtures":
            date_params = {"from": today.isoformat(), "to": (today + timedelta(days=14)).isoformat()}
        elif mode == "results":
            date_params = {"from": (today - timedelta(days=3)).isoformat(), "to": today.isoformat()}

        for league in load_enabled_leagues():
            params = {"league": league.api_football_id, "season": self.season, **date_params}
            for item in self._get("fixtures", **params):
                fixture = item.get("fixture", {})
                api_league = item.get("league", {})
                api_teams = item.get("teams", {})
                goals = item.get("goals", {})
                league_cfg = self._league_by_api_id.get(int(api_league.get("id", league.api_football_id)))
                if not league_cfg:
                    continue
                lk = league_cfg.key
                home_name = str(api_teams.get("home", {}).get("name", "")).strip()
                away_name = str(api_teams.get("away", {}).get("name", "")).strip()
                if not home_name or not away_name:
                    continue
                hk, ak = self._team_key(home_name), self._team_key(away_name)
                leagues[lk] = LeagueRecord(lk, league_cfg.name, league_cfg.country)
                teams[(lk, hk)] = TeamRecord(hk, home_name, lk)
                teams[(lk, ak)] = TeamRecord(ak, away_name, lk)
                kickoff = str(fixture.get("date"))
                status = self._status(str(fixture.get("status", {}).get("short", "NS")))
                match = MatchRecord(
                    self.name, lk, self._season_label(int(api_league.get("season", self.season))), kickoff,
                    hk, ak, status,
                    goals.get("home") if status == "finished" else None,
                    goals.get("away") if status == "finished" else None,
                    str(fixture.get("id")),
                )
                matches[str(fixture.get("id"))] = match
                venue = fixture.get("venue", {}) or {}
                self.match_metadata[match.match_key] = {
                    "venue_name": venue.get("name"),
                    "venue_latitude": venue.get("lat"),
                    "venue_longitude": venue.get("lon"),
                }
        return ProviderSnapshot(tuple(leagues.values()), tuple(teams.values()), tuple(matches.values()), datetime.now(timezone.utc).isoformat(timespec="seconds"), self.name)

    def fetch_standings(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for league in load_enabled_leagues():
            for block in self._get("standings", league=league.api_football_id, season=self.season):
                for standing in block.get("league", {}).get("standings", [[]])[0]:
                    team = standing.get("team", {})
                    rows.append({
                        "league_canonical_key": league.key,
                        "team_canonical_key": f"{league.key}:{self._team_key(str(team.get('name', '')))}",
                        "season": str(self.season),
                        "rank": standing.get("rank"),
                        "points": standing.get("points"),
                        "played": standing.get("all", {}).get("played"),
                        "wins": standing.get("all", {}).get("win"),
                        "draws": standing.get("all", {}).get("draw"),
                        "losses": standing.get("all", {}).get("lose"),
                        "goals_for": standing.get("all", {}).get("goals", {}).get("for"),
                        "goals_against": standing.get("all", {}).get("goals", {}).get("against"),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
        return rows
