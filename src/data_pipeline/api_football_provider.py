"""API-Football operational provider for fixtures, results, and standings only."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord
from .league_config import LeagueConfig, api_football_id_map, load_enabled_leagues
from .providers import ProviderSnapshot

logger = logging.getLogger(__name__)
API_BASE_URL = os.getenv("API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io")
REQUEST_TIMEOUT = 30

class APIFootballProviderError(RuntimeError):
    def __init__(self, path: str, errors: Any):
        super().__init__(f"API-Football error for {path}: {errors}")
        self.path = path
        self.errors = errors


class APIFootballProvider:
    name = "api-football"

    def __init__(self, api_key: str | None = None, season: int | None = None):
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "")
        if not self.api_key:
            raise RuntimeError("API_FOOTBALL_KEY is required for operational football updates")
        self.preferred_season = season
        self._league_by_api_id = api_football_id_map()
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Any] = {}
        self._season_cache: dict[str, int] = {}
        self._season_candidates_cache: dict[str, list[int]] = {}
        self.match_metadata: dict[str, dict[str, Any]] = {}
        self.request_count = 0

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
        self.request_count += 1
        response = requests.get(
            f"{API_BASE_URL}/{path.lstrip('/')}",
            headers={"x-apisports-key": self.api_key},
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
        if payload.get("errors"):
            raise APIFootballProviderError(path, payload["errors"])
        data = payload.get("response", [])
        self._cache[cache_key] = data
        return data

    @staticmethod
    def _is_season_plan_error(exc: APIFootballProviderError) -> bool:
        text = str(exc.errors).lower()
        return "plan" in text and "season" in text

    def _discover_season_candidates(self, league: LeagueConfig) -> list[int]:
        if league.key in self._season_candidates_cache:
            return self._season_candidates_cache[league.key]
        seasons: list[int] = []
        for row in self._get("leagues", id=league.api_football_id):
            for season_info in row.get("seasons", []):
                year = season_info.get("year")
                coverage = season_info.get("coverage", {})
                fixtures = coverage.get("fixtures", {}) if isinstance(coverage, dict) else {}
                if isinstance(year, int) and fixtures:
                    seasons.append(year)
        seasons = sorted(set(seasons), reverse=True)
        if self.preferred_season is not None:
            seasons = [season for season in seasons if season <= self.preferred_season]
        if not seasons:
            raise RuntimeError(f"No API-Football seasons discovered for league {league.key}")
        self._season_candidates_cache[league.key] = seasons
        return seasons

    def season_for_league(self, league: LeagueConfig) -> int:
        if league.key not in self._season_cache:
            season = self._discover_season_candidates(league)[0]
            self._season_cache[league.key] = season
            logger.info("League %s using season %s", league.key, season)
        return self._season_cache[league.key]

    def _mark_season_unavailable(self, league: LeagueConfig, season: int) -> int | None:
        candidates = [candidate for candidate in self._discover_season_candidates(league) if candidate < season]
        self._season_candidates_cache[league.key] = candidates
        self._season_cache.pop(league.key, None)
        if not candidates:
            logger.warning("No API-Football free-plan season remains available for league %s", league.key)
            return None
        self._season_cache[league.key] = candidates[0]
        logger.info("League %s using season %s", league.key, candidates[0])
        return candidates[0]

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

    def _get_league_fixtures(self, league: LeagueConfig, **params: Any) -> list[dict[str, Any]]:
        while True:
            season = self.season_for_league(league)
            try:
                return self._get("fixtures", league=league.api_football_id, season=season, **params)
            except APIFootballProviderError as exc:
                if not self._is_season_plan_error(exc):
                    raise
                fallback = self._mark_season_unavailable(league, season)
                if fallback is None:
                    return []

    def _get_standings(self, league: LeagueConfig) -> list[dict[str, Any]]:
        while True:
            season = self.season_for_league(league)
            try:
                return self._get("standings", league=league.api_football_id, season=season)
            except APIFootballProviderError as exc:
                if not self._is_season_plan_error(exc):
                    raise
                fallback = self._mark_season_unavailable(league, season)
                if fallback is None:
                    return []

    def _get_results_fixture_batches(self, days: tuple[Any, ...]) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        pending = list(load_enabled_leagues())
        while pending:
            leagues_by_season: dict[int, list[LeagueConfig]] = {}
            for league in pending:
                leagues_by_season.setdefault(self.season_for_league(league), []).append(league)
            pending = []
            for season, leagues in leagues_by_season.items():
                try:
                    for day in days:
                        batches.append(self._get("fixtures", season=season, date=day.isoformat()))
                except APIFootballProviderError as exc:
                    if not self._is_season_plan_error(exc):
                        raise
                    for league in leagues:
                        if self._mark_season_unavailable(league, season) is not None:
                            pending.append(league)
        return batches

    def fetch(self, mode: str = "fixtures") -> ProviderSnapshot:

        fixture_batches = []
        if mode == "results":
            fixture_batches = self._get_results_fixture_batches((today - timedelta(days=1), today))
        else:
            for league in load_enabled_leagues():
                fixture_batches.append(self._get_league_fixtures(league, **date_params))

        for batch in fixture_batches:
            for item in batch:
                fixture = item.get("fixture", {})
                api_league = item.get("league", {})
                api_teams = item.get("teams", {})
                goals = item.get("goals", {})
                league_id = api_league.get("id")
                if league_id is None:
                    continue
                league_cfg = self._league_by_api_id.get(int(league_id))
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
                season = int(api_league.get("season") or self.season_for_league(league_cfg))
                kickoff = str(fixture.get("date"))
                status = self._status(str(fixture.get("status", {}).get("short", "NS")))
                match = MatchRecord(
                    self.name,
                    lk,
                    self._season_label(season),
                    kickoff,
                    hk,
                    ak,
                    status,
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
        return ProviderSnapshot(
            tuple(leagues.values()),
            tuple(teams.values()),
            tuple(matches.values()),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            self.name,
        )

    def fetch_standings(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for league in load_enabled_leagues():
            for block in self._get_standings(league):
                season = self.season_for_league(league)
                for standing in block.get("league", {}).get("standings", [[]])[0]:
                    team = standing.get("team", {})
                    rows.append({
                        "league_canonical_key": league.key,
                        "team_canonical_key": f"{league.key}:{self._team_key(str(team.get('name', '')))}",
                        "season": str(season),
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