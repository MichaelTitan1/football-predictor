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
        self._unavailable_seasons: dict[str, set[int]] = {}
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
            raise APIFootballProviderError(path, payload["errors"])
        data = payload.get("response", [])
        self._cache[cache_key] = data
        return data

    @staticmethod
    def _is_season_plan_error(exc: APIFootballProviderError) -> bool:
        text = str(exc.errors).lower()
        return "plan" in text and "season" in text

    @staticmethod
    def _current_season() -> int:
        now = datetime.now(timezone.utc)
        return now.year if now.month >= 7 else now.year - 1

    @staticmethod
    def _fallback_season_from_error(exc: APIFootballProviderError, attempted_season: int) -> int | None:
        years = [int(year) for year in re.findall(r"\b(20\d{2})\b", str(exc.errors))]
        candidates = [year for year in years if year < attempted_season]
        return max(candidates) if candidates else None

    def season_for_league(self, league: LeagueConfig) -> int:
        if league.key not in self._season_cache:
            season = self.preferred_season if self.preferred_season is not None else self._current_season()
            self._season_cache[league.key] = season
            logger.info("League %s using configured season %s", league.key, season)
        return self._season_cache[league.key]

    def _mark_season_unavailable(
        self,
        league: LeagueConfig,
        season: int,
        exc: APIFootballProviderError,
    ) -> int | None:
        unavailable_by_league = getattr(self, "_unavailable_seasons", None)
        if unavailable_by_league is None:
            unavailable_by_league = self._unavailable_seasons = {}
        unavailable = unavailable_by_league.setdefault(league.key, set())
        if season in unavailable:
            logger.warning("Season %s already failed for league %s during this run", season, league.key)
            return None
        unavailable.add(season)

        fallback = self._fallback_season_from_error(exc, season)
        if fallback is None or fallback in unavailable:
            logger.warning("No API-Football free-plan fallback season found for league %s", league.key)
            self._season_cache.pop(league.key, None)
            return None

        self._season_cache[league.key] = fallback
        logger.info("League %s using fallback season %s", league.key, fallback)
        return fallback

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
                fallback = self._mark_season_unavailable(league, season, exc)
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
                fallback = self._mark_season_unavailable(league, season, exc)
                if fallback is None:
                    return []

    @staticmethod
    def _batch_index(total: int, rotation: str) -> int:
        now = datetime.now(timezone.utc)
        if rotation == "hourly":
            return (now.hour // max(1, 24 // total)) % total
        return now.toordinal() % total

    def _operational_leagues(self) -> tuple[LeagueConfig, ...]:
        leagues = load_enabled_leagues()
        total_text = os.getenv("FOVRA_API_FOOTBALL_BATCH_TOTAL", "1")
        try:
            total = max(1, int(total_text))
        except ValueError:
            logger.warning("Ignoring invalid FOVRA_API_FOOTBALL_BATCH_TOTAL=%s", total_text)
            return leagues
        if total == 1:
            return leagues

        index_text = os.getenv("FOVRA_API_FOOTBALL_BATCH_INDEX")
        if index_text is None:
            rotation = os.getenv("FOVRA_API_FOOTBALL_BATCH_ROTATION", "daily").strip().lower()
            index = self._batch_index(total, rotation)
        else:
            try:
                index = int(index_text)
            except ValueError:
                logger.warning("Ignoring invalid FOVRA_API_FOOTBALL_BATCH_INDEX=%s", index_text)
                return leagues

        index %= total
        selected = tuple(league for offset, league in enumerate(leagues) if offset % total == index)
        logger.info("API-Football league batch %s/%s selected %s of %s leagues", index + 1, total, len(selected), len(leagues))
        return selected

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
                        if self._mark_season_unavailable(league, season, exc) is not None:
                            pending.append(league)
        return batches

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

        fixture_batches = []
        if mode == "results":
            fixture_batches = self._get_results_fixture_batches((today - timedelta(days=1), today))
        else:
            for league in self._operational_leagues():
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
                    self.name, lk, self._season_label(season), kickoff,
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
        for league in self._operational_leagues():
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
