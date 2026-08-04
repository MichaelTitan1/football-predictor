from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.data_pipeline.neon_store import NeonStore


class FovraRepository:
    """Website read path backed only by canonical Neon PostgreSQL."""

    def __init__(self, store: NeonStore | None = None):
        self.store = store or NeonStore()
        self.timezone = ZoneInfo(os.getenv("FOVRA_TIMEZONE", "Europe/Athens"))

    def _select(self, table: str, *, select: str = "*", params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        # Compatibility for older callers/tests: prefer explicit repository methods for new code.
        return self.store.select(table, columns=select)

    def health(self) -> dict[str, Any]:
        sources = self.store.select("data_sources", columns="provider_key,last_success_at,last_data_at,last_error")
        models = self.store.select("model_versions", "is_active = %s", (True,), columns="version,is_active")
        return {"database": "ok", "providers": sources, "active_models": models}

    def matches_today(self, limit: int = 100) -> list[dict[str, Any]]:
        now_local = datetime.now(self.timezone)
        start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        return self.store.select("matches", "kickoff_at >= %s and kickoff_at < %s", (start, end), "kickoff_at asc", limit)

    def upcoming(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.select("matches", "kickoff_at >= %s and status = %s", (datetime.now(timezone.utc), "scheduled"), "kickoff_at asc", limit)

    def match(self, canonical_key: str) -> dict[str, Any] | None:
        rows = self.store.select("matches", "canonical_key = %s", (canonical_key,), limit=1)
        return rows[0] if rows else None

    def predictions(self, canonical_key: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if canonical_key:
            return self.store.select("predictions", "match_canonical_key = %s", (canonical_key,), "predicted_at desc", limit)
        return self.store.select("predictions", order="predicted_at desc", limit=limit)

    def archive(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.select("prediction_archive", order="predicted_at desc", limit=limit)

    def teams(self, league: str | None = None) -> list[dict[str, Any]]:
        if league:
            return self.store.select("teams", "league_canonical_key = %s", (league,), "name asc")
        return self.store.select("teams", order="name asc")

    def leagues(self) -> list[dict[str, Any]]:
        return self.store.select("leagues", order="name asc")

    def team_matches(self, team_key: str, limit: int = 200) -> list[dict[str, Any]]:
        return self.store.select("matches", "(home_team_canonical_key = %s or away_team_canonical_key = %s) and status = %s", (team_key, team_key, "finished"), "kickoff_at desc", limit)

    def h2h(self, home_team: str, away_team: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.select("matches", "((home_team_canonical_key = %s and away_team_canonical_key = %s) or (home_team_canonical_key = %s and away_team_canonical_key = %s)) and status = %s", (home_team, away_team, away_team, home_team, "finished"), "kickoff_at desc", limit)

    def standings(self, league_key: str) -> list[dict[str, Any]]:
        persisted = self.store.select("league_standings", "league_canonical_key = %s", (league_key,), "rank asc")
        if persisted:
            return persisted
        matches = self.store.select("matches", "league_canonical_key = %s and status = %s", (league_key, "finished"), "kickoff_at asc", 10000)
        teams: dict[str, dict[str, Any]] = {}
        for m in matches:
            for side in ("home", "away"):
                key = m.get(f"{side}_team_canonical_key")
                if key:
                    teams.setdefault(key, {"team": key, "played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "points": 0})
            h, a = m.get("home_team_canonical_key"), m.get("away_team_canonical_key")
            hs, aas = m.get("home_score"), m.get("away_score")
            if hs is None or aas is None or h not in teams or a not in teams:
                continue
            teams[h]["played"] += 1; teams[a]["played"] += 1
            teams[h]["gf"] += hs; teams[h]["ga"] += aas; teams[a]["gf"] += aas; teams[a]["ga"] += hs
            if hs > aas:
                teams[h]["wins"] += 1; teams[h]["points"] += 3; teams[a]["losses"] += 1
            elif hs < aas:
                teams[a]["wins"] += 1; teams[a]["points"] += 3; teams[h]["losses"] += 1
            else:
                teams[h]["draws"] += 1; teams[a]["draws"] += 1; teams[h]["points"] += 1; teams[a]["points"] += 1
        out = list(teams.values())
        for row in out:
            row["gd"] = row["gf"] - row["ga"]
        return sorted(out, key=lambda x: (-x["points"], -x["gd"], -x["gf"], x["team"]))

    def freshness(self) -> list[dict[str, Any]]:
        return self.store.select("data_sources", columns="provider_key,display_name,last_attempt_at,last_success_at,last_data_at,last_success_rows,last_error,freshness_policy")
