from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.data_pipeline.supabase_store import SupabaseStore


class FovraRepository:
    def __init__(self, store: SupabaseStore | None = None):
        self.store = store or SupabaseStore()
        self.timezone = ZoneInfo(os.getenv("FOVRA_TIMEZONE", "Europe/Athens"))

    def _select(self, table: str, *, select: str = "*", params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        params = dict(params or {})
        params["select"] = select
        return self.store._request("GET", table, params=params) or []

    def health(self) -> dict[str, Any]:
        sources = self._select("data_sources", select="provider_key,last_success_at,last_data_at,last_error")
        models = self._select("model_versions", select="version,is_active", params={"is_active": "eq.true"})
        return {"database": "ok", "providers": sources, "active_models": models}

    def matches_today(self, limit: int = 100) -> list[dict[str, Any]]:
        now_local = datetime.now(self.timezone)
        start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        return self._select("matches", params={"kickoff_at": f"gte.{start.isoformat()}", "and": f"(kickoff_at.lt.{end.isoformat()})", "order": "kickoff_at.asc", "limit": str(limit)})

    def upcoming(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._select("matches", params={"kickoff_at": f"gte.{datetime.now(timezone.utc).isoformat()}", "status": "eq.scheduled", "order": "kickoff_at.asc", "limit": str(limit)})

    def match(self, canonical_key: str) -> dict[str, Any] | None:
        rows = self._select("matches", params={"canonical_key": f"eq.{canonical_key}", "limit": "1"})
        return rows[0] if rows else None

    def predictions(self, canonical_key: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        params = {"order": "predicted_at.desc", "limit": str(limit)}
        if canonical_key:
            params["match_canonical_key"] = f"eq.{canonical_key}"
        return self._select("predictions", params=params)

    def archive(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._select("prediction_archive", params={"order": "predicted_at.desc", "limit": str(limit)})

    def teams(self, league: str | None = None) -> list[dict[str, Any]]:
        params = {"order": "name.asc"}
        if league:
            params["league_canonical_key"] = f"eq.{league}"
        return self._select("teams", params=params)

    def leagues(self) -> list[dict[str, Any]]:
        return self._select("leagues", params={"order": "name.asc"})

    def team_matches(self, team_key: str, limit: int = 200) -> list[dict[str, Any]]:
        return self._select("matches", params={"or": f"(home_team_canonical_key.eq.{team_key},away_team_canonical_key.eq.{team_key})", "status": "eq.finished", "order": "kickoff_at.desc", "limit": str(limit)})

    def h2h(self, home_team: str, away_team: str, limit: int = 50) -> list[dict[str, Any]]:
        params = {
            "or": f"(and(home_team_canonical_key.eq.{home_team},away_team_canonical_key.eq.{away_team}),and(home_team_canonical_key.eq.{away_team},away_team_canonical_key.eq.{home_team}))",
            "status": "eq.finished",
            "order": "kickoff_at.desc",
            "limit": str(limit),
        }
        return self._select("matches", params=params)

    def standings(self, league_key: str) -> list[dict[str, Any]]:
        persisted = self._select("league_standings", params={"league_canonical_key": f"eq.{league_key}", "order": "rank.asc"})
        if persisted:
            return persisted
        matches = self._select("matches", params={"league_canonical_key": f"eq.{league_key}", "status": "eq.finished", "order": "kickoff_at.asc", "limit": "10000"})
        teams: dict[str, dict[str, Any]] = {}
        for m in matches:
            for side in ("home", "away"):
                key = m.get(f"{side}_team_canonical_key")
                if not key:
                    continue
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
        return self._select("data_sources", select="provider_key,display_name,last_attempt_at,last_success_at,last_data_at,last_success_rows,last_error,freshness_policy")
