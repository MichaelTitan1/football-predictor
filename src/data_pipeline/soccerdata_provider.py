"""Optional Soccerdata enrichments for FBref team stats and ClubElo strength.

The provider is intentionally best-effort: unsupported leagues or missing
`soccerdata` installs are reported as skipped rows instead of failing the full
pipeline. League coverage is driven exclusively by config/leagues.json.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .league_config import load_enabled_leagues
from .supabase_store import SupabaseStore

logger = logging.getLogger(__name__)

FBREF_STAT_COLUMNS = {
    "xg": ("xg", "expected_goals", "Expected Goals"),
    "xga": ("xga", "expected_goals_against", "Expected Goals Against"),
    "goals": ("goals", "gf", "Gls"),
    "shots": ("shots", "sh", "Sh"),
    "shots_on_target": ("shots_on_target", "sot", "SoT"),
    "possession": ("possession", "poss", "Poss"),
    "passes": ("passes", "passes_completed", "Cmp"),
    "pass_accuracy": ("pass_accuracy", "cmp%", "Cmp%"),
    "defensive_actions": ("defensive_actions", "tkl", "Tkl"),
    "goalkeeper_stats": ("goalkeeper_stats", "saves", "Save%"),
}


def _slug(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _first(row: Any, names: tuple[str, ...]) -> Any:
    lower = {str(k).lower(): k for k in getattr(row, "index", [])}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            value = row.get(key)
            if value == value:
                return value
    return None


def refresh_fbref_team_stats(store: SupabaseStore | None = None, seasons: str | int | None = None) -> dict[str, Any]:
    store = store or SupabaseStore()
    try:
        import soccerdata as sd  # type: ignore
    except ImportError:
        return {"provider": "soccerdata-fbref", "rows": 0, "skipped": "soccerdata is not installed"}

    now = datetime.now(timezone.utc).isoformat()
    season = seasons or datetime.now(timezone.utc).year
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for league in load_enabled_leagues():
        fbref_name = league.fbref_name or league.name
        try:
            fbref = sd.FBref(leagues=fbref_name, seasons=season)
            frames = []
            for stat in ("standard", "shooting", "passing", "defense", "keeper"):
                try:
                    frames.append(fbref.read_team_season_stats(stat_type=stat))
                except Exception as exc:
                    logger.info("FBref %s %s unavailable: %s", fbref_name, stat, exc)
            if not frames:
                skipped.append(league.key)
                continue
            import pandas as pd
            df = pd.concat(frames, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            for idx, row in df.reset_index().iterrows():
                team = str(_first(row, ("team", "squad", "Team", "Squad")) or row.get("team") or row.get("squad") or idx)
                out = {"league_canonical_key": league.key, "team_slug": _slug(team), "team_name": team, "season": str(season), "updated_at": now, "raw_stats": row.dropna().to_dict()}
                for target, aliases in FBREF_STAT_COLUMNS.items():
                    out[target] = _first(row, aliases)
                rows.append(out)
        except Exception as exc:
            logger.info("Skipping unsupported FBref league %s: %s", fbref_name, exc)
            skipped.append(league.key)
    store.upsert("team_statistics", rows, "league_canonical_key,team_slug,season")
    return {"provider": "soccerdata-fbref", "rows": len(rows), "skipped_leagues": skipped, "updated_at": now}
