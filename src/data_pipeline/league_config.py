"""Single configuration source for enabled Fovra V1 leagues."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "leagues.json"

@dataclass(frozen=True)
class LeagueConfig:
    key: str
    name: str
    tier: int
    football_data_code: str | None = None
    api_football_id: int | None = None
    country: str = ""
    strength: float = 1.0
    fbref_name: str | None = None
    clubelo_country: str | None = None

@lru_cache(maxsize=1)
def load_enabled_leagues(path: str | Path = CONFIG_PATH) -> tuple[LeagueConfig, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    leagues = tuple(LeagueConfig(**row) for row in data.get("enabled_leagues", []))
    if not leagues:
        raise ValueError("config/leagues.json must enable at least one league")
    keys = [l.key for l in leagues]
    if len(set(keys)) != len(keys):
        raise ValueError("enabled league keys must be unique")
    return leagues

def league_config_map() -> dict[str, dict[str, Any]]:
    return {
        l.key: {
            "name": l.name,
            "code": l.football_data_code,
            "tier": l.tier,
            "strength": l.strength,
            "api_football_id": l.api_football_id,
            "country": l.country,
            "fbref_name": l.fbref_name or l.name,
            "clubelo_country": l.clubelo_country,
        }
        for l in load_enabled_leagues()
    }

def api_football_id_map() -> dict[int, LeagueConfig]:
    return {int(l.api_football_id): l for l in load_enabled_leagues() if l.api_football_id is not None}
