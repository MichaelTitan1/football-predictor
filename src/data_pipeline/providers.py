"""Replaceable provider boundary for Fovra's canonical data layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord


@dataclass(frozen=True)
class ProviderSnapshot:
    leagues: tuple[LeagueRecord, ...]
    teams: tuple[TeamRecord, ...]
    matches: tuple[MatchRecord, ...]
    fetched_at: str
    provider: str


class FootballDataProvider(Protocol):
    name: str

    def fetch(self) -> ProviderSnapshot:
        """Fetch a validated snapshot. Never return simulated football data."""
        ...
