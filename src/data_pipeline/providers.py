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

class FootballDataProviderProtocol(Protocol):
    name: str
    def fetch(self) -> ProviderSnapshot: ...
