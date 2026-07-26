"""Football-Data.co.uk adapter for the canonical Fovra V1 data layer.

Free public source used for historical results and the current fixture feed.
The adapter is deliberately isolated behind providers.FootballDataProvider so
another source can be added without changing the canonical schema or storage.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord
from .data_downloader import LEAGUE_CONFIG
from .providers import ProviderSnapshot

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
REQUEST_TIMEOUT = 30


class FootballDataProvider:
    name = "football-data.co.uk"

    def __init__(self, raw_dir: str | Path = "data/raw", include_remote: bool = True):
        self.raw_dir = Path(raw_dir)
        self.include_remote = include_remote

    @staticmethod
    def _season_code(year: int) -> str:
        return f"{year % 100:02d}{(year + 1) % 100:02d}"

    @staticmethod
    def _team_key(name: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not value:
            raise ValueError(f"invalid team name: {name!r}")
        return value

    @staticmethod
    def _parse_datetime(row: pd.Series) -> str:
        raw_date = row.get("Date")
        raw_time = row.get("Time")
        if pd.isna(raw_date):
            raise ValueError("match has no date")
        text = str(raw_date).strip()
        if pd.isna(raw_time) or not str(raw_time).strip():
            parsed = pd.to_datetime(text, dayfirst=True, errors="raise")
        else:
            parsed = pd.to_datetime(f"{text} {str(raw_time).strip()}", dayfirst=True, errors="raise")
        # Football-Data fixture times are UK-local. Use Europe/London when
        # zoneinfo is available, otherwise retain an explicit UTC conversion.
        from zoneinfo import ZoneInfo
        parsed = parsed.replace(tzinfo=ZoneInfo("Europe/London"))
        return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _read_csv_bytes(data: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(data))

    def _get(self, url: str) -> Optional[bytes]:
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            logger.warning("Football-Data fetch failed: %s (%s)", url, exc)
            return None

    def _local_frames(self) -> list[pd.DataFrame]:
        if not self.raw_dir.exists():
            return []
        frames: list[pd.DataFrame] = []
        for path in sorted(self.raw_dir.glob("*.csv")):
            try:
                frames.append(pd.read_csv(path))
            except Exception as exc:
                logger.warning("Skipping unreadable raw file %s: %s", path, exc)
        return frames

    def _remote_frames(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        current_year = datetime.now(timezone.utc).year
        season_year = current_year if datetime.now(timezone.utc).month >= 7 else current_year - 1

        # Refresh the active season even if a stale copy already exists locally.
        for league_key, info in LEAGUE_CONFIG.items():
            url = f"{BASE_URL}/{self._season_code(season_year)}/{info['code']}.csv"
            data = self._get(url)
            if data:
                try:
                    df = self._read_csv_bytes(data)
                    df["League"] = league_key
                    frames.append(df)
                except Exception as exc:
                    logger.warning("Could not parse %s: %s", url, exc)

        # The public fixture feed is the source for upcoming matches. It may
        # contain several leagues and is therefore mapped by Div/code.
        data = self._get(FIXTURES_URL)
        if data:
            try:
                fixtures = self._read_csv_bytes(data)
                frames.append(fixtures)
            except Exception as exc:
                logger.warning("Could not parse fixture feed: %s", exc)
        return frames

    def _normalize(self, frames: list[pd.DataFrame]) -> tuple[list[LeagueRecord], list[TeamRecord], list[MatchRecord]]:
        leagues: dict[str, LeagueRecord] = {}
        teams: dict[tuple[str, str], TeamRecord] = {}
        matches: dict[str, MatchRecord] = {}
        code_to_key = {info["code"]: key for key, info in LEAGUE_CONFIG.items()}

        for raw in frames:
            if raw is None or raw.empty:
                continue
            df = raw.copy()
            df.columns = [str(c).strip() for c in df.columns]
            league_value = df.get("League")
            if league_value is None:
                league_value = df.get("Div")
            if league_value is None:
                continue

            for idx, row in df.iterrows():
                code = str(league_value.loc[idx]).strip() if idx in league_value.index else ""
                league_key = code if code in LEAGUE_CONFIG else code_to_key.get(code)
                if league_key is None:
                    continue
                info = LEAGUE_CONFIG[league_key]
                home = str(row.get("HomeTeam", "")).strip()
                away = str(row.get("AwayTeam", "")).strip()
                if not home or not away or home == "nan" or away == "nan":
                    continue
                try:
                    kickoff = self._parse_datetime(row)
                except Exception:
                    continue

                leagues[league_key] = LeagueRecord(league_key, info["name"], None)
                home_key = self._team_key(home)
                away_key = self._team_key(away)
                teams[(league_key, home_key)] = TeamRecord(home_key, home, league_key)
                teams[(league_key, away_key)] = TeamRecord(away_key, away, league_key)

                fthg = pd.to_numeric(pd.Series([row.get("FTHG")]), errors="coerce").iloc[0]
                ftag = pd.to_numeric(pd.Series([row.get("FTAG")]), errors="coerce").iloc[0]
                ftr = str(row.get("FTR", "")).strip().upper()
                if ftr not in {"H", "D", "A"}:
                    ftr = ""
                finished = pd.notna(fthg) and pd.notna(ftag) and bool(ftr)
                status = "finished" if finished else "scheduled"
                match = MatchRecord(
                    provider=self.name,
                    league_key=league_key,
                    season=None,
                    kickoff_utc=kickoff,
                    home_team=home_key,
                    away_team=away_key,
                    status=status,
                    home_score=int(fthg) if pd.notna(fthg) else None,
                    away_score=int(ftag) if pd.notna(ftag) else None,
                    source_id=str(row.get("MatchID")) if pd.notna(row.get("MatchID")) else None,
                )
                matches[match.match_key] = match

        return list(leagues.values()), list(teams.values()), list(matches.values())

    def fetch(self) -> ProviderSnapshot:
        frames = self._local_frames()
        local_count = len(frames)
        if self.include_remote:
            frames.extend(self._remote_frames())
        if not frames:
            raise RuntimeError("no Football-Data data is available locally or remotely")
        leagues, teams, matches = self._normalize(frames)
        if not matches:
            raise RuntimeError("Football-Data returned no valid canonical matches")
        return ProviderSnapshot(
            leagues=tuple(leagues),
            teams=tuple(teams),
            matches=tuple(matches),
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            provider=self.name,
        )
