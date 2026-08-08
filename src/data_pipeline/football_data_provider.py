from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord
from .data_downloader import LEAGUE_CONFIG, is_football_data_unavailable, mark_football_data_unavailable
from .providers import ProviderSnapshot

logger = logging.getLogger(__name__)
BASE_URL = "https://www.football-data.co.uk/mmz4281"
NEW_URL = "https://www.football-data.co.uk/new"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
REQUEST_TIMEOUT = 30
SOURCE_COLUMNS = {
    "_FovraSourceType", "_FovraSourceKey", "_FovraSourceUrl",
    "_FovraSeasonStart", "_FovraRawDate", "_FovraRawTime",
}


@dataclass(frozen=True)
class SourceFrame:
    frame: pd.DataFrame
    source_type: str
    source_key: str
    source_url: str | None
    season_start: int | None


class FootballDataProvider:
    name = "football-data.co.uk"

    def __init__(self, raw_dir: str | Path = "data/raw", include_remote: bool = True):
        self.raw_dir = Path(raw_dir)
        self.include_remote = include_remote

    @staticmethod
    def _season_code(year: int) -> str:
        return f"{year % 100:02d}{(year + 1) % 100:02d}"

    @staticmethod
    def _season_from_date(kickoff: datetime) -> str:
        start = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
        return f"{start}-{start + 1}"

    @staticmethod
    def _team_key(name: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
        if not value:
            raise ValueError("empty team name")
        return value

    @staticmethod
    def _parse_datetime(row: pd.Series) -> tuple[datetime, str, str]:
        raw_date = row.get("_FovraRawDate", row.get("Date"))
        raw_time = row.get("_FovraRawTime", row.get("Time"))
        if pd.isna(raw_date):
            raise ValueError("match has no date")
        date_text = str(raw_date).strip()
        time_text = "" if pd.isna(raw_time) else str(raw_time).strip()
        value = f"{date_text} {time_text}".strip()
        formats = ("%d/%m/%Y %H:%M", "%d/%m/%y %H:%M", "%d/%m/%Y", "%d/%m/%y")
        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt).replace(tzinfo=ZoneInfo("Europe/London"))
                return parsed.astimezone(timezone.utc), date_text, time_text
            except ValueError:
                continue
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"unsupported Football-Data date format: {value}")
        if isinstance(parsed, pd.Timestamp):
            parsed = parsed.to_pydatetime()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("Europe/London"))
        return parsed.astimezone(timezone.utc), date_text, time_text

    @staticmethod
    def _season_bounds(season_start: int) -> tuple[datetime, datetime]:
        # Broad bounds intentionally cover legitimate early/late fixtures while
        # still catching a date accidentally assigned to a different season.
        return (
            datetime(season_start, 7, 1, tzinfo=timezone.utc),
            datetime(season_start + 1, 7, 31, 23, 59, 59, tzinfo=timezone.utc),
        )

    @staticmethod
    def _has_result(row: pd.Series) -> bool:
        fthg = pd.to_numeric(pd.Series([row.get("FTHG")]), errors="coerce").iloc[0]
        ftag = pd.to_numeric(pd.Series([row.get("FTAG")]), errors="coerce").iloc[0]
        ftr = str(row.get("FTR", "")).strip().upper()
        return pd.notna(fthg) and pd.notna(ftag) and ftr in {"H", "D", "A"}

    def _get(self, url: str):
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                return None, 404
            r.raise_for_status()
            return r.content, r.status_code
        except Exception as exc:
            logger.warning("Football-Data fetch failed: %s (%s)", url, exc)
            return None, None

    def _local_frames(self) -> list[SourceFrame]:
        frames: list[SourceFrame] = []
        code_to_key = {v["code"]: k for k, v in LEAGUE_CONFIG.items()}
        if not self.raw_dir.exists():
            return frames
        for path in sorted(self.raw_dir.glob("*.csv")):
            stem = path.stem
            match = re.fullmatch(r"(.+)_([0-9]{4})", stem)
            if match:
                code, year_text = match.groups()
                season_start = int(year_text)
                source_type = "season_results"
                source_key = f"{code}:{season_start}"
            elif stem.endswith("_new"):
                code = stem[:-4]
                season_start = None
                source_type = "combined_results"
                source_key = f"{code}:new"
            else:
                logger.info("Ignoring unclassified raw CSV: %s", path)
                continue
            league_key = code_to_key.get(code, code if code in LEAGUE_CONFIG else None)
            if not league_key:
                logger.info("Ignoring raw CSV for unconfigured league: %s", path)
                continue
            try:
                df = pd.read_csv(path)
                frames.append(SourceFrame(df, source_type, source_key, None, season_start))
            except Exception as exc:
                logger.warning("Skipping %s: %s", path, exc)
        return frames

    def _remote_frames(self) -> list[SourceFrame]:
        frames: list[SourceFrame] = []
        now = datetime.now(timezone.utc)
        season_year = now.year if now.month >= 7 else now.year - 1
        for league_key, info in LEAGUE_CONFIG.items():
            if is_football_data_unavailable(league_key, season_year):
                continue
            if info.get("source_type") == "single":
                url = f"{NEW_URL}/{info['code']}.csv"
                source_type, source_key, season_start = "combined_results", f"{league_key}:new", None
            else:
                url = f"{BASE_URL}/{self._season_code(season_year)}/{info['code']}.csv"
                source_type, source_key, season_start = "season_results", f"{league_key}:{season_year}", season_year
            data, status = self._get(url)
            if status == 404:
                mark_football_data_unavailable(league_key, url, status, season_year)
                continue
            if not data:
                continue
            try:
                frames.append(SourceFrame(pd.read_csv(io.BytesIO(data)), source_type, source_key, url, season_start))
            except Exception as exc:
                logger.warning("Could not parse %s: %s", url, exc)
        data, _ = self._get(FIXTURES_URL)
        if data:
            try:
                frames.append(SourceFrame(pd.read_csv(io.BytesIO(data)), "fixture_feed", "fixtures.csv", FIXTURES_URL, None))
            except Exception as exc:
                logger.warning("Could not parse fixture feed: %s", exc)
        return frames

    def _normalize(self, frames: list[SourceFrame]):
        leagues: dict[str, LeagueRecord] = {}
        teams: dict[tuple[str, str], TeamRecord] = {}
        matches: dict[str, MatchRecord] = {}
        code_to_key = {v["code"]: k for k, v in LEAGUE_CONFIG.items()}
        aliases = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR"}
        now = datetime.now(timezone.utc)

        for source in frames:
            df = source.frame.copy()
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            rename = {a: b for a, b in aliases.items() if b not in df.columns and a in df.columns}
            if rename:
                df = df.rename(columns=rename)
            for idx, row in df.iterrows():
                league_value = row.get("League", row.get("Div"))
                code = "" if pd.isna(league_value) else str(league_value).strip()
                lk = code if code in LEAGUE_CONFIG else code_to_key.get(code)
                if not lk:
                    continue
                home = str(row.get("HomeTeam", "")).strip()
                away = str(row.get("AwayTeam", "")).strip()
                if not home or not away or home == "nan" or away == "nan" or home == away:
                    continue
                try:
                    kickoff, raw_date, raw_time = self._parse_datetime(row)
                except ValueError:
                    continue

                if source.source_type == "season_results" and source.season_start is not None:
                    low, high = self._season_bounds(source.season_start)
                    if kickoff < low or kickoff > high:
                        logger.error(
                            "Rejected source/season date conflict: source=%s season=%s raw_date=%s kickoff=%s",
                            source.source_key, source.season_start, raw_date, kickoff.isoformat(),
                        )
                        continue
                    season = f"{source.season_start}-{source.season_start + 1}"
                else:
                    season = self._season_from_date(kickoff)

                has_result = self._has_result(row)
                if has_result and kickoff > now:
                    logger.error(
                        "Rejected future finished result: source=%s raw_date=%s kickoff=%s home=%s away=%s",
                        source.source_key, raw_date, kickoff.isoformat(), home, away,
                    )
                    continue
                status = "finished" if has_result else "scheduled"
                info = LEAGUE_CONFIG[lk]
                leagues[lk] = LeagueRecord(lk, info["name"])
                hk, ak = self._team_key(home), self._team_key(away)
                teams[(lk, hk)] = TeamRecord(hk, home, lk)
                teams[(lk, ak)] = TeamRecord(ak, away, lk)
                fthg = pd.to_numeric(pd.Series([row.get("FTHG")]), errors="coerce").iloc[0]
                ftag = pd.to_numeric(pd.Series([row.get("FTAG")]), errors="coerce").iloc[0]
                source_id_value = row.get("MatchID")
                source_id = str(source_id_value) if pd.notna(source_id_value) else None
                m = MatchRecord(
                    self.name, lk, season, kickoff.isoformat(timespec="seconds"), hk, ak,
                    status, int(fthg) if pd.notna(fthg) else None,
                    int(ftag) if pd.notna(ftag) else None, source_id,
                )
                matches[m.match_key] = m
        return list(leagues.values()), list(teams.values()), list(matches.values())

    def fetch(self) -> ProviderSnapshot:
        frames = self._local_frames()
        if self.include_remote:
            frames.extend(self._remote_frames())
        if not frames:
            raise RuntimeError("no Football-Data data is available locally or remotely")
        leagues, teams, matches = self._normalize(frames)
        if not matches:
            raise RuntimeError("Football-Data returned no valid canonical matches")
        return ProviderSnapshot(tuple(leagues), tuple(teams), tuple(matches), datetime.now(timezone.utc).isoformat(timespec="seconds"), self.name)
