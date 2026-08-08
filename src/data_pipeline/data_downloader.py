"""Raw Football-Data source downloader.

This module has one job: download provider CSVs and preserve their raw Date/Time
values exactly as supplied. It never normalizes dates or decides match status.
Season-based and combined /new feeds are kept as distinct source resources.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None

from src.data_pipeline.league_config import league_config_map

logger = logging.getLogger(__name__)
LEAGUE_CONFIG: Dict[str, Dict] = league_config_map()
ALLOWED_LEAGUES = list(LEAGUE_CONFIG)
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
START_YEAR = 2010
RAW_FORMAT_VERSION = 2
REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
HTTP_TIMEOUT = 45
MAX_RETRIES = 4
RETRY_DELAY = 2.0
UNAVAILABLE_TTL_HOURS = 1
UNAVAILABLE_PATH = RAW_DIR / "football_data_unavailable.json"
SOURCE_STATE_PATH = RAW_DIR / "football_data_source_state.json"


class FootballDataUnavailableError(RuntimeError):
    pass


def current_season_start(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def _load_unavailable() -> Dict[str, dict]:
    if not UNAVAILABLE_PATH.exists():
        return {}
    try:
        return json.loads(UNAVAILABLE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resource_key(league_key: str, season_start: int | None = None) -> str:
    if LEAGUE_CONFIG[league_key].get("source_type") == "single":
        return f"{league_key}:new"
    if season_start is None:
        raise ValueError("season_start is required for season-based sources")
    return f"{league_key}:{season_start}"


def is_football_data_unavailable(league_key: str, season_start: int | None = None) -> bool:
    entry = _load_unavailable().get(_resource_key(league_key, season_start))
    if not entry:
        return False
    try:
        marked = datetime.fromisoformat(str(entry["marked_at"]).replace("Z", "+00:00"))
        return datetime.now(timezone.utc) - marked <= timedelta(hours=UNAVAILABLE_TTL_HOURS)
    except Exception:
        return False


def mark_football_data_unavailable(league_key: str, url: str | None = None, status_code: int = 404, season_start: int | None = None) -> None:
    data = _load_unavailable()
    data[_resource_key(league_key, season_start)] = {
        "FootballDataUnavailable": True,
        "status_code": status_code,
        "url": url,
        "marked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    UNAVAILABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNAVAILABLE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_source_state() -> Dict[str, dict]:
    if not SOURCE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SOURCE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_source_state(state: Dict[str, dict]) -> None:
    SOURCE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _source_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _season_code(season_start: int) -> str:
    return f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"


def _candidate_url(league_key: str, season_start: int | None = None) -> Optional[str]:
    info = LEAGUE_CONFIG.get(league_key)
    if not info or not info.get("code"):
        return None
    if info.get("source_type") == "single":
        return f"https://www.football-data.co.uk/new/{info['code']}.csv"
    if season_start is None:
        raise ValueError("season_start is required for season-by-season Football-Data leagues")
    return f"https://www.football-data.co.uk/mmz4281/{_season_code(season_start)}/{info['code']}.csv"


def _http_get(url: str) -> Tuple[Optional[bytes], Optional[int]]:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests is None:
                from urllib.request import Request, urlopen
                with urlopen(Request(url, headers={"User-Agent": "Fovra/1.0"}), timeout=HTTP_TIMEOUT) as resp:
                    status = getattr(resp, "status", 200)
                    return (resp.read(), status) if status == 200 else (None, status)
            resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Fovra/1.0"})
            if resp.status_code == 200:
                return resp.content, resp.status_code
            if resp.status_code == 404:
                return None, 404
            last = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as exc:
            last = exc
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    logger.warning("Failed to download %s: %s", url, last)
    return None, None


def _validate_dataframe(df: pd.DataFrame) -> bool:
    columns = {str(c).strip().lstrip("\ufeff") for c in df.columns}
    return REQUIRED_COLS.issubset(columns)


def _atomic_save_csv(df: pd.DataFrame, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
            temp = Path(tmp.name)
            df.to_csv(temp, index=False)
        shutil.move(str(temp), str(path))
        return True
    except Exception:
        logger.exception("Failed to save %s", path)
        return False


def download_season_data(league_key: str, season_start: int, *, force_refresh: bool = False, ignore_unavailable: bool = False) -> bool:
    if league_key not in ALLOWED_LEAGUES:
        return False
    info = LEAGUE_CONFIG[league_key]
    if info.get("source_type") == "single" and season_start != START_YEAR:
        return False
    if season_start > current_season_start():
        return False
    if not ignore_unavailable and is_football_data_unavailable(league_key, season_start):
        return False

    url = _candidate_url(league_key, season_start)
    if not url:
        return False
    out_path = RAW_DIR / (f"{league_key}_new.csv" if info.get("source_type") == "single" else f"{league_key}_{season_start}.csv")
    if out_path.exists() and not force_refresh:
        try:
            if not _validate_dataframe(pd.read_csv(out_path, nrows=5)):
                return False
        except Exception:
            return False
        state = _load_source_state()
        if state.get(_resource_key(league_key, season_start), {}).get("raw_format_version") == RAW_FORMAT_VERSION:
            return True

    data, status = _http_get(url)
    if status == 404:
        mark_football_data_unavailable(league_key, url, status, season_start)
        raise FootballDataUnavailableError(f"Football-Data returned 404 for {league_key}")
    if data is None:
        return False

    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(data))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", url, exc)
        return False
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if not _validate_dataframe(df):
        logger.warning("Downloaded CSV failed required-column validation: %s", url)
        return False

    # Critical rule: Date and Time are written exactly as received. No pandas
    # datetime conversion is permitted in the raw layer.
    if "League" not in df.columns:
        df["League"] = league_key
    if "Tier" not in df.columns:
        df["Tier"] = info["tier"]
    if "LeagueStrength" not in df.columns:
        df["LeagueStrength"] = info["strength"]

    if not _atomic_save_csv(df, out_path):
        return False
    state = _load_source_state()
    state[_resource_key(league_key, season_start)] = {
        "url": url,
        "sha256": _source_fingerprint(data),
        "row_count": len(df),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_format_version": RAW_FORMAT_VERSION,
    }
    _save_source_state(state)
    return True


def download_all_leagues(start_year: int = START_YEAR, end_year: Optional[int] = None) -> Dict[str, List[int]]:
    end_year = current_season_start() if end_year is None else end_year
    results: Dict[str, List[int]] = {league: [] for league in ALLOWED_LEAGUES}
    for league in ALLOWED_LEAGUES:
        years = [start_year] if LEAGUE_CONFIG[league].get("source_type") == "single" else range(start_year, end_year + 1)
        for year in years:
            try:
                if download_season_data(league, year, force_refresh=True, ignore_unavailable=True):
                    results[league].append(year)
            except FootballDataUnavailableError:
                logger.warning("Source unavailable: %s %s", league, year)
    return results


def repair_missing_data() -> Dict[str, List[int]]:
    results: Dict[str, List[int]] = {league: [] for league in ALLOWED_LEAGUES}
    current = current_season_start()
    for league in ALLOWED_LEAGUES:
        info = LEAGUE_CONFIG[league]
        years = [START_YEAR] if info.get("source_type") == "single" else [current]
        for year in years:
            try:
                if download_season_data(league, year, force_refresh=True, ignore_unavailable=True):
                    results[league].append(year)
            except FootballDataUnavailableError:
                logger.warning("Source unavailable: %s %s", league, year)
    return results
