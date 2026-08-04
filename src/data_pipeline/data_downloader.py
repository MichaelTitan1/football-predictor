"""
data_downloader.py

Stable, production-safe football match CSV downloader.

Downloaded/generated datasets are local inputs and are intentionally excluded
from Git. The canonical production source of truth remains Neon.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import requests
except Exception:
    requests = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

from src.data_pipeline.league_config import league_config_map

LEAGUE_CONFIG: Dict[str, Dict] = league_config_map()

ALLOWED_LEAGUES = list(LEAGUE_CONFIG.keys())
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
START_YEAR = 2010
def current_season_start(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1

CURRENT_YEAR = current_season_start()
REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2.0
UNAVAILABLE_PATH = RAW_DIR / "football_data_unavailable.json"
SOURCE_STATE_PATH = RAW_DIR / "football_data_source_state.json"


class FootballDataUnavailableError(RuntimeError):
    """Raised when football-data.co.uk confirms a league code is unavailable."""


def _load_unavailable() -> Dict[str, dict]:
    if not UNAVAILABLE_PATH.exists():
        return {}
    try:
        return json.loads(UNAVAILABLE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", UNAVAILABLE_PATH, exc)
        return {}


def _resource_key(league_key: str, season_start: int | None = None) -> str:
    info = LEAGUE_CONFIG.get(league_key, {})
    if info.get("source_type") == "single":
        return f"{league_key}:new"
    return f"{league_key}:{season_start}" if season_start is not None else league_key

def is_football_data_unavailable(league_key: str, season_start: int | None = None) -> bool:
    data = _load_unavailable()
    return _resource_key(league_key, season_start) in data or league_key in data


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
    logger.info("Football-Data unavailable for %s; future runs will skip it immediately", league_key)



def _load_source_state() -> Dict[str, dict]:
    if not SOURCE_STATE_PATH.exists():
        return {}
    try:
        return json.loads(SOURCE_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read %s: %s", SOURCE_STATE_PATH, exc)
        return {}


def _save_source_state(state: Dict[str, dict]) -> None:
    SOURCE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _source_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_changed(resource_key: str, data: bytes) -> bool:
    state = _load_source_state()
    return state.get(resource_key, {}).get("sha256") != _source_fingerprint(data)


def _record_source_state(resource_key: str, url: str, data: bytes, row_count: int) -> None:
    state = _load_source_state()
    state[resource_key] = {
        "url": url,
        "sha256": _source_fingerprint(data),
        "row_count": row_count,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_source_state(state)


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
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests is None:
                from urllib.request import urlopen
                with urlopen(url, timeout=HTTP_TIMEOUT) as resp:
                    status = getattr(resp, "status", 200)
                    if status == 200:
                        return resp.read(), status
                    if status == 404:
                        return None, status
                    last_exc = Exception(f"HTTP {status}")
            else:
                resp = requests.get(url, timeout=HTTP_TIMEOUT)
                if resp.status_code == 200:
                    return resp.content, resp.status_code
                if resp.status_code == 404:
                    return None, resp.status_code
                last_exc = Exception(f"HTTP {resp.status_code}")
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404:
                return None, status
            last_exc = exc
            logger.debug("Attempt %d failed for %s: %s", attempt, url, exc)
        time.sleep(RETRY_DELAY)
    logger.info("Failed to download %s after %d attempts: %s", url, MAX_RETRIES, last_exc)
    return None, None


def _validate_dataframe(df: pd.DataFrame) -> bool:
    missing = REQUIRED_COLS - {str(c).strip() for c in df.columns}
    if missing:
        logger.warning("Validation failed; missing columns: %s", missing)
        return False
    return True


def _atomic_save_csv(df: pd.DataFrame, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=str(path.parent), suffix=".tmp") as tf:
            tmp = Path(tf.name)
            df.to_csv(tmp, index=False)
        shutil.move(str(tmp), str(path))
        return True
    except Exception:
        logger.exception("Failed to save CSV to %s", path)
        return False


def _file_is_valid(path: Path) -> bool:
    try:
        return _validate_dataframe(pd.read_csv(path, nrows=5))
    except Exception as exc:
        logger.warning("Existing file %s failed quick-parse: %s", path, exc)
        return False


def download_season_data(league_key: str, season_start: int, *, force_refresh: bool = False) -> bool:
    if league_key not in ALLOWED_LEAGUES:
        logger.info("League %s is not in allowed list; skipping.", league_key)
        return False
    if is_football_data_unavailable(league_key, season_start):
        logger.info("Football-Data resource unavailable for %s %s; skipping without HTTP requests.", league_key, season_start)
        return False
    if LEAGUE_CONFIG[league_key].get("source_type") == "single" and season_start != START_YEAR:
        return False
    if season_start > current_season_start():
        logger.info("Season %s is in the future; skipping.", season_start)
        return False
    url = _candidate_url(league_key, season_start)
    if not url:
        logger.info("No URL candidate for league %s; skipping.", league_key)
        return False

    source_type = LEAGUE_CONFIG[league_key].get("source_type")
    out_path = RAW_DIR / (f"{league_key}_new.csv" if source_type == "single" else f"{league_key}_{season_start}.csv")
    if out_path.exists() and _file_is_valid(out_path) and not force_refresh:
        logger.info("File already present and valid: %s", out_path)
        return True

    logger.info("Downloading %s season %s from %s", league_key, season_start, url)
    data, status_code = _http_get(url)
    if status_code == 404:
        mark_football_data_unavailable(league_key, url, status_code, season_start)
        raise FootballDataUnavailableError(f"Football-Data returned 404 for {league_key}")
    if data is None:
        return False
    resource_key = _resource_key(league_key, season_start)
    changed = _source_changed(resource_key, data)
    if out_path.exists() and _file_is_valid(out_path) and not changed:
        _record_source_state(resource_key, url, data, int(pd.read_csv(out_path, usecols=[0]).shape[0]))
        logger.info("Football-Data source unchanged: %s", url)
        return False
    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(data))
    except Exception as exc:
        logger.warning("Downloaded bytes could not be parsed as CSV for %s %s: %s", league_key, season_start, exc)
        return False

    df.columns = [str(c).strip() for c in df.columns]
    if not _validate_dataframe(df):
        logger.warning("Downloaded CSV for %s %s failed validation; not saved.", league_key, season_start)
        return False
    if "League" not in df.columns:
        df["League"] = league_key
    if "Tier" not in df.columns:
        df["Tier"] = LEAGUE_CONFIG[league_key]["tier"]
    if "LeagueStrength" not in df.columns:
        df["LeagueStrength"] = LEAGUE_CONFIG[league_key]["strength"]
    saved = _atomic_save_csv(df, out_path)
    if saved:
        _record_source_state(resource_key, url, data, len(df))
    return saved


def download_all_leagues(start_year: int = START_YEAR, end_year: Optional[int] = None) -> Dict[str, List[int]]:
    if end_year is None:
        end_year = current_season_start()
    started_at = time.monotonic()
    total_seasons = sum(1 if LEAGUE_CONFIG[l].get("source_type") == "single" else (end_year - start_year + 1) for l in ALLOWED_LEAGUES)
    completed = downloaded = skipped = unavailable = 0
    results: Dict[str, List[int]] = {}
    for league in ALLOWED_LEAGUES:
        results[league] = []
        years = [start_year] if LEAGUE_CONFIG[league].get("source_type") == "single" else range(start_year, end_year + 1)
        for year in years:
            completed += 1
            remaining = max(total_seasons - completed, 0)
            elapsed = time.monotonic() - started_at
            eta_seconds = (elapsed / completed * remaining) if completed else 0
            logger.info(
                "Current batch: %s %s | Downloaded=%s Skipped=%s Unavailable=%s Uploaded=%s Remaining=%s Estimated completion=%.1f minutes",
                league,
                year,
                downloaded,
                skipped,
                unavailable,
                downloaded,
                remaining,
                eta_seconds / 60,
            )
            try:
                if download_season_data(league, year):
                    results[league].append(year)
                    downloaded += 1
                else:
                    skipped += 1
            except FootballDataUnavailableError:
                unavailable += 1
                skipped += 0 if LEAGUE_CONFIG[league].get("source_type") == "single" else end_year - year
                logger.info("Stopping Football-Data requests for unavailable league %s after %s", league, year)
                break
            except Exception:
                skipped += 1
                logger.exception("Unexpected error downloading %s %s", league, year)
    logger.info(
        "Downloaded=%s Skipped=%s Unavailable=%s Uploaded=%s Remaining=0 Estimated completion=0.0 minutes",
        downloaded,
        skipped,
        unavailable,
        downloaded,
    )
    return results


def update_latest_season() -> Dict[str, List[int]]:
    """Refresh changed Football-Data sources and discover newly available seasons.

    Single-file leagues are checked against their fixed new/{code}.csv source.
    Season-based leagues refresh the latest known season and probe the next
    automatically detected season without reprocessing every historical file.
    """
    results: Dict[str, List[int]] = {}
    for league in ALLOWED_LEAGUES:
        existing_years = []
        newly_downloaded: List[int] = []
        if LEAGUE_CONFIG[league].get("source_type") == "single":
            if download_season_data(league, START_YEAR, force_refresh=True):
                newly_downloaded.append(START_YEAR)
            results[league] = newly_downloaded
            continue
        for f in RAW_DIR.glob(f"{league}_*.csv"):
            try:
                existing_years.append(int(f.stem.split("_")[-1]))
            except ValueError:
                continue
        if not existing_years:
            logger.info("No local baseline for %s; run --bootstrap during first setup before daily season checks.", league)
            results[league] = newly_downloaded
            continue
        latest = max(existing_years)
        if download_season_data(league, latest, force_refresh=True):
            newly_downloaded.append(latest)
        next_season = latest + 1
        if next_season <= current_season_start() and download_season_data(league, next_season):
            newly_downloaded.append(next_season)
        results[league] = newly_downloaded
    return results


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Manage football-data.co.uk historical CSV downloads")
    parser.add_argument("--bootstrap", action="store_true", help="Initial setup only: download all configured leagues from 2010 to the current season")
    args = parser.parse_args()

    if args.bootstrap:
        end_year = current_season_start()
        logger.info("Initial historical bootstrap: %d -> %d", START_YEAR, end_year)
        result = download_all_leagues(START_YEAR, end_year)
    else:
        logger.info("Checking for missing/new football-data.co.uk seasons only")
        result = update_latest_season()
    print(json.dumps({"downloaded_files": sum(len(v) for v in result.values()), "result": result}, indent=2))
