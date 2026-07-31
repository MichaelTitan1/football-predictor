"""
data_downloader.py

Stable, production-safe football match CSV downloader.

Downloaded/generated datasets are local inputs and are intentionally excluded
from Git. The canonical production source of truth remains Supabase.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
CURRENT_YEAR = datetime.now().year
REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2.0
PERMANENT_UNAVAILABLE_STATUS = {404}
SKIPPED_LEAGUES: dict[str, str] = {}
ROWS_PROCESSED = 0


def _season_code(season_start: int) -> str:
    return f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"


def _candidate_url(league_key: str, season_start: int) -> Optional[str]:
    info = LEAGUE_CONFIG.get(league_key)
    if not info or not info.get("code"):
        return None
    return f"https://www.football-data.co.uk/mmz4281/{_season_code(season_start)}/{info['code']}.csv"


def _http_get(url: str) -> Optional[bytes]:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests is None:
                from urllib.error import HTTPError
                from urllib.request import urlopen
                try:
                    with urlopen(url, timeout=HTTP_TIMEOUT) as resp:
                        status = getattr(resp, "status", 200)
                        if status == 200:
                            return resp.read()
                        if status in PERMANENT_UNAVAILABLE_STATUS:
                            logger.info("provider unavailable: %s returned HTTP %s", url, status)
                            return None
                        last_exc = Exception(f"HTTP {status}")
                except HTTPError as exc:
                    if exc.code in PERMANENT_UNAVAILABLE_STATUS:
                        logger.info("provider unavailable: %s returned HTTP %s", url, exc.code)
                        return None
                    raise
            else:
                resp = requests.get(url, timeout=HTTP_TIMEOUT)
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code in PERMANENT_UNAVAILABLE_STATUS:
                    logger.info("provider unavailable: %s returned HTTP %s", url, resp.status_code)
                    return None
                last_exc = Exception(f"HTTP {resp.status_code}")
        except Exception as exc:
            last_exc = exc
            logger.debug("Attempt %d failed for %s: %s", attempt, url, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    logger.info("Failed to download %s after %d attempts: %s", url, MAX_RETRIES, last_exc)
    return None


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
    if season_start > CURRENT_YEAR:
        logger.info("Season %s is in the future; skipping.", season_start)
        return False
    url = _candidate_url(league_key, season_start)
    if not url:
        logger.info("No URL candidate for league %s; skipping.", league_key)
        return False

    out_path = RAW_DIR / f"{league_key}_{season_start}.csv"
    if out_path.exists() and _file_is_valid(out_path) and not force_refresh:
        logger.info("File already present and valid: %s", out_path)
        return True

    logger.info("Downloading %s season %s from %s", league_key, season_start, url)
    data = _http_get(url)
    if data is None:
        SKIPPED_LEAGUES[league_key] = "provider unavailable"
        logger.info("provider unavailable: skipping %s season %s", league_key, season_start)
        return False
    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(data))
    except Exception as exc:
        logger.warning("Downloaded bytes could not be parsed as CSV for %s %s: %s", league_key, season_start, exc)
        return False

    df.columns = [str(c).strip() for c in df.columns]
    global ROWS_PROCESSED
    ROWS_PROCESSED += len(df)
    if not _validate_dataframe(df):
        logger.warning("Downloaded CSV for %s %s failed validation; not saved.", league_key, season_start)
        return False
    if "League" not in df.columns:
        df["League"] = league_key
    if "Tier" not in df.columns:
        df["Tier"] = LEAGUE_CONFIG[league_key]["tier"]
    if "LeagueStrength" not in df.columns:
        df["LeagueStrength"] = LEAGUE_CONFIG[league_key]["strength"]
    return _atomic_save_csv(df, out_path)


def download_all_leagues(start_year: int = START_YEAR, end_year: Optional[int] = None) -> Dict[str, List[int]]:
    if end_year is None:
        end_year = CURRENT_YEAR
    results: Dict[str, List[int]] = {}
    for league in ALLOWED_LEAGUES:
        results[league] = []
        for year in range(start_year, end_year + 1):
            try:
                if download_season_data(league, year):
                    results[league].append(year)
            except Exception:
                logger.exception("Unexpected error downloading %s %s", league, year)
    return results


def update_latest_season() -> Dict[str, List[int]]:
    """Download only seasons that are not already present locally.

    This is the post-bootstrap path for permanent free deployments: existing
    football-data.co.uk CSV files are never redownloaded or refreshed here.
    """
    results: Dict[str, List[int]] = {}
    for league in ALLOWED_LEAGUES:
        existing_years = []
        for f in RAW_DIR.glob(f"{league}_*.csv"):
            try:
                existing_years.append(int(f.stem.split("_")[-1]))
            except ValueError:
                continue
        newly_downloaded: List[int] = []
        if not existing_years:
            logger.info("No local baseline for %s; run --bootstrap during first setup before daily season checks.", league)
            results[league] = newly_downloaded
            continue
        latest = max(existing_years)
        next_season = latest + 1
        if next_season <= CURRENT_YEAR and download_season_data(league, next_season):
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
        logger.info("Initial historical bootstrap: %d -> %d", START_YEAR, CURRENT_YEAR)
        result = download_all_leagues(START_YEAR, CURRENT_YEAR)
    else:
        logger.info("Checking for missing/new football-data.co.uk seasons only")
        result = update_latest_season()
    downloaded = {league: years for league, years in result.items() if years}
    remaining = [league for league in ALLOWED_LEAGUES if league not in downloaded and league not in SKIPPED_LEAGUES]
    print(json.dumps({
        "downloaded_files": sum(len(v) for v in result.values()),
        "downloaded_leagues": downloaded,
        "skipped_leagues": SKIPPED_LEAGUES,
        "rows_processed": ROWS_PROCESSED,
        "rows_uploaded": 0,
        "rows_remaining": 0,
        "estimated_completion": "download phase complete",
        "remaining_without_new_download": remaining,
        "result": result,
    }, indent=2))
