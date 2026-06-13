"""
data_downloader.py

Stable, production-safe football match CSV downloader.

This implementation is intentionally minimal and conservative:
- Only uses one stable public source pattern (football-data.co.uk mmz4281)
- Only attempts leagues for which we have a known competition code
- Skips and logs failures without raising
- Validates downloaded CSVs before saving

Public functions:
- download_season_data(league_key: str, season_start_year: int) -> bool
- download_all_leagues(start_year: int = 2010, end_year: Optional[int] = None) -> Dict[str, List[int]]
- update_latest_season() -> Dict[str, List[int]]

Files saved to: data/raw/{LEAGUE}_{SEASON}.csv
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil

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

# Configuration: allowed leagues and their football-data.co.uk competition codes
# We only include leagues we expect to be reliably present on football-data.co.uk.
LEAGUE_CONFIG: Dict[str, Dict] = {
    # Tier 1
    "EPL": {"name": "Premier League", "code": "E0", "tier": 1, "strength": 1.0},
    "LaLiga": {"name": "La Liga", "code": "SP1", "tier": 1, "strength": 0.98},
    "SerieA": {"name": "Serie A", "code": "I1", "tier": 1, "strength": 0.97},
    "Bundesliga": {"name": "Bundesliga", "code": "D1", "tier": 1, "strength": 0.96},
    "Ligue1": {"name": "Ligue 1", "code": "F1", "tier": 1, "strength": 0.94},
    # Tier 2 (only if available on football-data.co.uk)
    "PrimeiraLiga": {"name": "Primeira Liga", "code": "P1", "tier": 2, "strength": 0.86},
    "Eredivisie": {"name": "Eredivisie", "code": "N1", "tier": 2, "strength": 0.85},
    "BelgianPro": {"name": "Belgian Pro League", "code": "B1", "tier": 2, "strength": 0.84},
    "SuperLig": {"name": "Süper Lig", "code": "T1", "tier": 2, "strength": 0.80},
    "MLS": {"name": "MLS", "code": "M1", "tier": 2, "strength": 0.78},
}

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


def _season_code(season_start: int) -> str:
    a = season_start % 100
    b = (season_start + 1) % 100
    return f"{a:02d}{b:02d}"


def _candidate_url(league_key: str, season_start: int) -> Optional[str]:
    info = LEAGUE_CONFIG.get(league_key)
    if not info:
        return None
    comp = info.get("code")
    if not comp:
        return None
    season = _season_code(season_start)
    return f"https://www.football-data.co.uk/mmz4281/{season}/{comp}.csv"


def _http_get(url: str) -> Optional[bytes]:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests is None:
                from urllib.request import urlopen

                with urlopen(url, timeout=HTTP_TIMEOUT) as resp:
                    if getattr(resp, "status", 200) >= 200:
                        return resp.read()
                    last_exc = Exception(f"HTTP {getattr(resp, 'status', 'unknown')}")
            else:
                resp = requests.get(url, timeout=HTTP_TIMEOUT)
                if resp.status_code == 200:
                    return resp.content
                last_exc = Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            last_exc = e
            logger.debug("Attempt %d failed for %s: %s", attempt, url, e)
        time.sleep(RETRY_DELAY)
    logger.info("Failed to download %s after %d attempts: %s", url, MAX_RETRIES, last_exc)
    return None


def _validate_dataframe(df: pd.DataFrame) -> bool:
    cols = {c.strip() for c in df.columns}
    missing = REQUIRED_COLS - cols
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
    except Exception as e:
        logger.exception("Failed to save CSV to %s: %s", path, e)
        return False


def _file_is_valid(path: Path) -> bool:
    try:
        df = pd.read_csv(path, nrows=5)
        return _validate_dataframe(df)
    except Exception as e:
        logger.warning("Existing file %s failed quick-parse: %s", path, e)
        return False


def download_season_data(league_key: str, season_start: int) -> bool:
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

    if out_path.exists() and _file_is_valid(out_path):
        logger.info("File already present and valid: %s", out_path)
        return True

    logger.info("Downloading %s season %s from %s", league_key, season_start, url)
    data = _http_get(url)
    if data is None:
        logger.info("No data available for %s %s (URL tried: %s)", league_key, season_start, url)
        return False

    try:
        from io import BytesIO

        df = pd.read_csv(BytesIO(data))
    except Exception as e:
        logger.warning("Downloaded bytes could not be parsed as CSV for %s %s: %s", league_key, season_start, e)
        return False

    df.columns = [str(c).strip() for c in df.columns]

    if not _validate_dataframe(df):
        logger.warning("Downloaded CSV for %s %s failed validation; not saved.", league_key, season_start)
        return False

    if "League" not in df.columns:
        df["League"] = league_key
    if "Tier" not in df.columns:
        df["Tier"] = LEAGUE_CONFIG.get(league_key, {}).get("tier")
    if "LeagueStrength" not in df.columns:
        df["LeagueStrength"] = LEAGUE_CONFIG.get(league_key, {}).get("strength")

    saved = _atomic_save_csv(df, out_path)
    if saved:
        logger.info("Saved %s rows to %s", len(df), out_path)
        return True
    logger.warning("Failed to save file for %s %s", league_key, season_start)
    return False


def download_all_leagues(start_year: int = START_YEAR, end_year: Optional[int] = None) -> Dict[str, List[int]]:
    if end_year is None:
        end_year = CURRENT_YEAR
    results: Dict[str, List[int]] = {}
    for league in ALLOWED_LEAGUES:
        results[league] = []
        for year in range(start_year, end_year + 1):
            try:
                ok = download_season_data(league, year)
                if ok:
                    results[league].append(year)
            except Exception as e:
                logger.exception("Unexpected error downloading %s %s: %s", league, year, e)
                continue
    return results


def update_latest_season() -> Dict[str, List[int]]:
    results: Dict[str, List[int]] = {}
    for league in ALLOWED_LEAGUES:
        existing_years: List[int] = []
        pattern = f"{league}_*.csv"
        for f in RAW_DIR.glob(pattern):
            try:
                stem = f.stem
                parts = stem.split("_")
                if len(parts) >= 2:
                    y = int(parts[-1])
                    existing_years.append(y)
            except Exception:
                continue
        latest = max(existing_years) if existing_years else (START_YEAR - 1)
        to_download = [y for y in range(latest + 1, CURRENT_YEAR + 1) if y <= CURRENT_YEAR]
        newly_downloaded: List[int] = []
        for y in to_download:
            try:
                ok = download_season_data(league, y)
                if ok:
                    newly_downloaded.append(y)
            except Exception as e:
                logger.exception("Error updating %s season %s: %s", league, y, e)
                continue
        results[league] = newly_downloaded
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting conservative bulk download: %d -> %d", START_YEAR, CURRENT_YEAR)
    res = download_all_leagues(START_YEAR, CURRENT_YEAR)
    logger.info("Download complete. Summary: %s", res)
