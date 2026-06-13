"""
data_downloader.py

Automated downloader for football match CSVs (football-data.co.uk style) used as the
official data engine for the football prediction system.

Responsibilities:
- Download historical and seasonal CSVs for multiple leagues
- Validate files for required schema
- Organize files under data/raw/ and avoid duplicates
- Provide automated update routines for continuous operation

Notes:
- Uses public, free sources (football-data.co.uk pattern). Does NOT rely on paid APIs.
- Designed for automation (cron / GitHub Actions). Robust to network failures and retries.
- Adds league metadata into each downloaded file (columns Tier and LeagueStrength) for downstream use.

Public API:
- download_season_data(league_key: str, season: int | str) -> bool
- download_all_leagues(start_year: int = 2010, end_year: Optional[int] = None) -> Dict
- update_latest_season() -> Dict

Configuration:
- LEAGUE_CONFIG: metadata for supported leagues
- LEAGUE_CODE_MAP: mapping to football-data.co.uk competition codes when available

"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Prefer requests but fall back to urllib if not installed
try:
    import requests
except Exception:  # pragma: no cover - runtime environment may or may not have requests
    requests = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# League configuration (user-facing keys -> metadata)
LEAGUE_CONFIG: Dict[str, Dict] = {
    "EPL": {"tier": 1, "strength": 1.0},
    "LaLiga": {"tier": 1, "strength": 0.98},
    "SerieA": {"tier": 1, "strength": 0.97},
    "Bundesliga": {"tier": 1, "strength": 0.96},
    "Ligue1": {"tier": 1, "strength": 0.94},

    "Brazil": {"tier": 2, "strength": 0.90},
    "Argentina": {"tier": 2, "strength": 0.88},
    "MLS": {"tier": 2, "strength": 0.85},

    "Friendlies": {"tier": 3, "strength": 0.50},
}

# Mapping from our LEAGUE_CONFIG keys to football-data.co.uk competition codes (where available)
# football-data.co.uk uses two-letter season code directories like 2122 and competition codes like E0, SP1, I1, D1, F1 etc.
LEAGUE_CODE_MAP: Dict[str, str] = {
    "EPL": "E0",
    "LaLiga": "SP1",
    "SerieA": "I1",
    "Bundesliga": "D1",
    "Ligue1": "F1",
    "Brazil": "BSA",  # Not standard on football-data; placeholder (may fail)
    "Argentina": "ARG1",
    "MLS": "MLS",  # often not available on football-data.uk
    # Add other mappings as available. Non-mapped leagues will be attempted but may not exist on football-data
}

# Where to store downloaded CSVs
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Required columns for validation
_REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Downloader configuration
HTTP_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds (exponential multiplier)

# Season range configuration
START_YEAR = 2010
CURRENT_YEAR = datetime.now().year


def _season_to_mmz_code(season_start: int) -> str:
    """Convert a season start year to the mmz4281 two-year code used on football-data.co.uk.

    Example: 2010 -> '1011', 2021 -> '2122'
    """
    a = int(season_start) % 100
    b = int(season_start + 1) % 100
    return f"{a:02d}{b:02d}"


def _season_to_label(season_start: int) -> str:
    """Human-friendly season label like '2010-2011' or '2021-22'"""
    return f"{int(season_start)}-{int(season_start + 1)}"


def _download_url_candidates(league_key: str, season_start: int) -> List[str]:
    """Generate a list of candidate URLs to try for downloading a season for a league.

    Primary source: football-data.co.uk mmz4281 pattern
    e.g. https://www.football-data.co.uk/mmz4281/2122/E0.csv
    
    We generate a small list of candidate patterns to improve hit-rate for different leagues.
    """
    candidates: List[str] = []
    season_code = _season_to_mmz_code(season_start)

    # If we have a known football-data competition code, use it
    comp_code = LEAGUE_CODE_MAP.get(league_key)
    if comp_code:
        candidates.append(f"https://www.football-data.co.uk/mmz4281/{season_code}/{comp_code}.csv")
        # older pattern sometimes used lower-case extension or different path; include alternate
        candidates.append(f"https://www.football-data.co.uk/mmz4281/{season_code}/{comp_code}.CSV")

    # Also try a year-only style (some datasets name files by year)
    candidates.append(f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_key}.csv")

    # Also try root path with season folder
    candidates.append(f"https://www.football-data.co.uk/{season_code}/{league_key}.csv")

    # Add a GitHub raw openfootball-style fallback if known patterns exist (lightweight best-effort)
    # Example pattern: https://raw.githubusercontent.com/openfootball/{league-repo}/master/{season}.csv
    # We will not hardcode many repos; this is optional and best-effort

    return candidates


def _http_get(url: str, timeout: int = HTTP_TIMEOUT) -> Optional[bytes]:
    """Download bytes from a URL with retries. Returns bytes or None on permanent failure."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests is None:
                # fallback to urllib
                from urllib.request import urlopen

                with urlopen(url, timeout=timeout) as resp:
                    return resp.read()
            else:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 200:
                    return resp.content
                else:
                    logger.debug("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt)
                    last_exc = Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            logger.debug("Download attempt %d failed for %s: %s", attempt, url, e)
            last_exc = e
        # Backoff
        time.sleep(RETRY_BACKOFF ** attempt)
    logger.warning("Failed to download %s after %d attempts: last error: %s", url, MAX_RETRIES, last_exc)
    return None


def _validate_csv_bytes(csv_bytes: bytes) -> Tuple[bool, Optional[pd.DataFrame], str]:
    """Validate CSV bytes: read into pandas, normalize common column names, and check required cols.

    Returns (ok, dataframe or None, message)
    """
    try:
        # Read CSV into DataFrame using pandas
        from io import BytesIO

        df = pd.read_csv(BytesIO(csv_bytes))
    except Exception as e:
        logger.exception("Pandas failed to parse CSV bytes: %s", e)
        return False, None, f"parse_error: {e}"

    # Normalize columnnames (strip)
    df.columns = [str(c).strip() for c in df.columns]

    # Common alias normalization (football-data style columns usually match canonical names)
    # Attempt to map variants to canonical names
    mapping = {}
    for c in df.columns:
        cstrip = c.strip()
        if cstrip in ["HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG", "FTR"]:
            mapping[c] = cstrip
        else:
            # simple heuristics
            low = cstrip.lower()
            if low in ("home", "hometeam", "home_team"):
                mapping[c] = "HomeTeam"
            elif low in ("away", "awayteam", "away_team"):
                mapping[c] = "AwayTeam"
            elif low in ("fthg", "hg", "homegoals", "home_goals"):
                mapping[c] = "FTHG"
            elif low in ("ftag", "ag", "awaygoals", "away_goals"):
                mapping[c] = "FTAG"
            elif low in ("ftr", "res", "result"):
                mapping[c] = "FTR"
            elif low in ("date", "matchdate"):
                mapping[c] = "Date"
    if mapping:
        df = df.rename(columns=mapping)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, None, f"missing_columns: {missing}"

    # Optionally coerce types
    try:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", infer_datetime_format=True)
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Check some non-null count for Date/Home/Away
    if df["Date"].notna().sum() == 0 or df["HomeTeam"].notna().sum() == 0:
        return False, None, "empty_essential_columns"

    return True, df, "ok"


def _save_raw_csv(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV atomically and ensure metadata columns exist (Tier, LeagueStrength, SourceDownloadedAt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Add metadata placeholders if not present
    if "Tier" not in df.columns:
        df["Tier"] = None
    if "LeagueStrength" not in df.columns:
        df["LeagueStrength"] = None
    df["SourceDownloadedAt"] = pd.Timestamp.utcnow()

    # Atomic write
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=str(path.parent), suffix=".tmp") as tf:
        tmp = Path(tf.name)
        df.to_csv(tmp, index=False)
    shutil.move(str(tmp), str(path))


def _file_exists_and_valid(path: Path) -> bool:
    """Check if file exists and contains valid CSV with required columns."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        cols = [c.strip() for c in df.columns]
        for c in _REQUIRED_COLS:
            if c not in cols:
                logger.warning("Existing file %s is missing required column %s", path, c)
                return False
        return True
    except Exception as e:
        logger.warning("Existing file %s failed to parse: %s", path, e)
        return False


def download_season_data(league_key: str, season: int | str) -> bool:
    """Download CSV data for a given league and season.

    season may be an integer start year (e.g., 2010) or a string like '2010-2011' or '2010/2011'.

    Returns True if file downloaded or already present and valid; False on failure.
    """
    # Normalize league key
    if league_key not in LEAGUE_CONFIG:
        logger.error("League %s not in LEAGUE_CONFIG; skipping", league_key)
        return False

    # Normalize season start year
    if isinstance(season, str):
        # extract first 4-digit year
        import re

        m = re.search(r"(20\d{2}|19\d{2})", season)
        if not m:
            logger.error("Could not parse season string %s", season)
            return False
        season_start = int(m.group(0))
    else:
        season_start = int(season)

    season_label = _season_to_label(season_start)
    filename = f"{league_key}_{season_start}.csv"
    out_path = RAW_DIR / filename

    # Do not overwrite valid existing files
    if _file_exists_and_valid(out_path):
        logger.info("File already exists and valid: %s — skipping download", out_path)
        return True

    # Generate URLs to try
    candidates = _download_url_candidates(league_key, season_start)

    downloaded = False
    for url in candidates:
        logger.info("Attempting download for %s %s from %s", league_key, season_label, url)
        data = _http_get(url)
        if data is None:
            logger.debug("No data from %s", url)
            continue
        ok, df, msg = _validate_csv_bytes(data)
        if not ok:
            logger.warning("Downloaded file from %s failed validation: %s", url, msg)
            continue

        # Add league metadata columns
        meta = LEAGUE_CONFIG.get(league_key, {})
        df["League"] = league_key
        df["Tier"] = meta.get("tier")
        df["LeagueStrength"] = meta.get("strength")

        # Save atomically and don't overwrite existing unless invalid
        try:
            # If file exists but invalid, overwrite
            if out_path.exists() and not _file_exists_and_valid(out_path):
                logger.info("Overwriting previously invalid file %s", out_path)
                out_path.unlink()
            # Save CSV
            df.to_csv(out_path, index=False)
            logger.info("Saved %s rows to %s", len(df), out_path)
            downloaded = True
            break
        except Exception as e:
            logger.exception("Failed to save downloaded file to %s: %s", out_path, e)
            downloaded = False
            continue

    if not downloaded:
        logger.error("Failed to download any candidate for %s %s", league_key, season_label)
        return False

    # Final validation read-back
    if not _file_exists_and_valid(out_path):
        logger.error("Saved file %s failed post-save validation", out_path)
        return False

    return True


def download_all_leagues(start_year: int = START_YEAR, end_year: Optional[int] = None) -> Dict[str, List[int]]:
    """Download seasons for all leagues in LEAGUE_CONFIG between start_year and end_year (inclusive).

    Returns a dict mapping league_key -> list of seasons successfully downloaded/skipped.
    """
    if end_year is None:
        end_year = CURRENT_YEAR
    results: Dict[str, List[int]] = {}
    for league in LEAGUE_CONFIG.keys():
        results[league] = []
        for year in range(start_year, end_year + 1):
            try:
                ok = download_season_data(league, year)
                if ok:
                    results[league].append(year)
            except Exception as e:
                logger.exception("Failed downloading %s %d: %s", league, year, e)
                # continue to next year
                continue
    return results


def _parse_season_from_filename(filename: str) -> Optional[int]:
    """Extract season start year from filename pattern LEAGUE_YYYY.csv"""
    try:
        name = Path(filename).stem
        parts = name.split("_")
        if len(parts) >= 2:
            year = int(parts[-1])
            return year
    except Exception:
        return None
    return None


def update_latest_season() -> Dict[str, List[int]]:
    """Detect latest seasons already present and download any missing new seasons up to CURRENT_YEAR.

    For each league, scans data/raw for existing files named {LEAGUE}_{season}.csv and downloads seasons after
    the latest found up to CURRENT_YEAR.

    Returns dict mapping league -> list of seasons downloaded.
    """
    results: Dict[str, List[int]] = {}
    for league in LEAGUE_CONFIG.keys():
        existing_years: List[int] = []
        pattern = f"{league}_*.csv"
        for f in RAW_DIR.glob(pattern):
            s = _parse_season_from_filename(f.name)
            if s:
                existing_years.append(s)
        if existing_years:
            latest = max(existing_years)
        else:
            latest = START_YEAR - 1
        to_download = []
        for year in range(latest + 1, CURRENT_YEAR + 1):
            to_download.append(year)
        downloaded = []
        for y in to_download:
            try:
                ok = download_season_data(league, y)
                if ok:
                    downloaded.append(y)
            except Exception as e:
                logger.exception("Error updating %s season %d: %s", league, y, e)
                continue
        results[league] = downloaded
    return results


if __name__ == "__main__":
    # Demo: do an initial download from START_YEAR to CURRENT_YEAR for configured leagues
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting bulk download from %d to %d", START_YEAR, CURRENT_YEAR)
    res = download_all_leagues(START_YEAR, CURRENT_YEAR)
    logger.info("Download summary: %s", res)
