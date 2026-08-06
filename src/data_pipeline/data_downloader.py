"""
data_downloader.py

Stable, production-safe football match CSV downloader.

Football-Data has two layouts in this project:
- the 22 main divisions use one CSV per season;
- the 16 extra divisions use one current/combined CSV per league.

The downloader keeps a small local source-state cache so unchanged files are
not rewritten, but a transient 404 never permanently disables a source.
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
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
HTTP_TIMEOUT = 45
MAX_RETRIES = 4
RETRY_DELAY = 2.0
UNAVAILABLE_TTL_HOURS = 1
UNAVAILABLE_PATH = RAW_DIR / "football_data_unavailable.json"
SOURCE_STATE_PATH = RAW_DIR / "football_data_source_state.json"


class FootballDataUnavailableError(RuntimeError):
    """Raised when Football-Data confirms a resource is unavailable."""


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
    """Return true only for a very recent 404 marker.

    A previous 404 must never permanently suppress a league. Football-Data
    publishes the 16 extra-league files independently and can change their
    availability between refreshes.
    """
    entry = _load_unavailable().get(_resource_key(league_key, season_start))
    if not entry:
        return False
    try:
        marked_at = datetime.fromisoformat(str(entry.get("marked_at")).replace("Z", "+00:00"))
        if marked_at.tzinfo is None:
            marked_at = marked_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - marked_at > timedelta(hours=UNAVAILABLE_TTL_HOURS):
            return False
    except Exception:
        return False
    return True


def mark_football_data_unavailable(
    league_key: str,
    url: str | None = None,
    status_code: int = 404,
    season_start: int | None = None,
) -> None:
    data = _load_unavailable()
    data[_resource_key(league_key, season_start)] = {
        "FootballDataUnavailable": True,
        "status_code": status_code,
        "url": url,
        "marked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    UNAVAILABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNAVAILABLE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Football-Data unavailable for %s; retrying automatically after the short backoff window", league_key)


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
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if requests is None:
                from urllib.request import Request, urlopen
                req = Request(url, headers={"User-Agent": "Fovra/1.0"})
                with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    status = getattr(resp, "status", 200)
                    if status == 200:
                        return resp.read(), status
                    if status == 404:
                        return None, status
                    last_exc = Exception(f"HTTP {status}")
            else:
                resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Fovra/1.0"})
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
            logger.warning("Football-Data request attempt %d/%d failed for %s: %s", attempt, MAX_RETRIES, url, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
    logger.warning("Failed to download %s after %d attempts: %s", url, MAX_RETRIES, last_exc)
    return None, None


def _validate_dataframe(df: pd.DataFrame) -> bool:
    missing = REQUIRED_COLS - {str(c).strip().lstrip("\ufeff") for c in df.columns}
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


def download_season_data(
    league_key: str,
    season_start: int,
    *,
    force_refresh: bool = False,
    ignore_unavailable: bool = False,
) -> bool:
    if league_key not in ALLOWED_LEAGUES:
        logger.info("League %s is not in allowed list; skipping.", league_key)
        return False
    if not ignore_unavailable and is_football_data_unavailable(league_key, season_start):
        logger.info("Recent Football-Data 404 marker for %s; skipping briefly and allowing automatic retry later", league_key)
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

    logger.info("Downloading %s %s from %s", league_key, season_start, url)
    data, status_code = _http_get(url)
    if status_code == 404:
        mark_football_data_unavailable(league_key, url, status_code, season_start)
        raise FootballDataUnavailableError(f"Football-Data returned 404 for {league_key}")
    if data is None:
        return False

    resource_key = _resource_key(league_key, season_start)
    changed = _source_changed(resource_key, data)
    if out_path.exists() and _file_is_valid(out_path) and not changed:
        try:
            row_count = int(pd.read_csv(out_path, usecols=[0]).shape[0])
        except Exception:
            row_count = 0
        _record_source_state(resource_key, url, data, row_count)
        logger.info("Football-Data source unchanged: %s", url)
        return False

    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(data))
    except Exception as exc:
        logger.warning("Downloaded bytes could not be parsed as CSV for %s %s: %s", league_key, season_start, exc)
        return False

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
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
        logger.info("Saved %s: %d rows", out_path, len(df))
    return saved


def download_all_leagues(start_year: int = START_YEAR, end_year: Optional[int] = None) -> Dict[str, List[int]]:
    if end_year is None:
        end_year = current_season_start()
    started_at = time.monotonic()
    total_resources = sum(
        1 if LEAGUE_CONFIG[l].get("source_type") == "single" else (end_year - start_year + 1)
        for l in ALLOWED_LEAGUES
    )
    completed = downloaded = skipped = unavailable = 0
    results: Dict[str, List[int]] = {league: [] for league in ALLOWED_LEAGUES}

    for league in ALLOWED_LEAGUES:
        years = [start_year] if LEAGUE_CONFIG[league].get("source_type") == "single" else range(start_year, end_year + 1)
        for year in years:
            completed += 1
            remaining = max(total_resources - completed, 0)
            elapsed = time.monotonic() - started_at
            eta_seconds = (elapsed / completed * remaining) if completed else 0
            logger.info(
                "Current batch: %s %s | Downloaded=%s Skipped=%s Unavailable=%s Remaining=%s ETA=%.1f minutes",
                league, year, downloaded, skipped, unavailable, remaining, eta_seconds / 60,
            )
            try:
                if download_season_data(league, year, ignore_unavailable=True):
                    results[league].append(year)
                    downloaded += 1
                else:
                    skipped += 1
            except FootballDataUnavailableError:
                unavailable += 1
                skipped += 1
                logger.warning("Source %s is currently unavailable; continuing with the other configured leagues", league)
            except Exception:
                skipped += 1
                logger.exception("Unexpected error downloading %s %s", league, year)

    logger.info("Downloaded=%s Skipped=%s Unavailable=%s Remaining=0", downloaded, skipped, unavailable)
    return results


def repair_missing_data() -> Dict[str, List[int]]:
    """Ensure every configured league has a usable local Football-Data source.

    This is intentionally different from --bootstrap. It repairs an existing
    cache without downloading every historical season again. It also ignores
    old 404 markers so a previously unavailable extra league is retried.
    """
    results: Dict[str, List[int]] = {league: [] for league in ALLOWED_LEAGUES}
    current = current_season_start()

    for league in ALLOWED_LEAGUES:
        info = LEAGUE_CONFIG[league]
        if info.get("source_type") == "single":
            path = RAW_DIR / f"{league}_new.csv"
            if path.exists() and _file_is_valid(path):
                results[league].append(START_YEAR)
                continue
            try:
                if download_season_data(league, START_YEAR, ignore_unavailable=True):
                    results[league].append(START_YEAR)
            except FootballDataUnavailableError:
                logger.warning("Extra league %s is unavailable right now; it will be retried on the next scheduled run", league)
            continue

        existing_years: List[int] = []
        for path in RAW_DIR.glob(f"{league}_*.csv"):
            try:
                existing_years.append(int(path.stem.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                continue

        if not existing_years:
            try:
                if download_season_data(league, current, ignore_unavailable=True):
                    results[league].append(current)
            except FootballDataUnavailableError:
                logger.warning("No local baseline and current source unavailable for %s", league)
            continue

        latest = max(existing_years)
        try:
            if download_season_data(league, latest, force_refresh=True, ignore_unavailable=True):
                results[league].append(latest)
        except FootballDataUnavailableError:
            logger.warning("Current source for %s is unavailable; keeping the existing local season data", league)

        if latest < current:
            try:
                if download_season_data(league, latest + 1, ignore_unavailable=True):
                    results[league].append(latest + 1)
            except FootballDataUnavailableError:
                logger.warning("New season source for %s is not published yet", league)

    present = 0
    for league in ALLOWED_LEAGUES:
        info = LEAGUE_CONFIG[league]
        if info.get("source_type") == "single":
            present += int(_file_is_valid(RAW_DIR / f"{league}_new.csv"))
        else:
            present += int(any(_file_is_valid(p) for p in RAW_DIR.glob(f"{league}_*.csv")))
    logger.info("Football-Data local coverage: %d/%d configured leagues have usable files", present, len(ALLOWED_LEAGUES))
    return results


def update_latest_season() -> Dict[str, List[int]]:
    """Refresh changed sources and discover newly available seasons."""
    return repair_missing_data()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Manage football-data.co.uk historical CSV downloads")
    parser.add_argument("--bootstrap", action="store_true", help="Initial setup: download all configured leagues from 2010 to current season")
    parser.add_argument("--repair", action="store_true", help="Repair missing league files without re-downloading all history")
    args = parser.parse_args()

    if args.bootstrap:
        end_year = current_season_start()
        logger.info("Initial historical bootstrap: %d -> %d", START_YEAR, end_year)
        result = download_all_leagues(START_YEAR, end_year)
    else:
        logger.info("Checking for missing/new Football-Data sources only")
        result = repair_missing_data()

    coverage = sum(bool(v) for v in result.values())
    print(json.dumps({"downloaded_files": sum(len(v) for v in result.values()), "leagues_with_activity": coverage, "result": result}, indent=2))
