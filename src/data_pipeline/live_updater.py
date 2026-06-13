"""
live_updater.py

Production-ready incremental data updater for the football prediction system.

Responsibilities:
- Fetch or simulate daily new match results (fetch_daily_matches)
- Append new matches safely to historical raw data (creates a daily CSV in data/raw/)
- Rebuild and validate a merged dataset saved to data/processed/merged_dataset.csv
- Provide helper to fetch latest matches for monitoring or quick checks

Design goals:
- Safe for automation (idempotent, does not corrupt historical files)
- Defensive validation and logging on every step
- Minimal external deps: pandas + standard library

Functions:
- fetch_daily_matches(source: Optional[str] = None) -> pandas.DataFrame
- update_dataset(new_matches: pd.DataFrame, data_dir: str = "data/raw", processed_path: str = "data/processed/merged_dataset.csv") -> pandas.DataFrame
- get_latest_matches(n: int = 10, processed_path: str = "data/processed/merged_dataset.csv", data_dir: str = "data/raw") -> pandas.DataFrame

Usage:
- Use in a daily cron/airflow job to fetch new matches and call update_dataset()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import tempfile
import shutil
import datetime

import pandas as pd

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Required canonical columns
_REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def _ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame has required columns, coerce types, drop invalid rows.

    Returns a cleaned DataFrame.
    Raises ValueError if required columns missing.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_REQUIRED_COLS)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Strip whitespace on team names
    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()

    # Parse Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)

    # Convert goals to numeric
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

    # Normalize FTR to single letter H/D/A
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
    df["FTR"] = df["FTR"].replace({"HOME": "H", "AWAY": "A", "DRAW": "D"})

    # Drop rows that are missing any critical values
    before = len(df)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    after = len(df)
    if before != after:
        logger.info("Dropped %d invalid rows during validation", before - after)

    # Cast goals to int
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Sort by Date (ascending)
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def fetch_daily_matches(source: Optional[str] = None) -> pd.DataFrame:
    """Fetch or simulate daily matches.

    Args:
        source: Optional path to CSV file. If provided and exists it will be read. If None, a small simulated
                set of matches for "today" will be returned (useful for testing).

    Returns: DataFrame with required columns.
    """
    if source:
        p = Path(source)
        if p.exists():
            logger.info("Loading daily matches from %s", source)
            df = pd.read_csv(p)
            df = _validate_and_clean(df)
            return df
        else:
            logger.warning("Provided source path does not exist: %s — falling back to simulation", source)

    # Simulate a small set of matches for today
    today = pd.to_datetime(datetime.date.today())
    simulated = pd.DataFrame(
        [
            {"Date": today, "HomeTeam": "Sample FC", "AwayTeam": "Example United", "FTHG": 2, "FTAG": 1, "FTR": "H"},
            {"Date": today, "HomeTeam": "Demo Town", "AwayTeam": "Trial City", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        ]
    )
    logger.info("Simulating %d daily matches for %s", len(simulated), str(today.date()))
    simulated = _validate_and_clean(simulated)
    return simulated


def _read_raw_csvs(data_dir: str = "data/raw") -> pd.DataFrame:
    """Read all CSV files in data_dir and concatenate into a single DataFrame."""
    p = Path(data_dir)
    if not p.exists():
        logger.warning("Raw data directory does not exist: %s", data_dir)
        return pd.DataFrame(columns=_REQUIRED_COLS)

    files = sorted(p.glob("*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = _validate_and_clean(df)
            frames.append(df)
            logger.info("Read %d rows from %s", len(df), f.name)
        except Exception as e:
            logger.exception("Failed to read/validate %s: %s", f, e)
            continue

    if not frames:
        return pd.DataFrame(columns=_REQUIRED_COLS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").reset_index(drop=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    logger.info("Combined raw CSVs -> %d total rows", len(combined))
    return combined


def update_dataset(new_matches: pd.DataFrame, data_dir: str = "data/raw", processed_path: str = "data/processed/merged_dataset.csv") -> pd.DataFrame:
    """Append new_matches into historical data safely and update merged dataset.

    Steps:
    - Validate new_matches
    - Save new matches as a dated CSV in data/raw/ (daily snapshot)
    - Read existing raw CSVs and concatenate
    - Remove duplicates by (Date, HomeTeam, AwayTeam)
    - Save merged dataset to processed_path using atomic write

    Returns the merged DataFrame
    """
    if new_matches is None or len(new_matches) == 0:
        logger.info("No new matches to update")
        return _read_raw_csvs(data_dir)

    # Validate incoming matches
    new_matches = _validate_and_clean(new_matches)
    if len(new_matches) == 0:
        logger.info("After validation no valid new matches remain")
        return _read_raw_csvs(data_dir)

    # Save daily snapshot into raw directory to preserve provenance
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    date_tag = pd.to_datetime(new_matches["Date"].min()).strftime("%Y-%m-%d")
    daily_file = raw_dir / f"daily_{date_tag}.csv"

    # Avoid overwriting existing daily file: if exists, append only new rows that aren't already present
    if daily_file.exists():
        try:
            existing_daily = pd.read_csv(daily_file)
            existing_daily = _validate_and_clean(existing_daily)
        except Exception:
            existing_daily = pd.DataFrame(columns=_REQUIRED_COLS)
        combined_daily = pd.concat([existing_daily, new_matches], ignore_index=True)
        combined_daily = combined_daily.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").reset_index(drop=True)
        _atomic_write_csv(combined_daily, daily_file)
        logger.info("Appended/merged %d rows into existing %s", len(new_matches), daily_file.name)
    else:
        _atomic_write_csv(new_matches, daily_file)
        logger.info("Saved daily new matches to %s", daily_file.name)

    # Rebuild merged dataset from all raw CSVs
    merged = _read_raw_csvs(data_dir)

    # Remove duplicates across merged
    before = len(merged)
    merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last").reset_index(drop=True)
    after = len(merged)
    if after != before:
        logger.info("Removed %d duplicates from merged dataset", before - after)

    # Ensure processed directory exists and save merged dataset atomically
    processed_p = Path(processed_path)
    _ensure_dirs(processed_p)
    _atomic_write_csv(merged, processed_p)
    logger.info("Wrote merged dataset to %s (%d rows)", processed_p, len(merged))

    return merged


def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    """Write DataFrame to CSV atomically to avoid partial writes.

    Writes to a temporary file in the same directory and then renames.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=str(target.parent), suffix='.tmp') as tf:
        tmp_path = Path(tf.name)
        df.to_csv(tmp_path, index=False)
    # Move into place
    shutil.move(str(tmp_path), str(target))


def get_latest_matches(n: int = 10, processed_path: str = "data/processed/merged_dataset.csv", data_dir: str = "data/raw") -> pd.DataFrame:
    """Return the latest n matches from processed dataset, falling back to raw CSVs if missing."""
    p = Path(processed_path)
    if p.exists():
        try:
            df = pd.read_csv(p)
            df = _validate_and_clean(df)
            return df.sort_values('Date', ascending=False).head(n).reset_index(drop=True)
        except Exception as e:
            logger.exception("Failed to read processed dataset %s: %s — falling back to raw CSVs", processed_path, e)
    # fallback
    df = _read_raw_csvs(data_dir)
    return df.sort_values('Date', ascending=False).head(n).reset_index(drop=True)


if __name__ == "__main__":
    # Simple CLI demo: simulate daily fetch and update merged dataset
    logging.basicConfig(level=logging.INFO)
    logger.info("Running live_updater demo: fetching simulated daily matches and updating dataset")
    new = fetch_daily_matches()
    merged = update_dataset(new)
    latest = get_latest_matches(10)
    logger.info("Latest matches after update:\n%s", latest)
