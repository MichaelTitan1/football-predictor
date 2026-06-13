"""
data_loader.py

Production-ready data loader for historical football match CSV files.

Responsibilities:
- Load CSV files from a data directory (default: ./data/raw/)
- Standardize column names across inconsistent season files
- Ensure required columns are present (Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR)
- Clean data:
    - remove rows with missing critical values
    - parse Date into pandas.Timestamp
    - normalize team names (trim, collapse spaces, consistent casing)
    - drop exact duplicates
- Merge seasons into a single DataFrame sorted by Date
- Provide load_all_data() as the public entrypoint
- Log files loaded, rows processed per file, and final dataset size

Only uses standard libraries + pandas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import re

# Public API
__all__ = ["load_all_data"]

# Configure module-level logger (safe to call multiple times)
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration; in production the application may configure logging differently.
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# REQUIRED canonical column names
_REQUIRED_COLS: Tuple[str, ...] = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")


# Mapping of common alternate column names to canonical names.
# This list is not exhaustive but covers common variations across datasets.
_COLUMN_RENAMES: Dict[str, str] = {
    # Date variations
    "date": "Date",
    "match_date": "Date",
    "matchdate": "Date",
    "kickoff": "Date",
    "time": "Date",
    # Home team
    "home": "HomeTeam",
    "home_team": "HomeTeam",
    "hometeam": "HomeTeam",
    "team_home": "HomeTeam",
    # Away team
    "away": "AwayTeam",
    "away_team": "AwayTeam",
    "awayteam": "AwayTeam",
    "team_away": "AwayTeam",
    # Full time home goals
    "hg": "FTHG",
    "homegoals": "FTHG",
    "home_goals": "FTHG",
    "full_time_home_goals": "FTHG",
    # Full time away goals
    "ag": "FTAG",
    "awaygoals": "FTAG",
    "away_goals": "FTAG",
    "full_time_away_goals": "FTAG",
    # Full time result
    "result": "FTR",
    "res": "FTR",
    "full_time_result": "FTR",
    "ftresult": "FTR",
    "ftr": "FTR",
}


def _find_csv_files(data_dir: str = "data/raw", pattern: str = "*.csv") -> List[Path]:
    """Return a sorted list of Path objects pointing to CSV files under data_dir matching pattern."""
    path = Path(data_dir)
    if not path.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        return []
    files = sorted(Path(data_dir).glob(pattern))
    return files


def _normalize_column_name(col: str) -> str:
    """Return a normalized, simplified column name (lowercase, no spaces/punctuation) for mapping lookup."""
    c = col.strip()
    c = c.lstrip("\ufeff")
    c = c.lower()
    c = re.sub(r"[^0-9a-z]+", "_", c)
    c = c.strip("_")
    return c


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to canonical names based on heuristics and mappings.

    This function does a best-effort mapping using:
    - exact lower-cased name matches against _COLUMN_RENAMES keys
    - normalized keys via _normalize_column_name
    - preserves other columns unchanged
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = _normalize_column_name(col)
        if key in _COLUMN_RENAMES:
            rename_map[col] = _COLUMN_RENAMES[key]
        else:
            for canonical in _REQUIRED_COLS:
                if _normalize_column_name(canonical) == key:
                    rename_map[col] = canonical
                    break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _attempt_parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse a pandas Series of date-like strings into Timestamps.

    Strategy:
    - Try to parse with infer_datetime_format and dayfirst=False
    - If more than 20% are NaT, try dayfirst=True
    - Final result may still contain NaT for unparsable entries (they will be dropped later)
    """
    parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=False)
    nat_fraction = parsed.isna().mean()
    if nat_fraction > 0.2:
        logger.debug("High NaT fraction (%.2f) with dayfirst=False; retrying with dayfirst=True", nat_fraction)
        parsed_alt = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
        if parsed_alt.isna().mean() < nat_fraction:
            parsed = parsed_alt
    return parsed


def _normalize_team_name(series: pd.Series) -> pd.Series:
    """
    Normalize team names:
    - strip leading/trailing spaces
    - collapse multiple internal spaces to single space
    - convert to consistent casing (Title case)
    - preserve NaN
    """
    s = series.astype("string")
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.title()
    return s


def _validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check for presence of required columns. Return (is_valid, missing_list)."""
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


def _clean_dataframe(df: pd.DataFrame, source_label: Optional[str] = None) -> pd.DataFrame:
    """
    Perform cleaning steps on a single season DataFrame and return cleaned DataFrame.

    Steps:
    - Standardize column names
    - Validate required columns exist (will raise ValueError if missing)
    - Parse Date column to datetime (coercing unparsable to NaT)
    - Trim rows with missing critical values (Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR)
    - Normalize team names
    - Convert goal columns to integers (if possible)
    - Drop exact duplicates
    """
    if source_label is None:
        source_label = "<unknown>"

    df = _standardize_columns(df)

    ok, missing = _validate_required_columns(df)
    if not ok:
        raise ValueError(f"Missing required columns {missing} in file {source_label}")

    initial_rows = len(df)

    # Parse dates
    df["Date"] = _attempt_parse_dates(df["Date"])

    # Convert goal columns to numeric (coerce errors to NaN)
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce").astype("Float64")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce").astype("Float64")

    # Normalize team name strings
    df["HomeTeam"] = _normalize_team_name(df["HomeTeam"])
    df["AwayTeam"] = _normalize_team_name(df["AwayTeam"])

    # Standardize FTR: trim, upper-case single-letter H/D/A
    df["FTR"] = df["FTR"].astype("string").str.strip().str.upper()
    df["FTR"] = df["FTR"].replace(
        {
            "HOME": "H",
            "AWAY": "A",
            "DRAW": "D",
            "D": "D",
            "H": "H",
            "A": "A",
        }
    )

    # Remove rows with missing critical values
    crit_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    before_drop = len(df)
    df = df.dropna(subset=crit_cols)
    after_drop = len(df)

    # Convert goal columns to integer dtype now that NaNs are removed
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Drop exact duplicates (all columns equal)
    before_dedup = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)

    logger.info(
        "Processed %s: rows read=%d, dropped_invalid=%d, dropped_duplicates=%d, final=%d",
        source_label,
        initial_rows,
        before_drop - after_drop,
        before_dedup - after_dedup,
        after_dedup,
    )
    return df.reset_index(drop=True)


def load_all_data(data_dir: str = "data/raw", pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load all CSV season files from data_dir, clean them, and return a merged DataFrame.

    Args:
        data_dir: directory containing season CSV files (default: "data/raw")
        pattern: glob pattern to match files (default: "*.csv")

    Returns:
        pandas.DataFrame: merged and cleaned dataset containing all seasons, sorted by Date.

    Raises:
        FileNotFoundError: if no CSV files are found in data_dir.
        ValueError: if a file does not contain required columns after standardization.
    """
    files = _find_csv_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching pattern {pattern}")

    cleaned_frames: List[pd.DataFrame] = []
    loaded_files: List[str] = []
    for file_path in files:
        try:
            # Read CSV with low_memory=False to avoid dtype inference issues across columns
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            logger.warning("Failed to read %s: %s; skipping", file_path, e)
            continue

        try:
            cleaned = _clean_dataframe(df, source_label=str(file_path.name))
        except ValueError as e:
            # Missing required columns: log and skip file
            logger.error("Skipping file %s due to error: %s", file_path, e)
            continue
        except Exception as e:
            logger.exception("Unexpected error while processing %s: %s", file_path, e)
            continue

        cleaned["season_file"] = file_path.name  # provenance column (optional, useful for debugging)
        cleaned_frames.append(cleaned)
        loaded_files.append(str(file_path))

    if not cleaned_frames:
        raise ValueError(f"No valid dataframes could be loaded from {data_dir}")

    # Concatenate all seasons
    full = pd.concat(cleaned_frames, ignore_index=True)

    # Sort by Date ascending
    full = full.sort_values(by="Date").reset_index(drop=True)

    # Final dataset size logging
    logger.info("Files loaded (%d): %s", len(loaded_files), loaded_files)
    logger.info("Final dataset: rows=%d, columns=%d", full.shape[0], full.shape[1])

    return full


# If run as script, demonstrate loading (but don't execute automatically in import)
if __name__ == "__main__":
    try:
        df_all = load_all_data()
        print(df_all.head())
    except Exception as exc:
        logger.error("Failed to load data: %s", exc)
