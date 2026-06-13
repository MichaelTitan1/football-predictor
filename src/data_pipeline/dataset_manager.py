"""
dataset_manager.py

Dataset manager for the football prediction system.

Responsibilities:
- Scan CSVs in data/raw/
- Validate schema consistency (supports football-data.co.uk style CSVs and common variants)
- Detect duplicate matches (Date + HomeTeam + AwayTeam)
- Merge seasons into one master dataset
- Generate dataset summary statistics
- Save merged dataset to data/processed/merged_dataset.csv (atomic write)

Public functions:
- validate_dataset(data_dir: str = "data/raw") -> dict
- merge_datasets(data_dir: str = "data/raw", processed_path: str = "data/processed/merged_dataset.csv") -> pd.DataFrame
- dataset_report(merged_df: Optional[pd.DataFrame] = None, processed_path: str = "data/processed/merged_dataset.csv") -> dict

Design goals:
- Defensive parsing of dates (infer formats, support common football-data date formats)
- Schema alignment via alias mapping for typical column name variants
- Safe, idempotent merging with duplicate detection and logging
- Minimal external deps: pandas + standard library

"""
from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Canonical columns we expect for the downstream pipeline
_CANONICAL_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Common column name aliases (map variant -> canonical)
_ALIAS_MAP = {
    "Home Team": "HomeTeam",
    "Away Team": "AwayTeam",
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "HS": "HomeShots",
    "AS": "AwayShots",
    "FTHG": "FTHG",
    "FTAG": "FTAG",
    "FTR": "FTR",
    "Res": "FTR",
    "Result": "FTR",
    "Date": "Date",
}


def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=str(target.parent), suffix=".tmp") as tf:
        tmp_path = Path(tf.name)
        df.to_csv(tmp_path, index=False)
    shutil.move(str(tmp_path), str(target))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names using alias map and strip whitespace."""
    col_map = {}
    for c in df.columns:
        c_stripped = str(c).strip()
        mapped = _ALIAS_MAP.get(c_stripped, c_stripped)
        col_map[c] = mapped
    df = df.rename(columns=col_map)
    return df


def _coerce_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical columns exist when possible, coerce types and normalize FTR."""
    df = df.copy()

    # Date parsing
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", infer_datetime_format=True)
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Goals to numeric
    for g in ["FTHG", "FTAG"]:
        if g in df.columns:
            df[g] = pd.to_numeric(df[g], errors="coerce")

    # Normalize FTR to H/D/A
    if "FTR" in df.columns:
        df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
        df["FTR"] = df["FTR"].replace({"HOME": "H", "AWAY": "A", "DRAW": "D", "": None})

    return df


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.exception("Failed to read %s: %s", path, e)
        return None


def validate_dataset(data_dir: str = "data/raw") -> Dict[str, Dict]:
    """Scan CSVs in data_dir and validate each file's schema consistency.

    Returns a dict keyed by filename with validation details:
    { filename: { 'ok': bool, 'missing_columns': [...], 'extra_columns': [...], 'rows': int, 'parsed_dates': int } }
    """
    p = Path(data_dir)
    report = {}
    if not p.exists():
        logger.warning("Data directory does not exist: %s", data_dir)
        return report

    files = sorted(p.glob("*.csv"))
    for f in files:
        info = {"ok": True, "missing_columns": [], "extra_columns": [], "rows": 0, "parsed_dates": 0}
        df = _read_csv_safe(f)
        if df is None:
            info["ok"] = False
            report[f.name] = info
            continue
        info["rows"] = len(df)
        df = _normalize_columns(df)
        # Check canonical presence
        missing = [c for c in _CANONICAL_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in set(_CANONICAL_COLS) and c not in _ALIAS_MAP.values()]
        info["missing_columns"] = missing
        info["extra_columns"] = extra

        # Try to coerce date parsing count
        df2 = _coerce_standard_columns(df)
        if "Date" in df2.columns:
            parsed = df2["Date"].notna().sum()
        else:
            parsed = 0
        info["parsed_dates"] = int(parsed)
        if missing:
            info["ok"] = False
        report[f.name] = info
        logger.info("Validated %s: rows=%d parsed_dates=%d missing=%s extras=%d", f.name, info["rows"], info["parsed_dates"], missing, len(extra))
    return report


def merge_datasets(data_dir: str = "data/raw", processed_path: str = "data/processed/merged_dataset.csv") -> pd.DataFrame:
    """Read all CSV files from data_dir, normalize, concatenate, deduplicate by Date+HomeTeam+AwayTeam, and save merged dataset.

    Returns the merged DataFrame.
    """
    p = Path(data_dir)
    merged = []
    if not p.exists():
        logger.warning("Raw data directory %s does not exist", data_dir)
        df_merged = pd.DataFrame(columns=_CANONICAL_COLS)
        _atomic_write_csv(df_merged, Path(processed_path))
        return df_merged

    files = sorted(p.glob("*.csv"))
    for f in files:
        df = _read_csv_safe(f)
        if df is None:
            continue
        df = _normalize_columns(df)
        df = _coerce_standard_columns(df)

        # Keep only canonical columns plus any extra columns that may be useful
        keep = [c for c in df.columns if c in set(_CANONICAL_COLS) or c not in _ALIAS_MAP.values()]
        df = df.loc[:, [c for c in keep if c in df.columns]]

        # Ensure canonical columns exist (fill with NaN if missing)
        for c in _CANONICAL_COLS:
            if c not in df.columns:
                df[c] = np.nan

        # Reorder canonical first
        cols = _CANONICAL_COLS + [c for c in df.columns if c not in _CANONICAL_COLS]
        df = df.loc[:, cols]

        merged.append(df)
        logger.info("Loaded %s (%d rows)", f.name, len(df))

    if not merged:
        logger.warning("No CSVs read from %s", data_dir)
        df_merged = pd.DataFrame(columns=_CANONICAL_COLS)
        _atomic_write_csv(df_merged, Path(processed_path))
        return df_merged

    df_merged = pd.concat(merged, ignore_index=True, sort=False)

    # Normalize FTR again and drop fully empty date rows
    df_merged["FTR"] = df_merged["FTR"].astype(str).str.strip().str.upper().replace({"NAN": None, "NONE": None})
    df_merged["Date"] = pd.to_datetime(df_merged["Date"], dayfirst=True, errors="coerce", infer_datetime_format=True)

    before = len(df_merged)
    # Drop rows without date or home/away
    df_merged = df_merged.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    after_drop = len(df_merged)
    if after_drop != before:
        logger.info("Dropped %d rows without essential identifiers", before - after_drop)

    # Deduplicate by Date+HomeTeam+AwayTeam keeping last occurrence (prefer later files)
    df_merged["_dup_key"] = df_merged["Date"].dt.strftime("%Y-%m-%d") + "__" + df_merged["HomeTeam"].astype(str) + "__" + df_merged["AwayTeam"].astype(str)
    before_dup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=["_dup_key"], keep="last").drop(columns=["_dup_key"]).reset_index(drop=True)
    after_dup = len(df_merged)
    logger.info("Removed %d duplicate matches during merge", before_dup - after_dup)

    # Sort by Date
    df_merged = df_merged.sort_values("Date").reset_index(drop=True)

    # Save merged dataset atomically
    _atomic_write_csv(df_merged, Path(processed_path))
    logger.info("Wrote merged dataset to %s (%d rows)", processed_path, len(df_merged))
    return df_merged


def dataset_report(merged_df: Optional[pd.DataFrame] = None, processed_path: str = "data/processed/merged_dataset.csv") -> Dict:
    """Generate summary statistics for the merged dataset.

    If merged_df not provided, reads from processed_path.

    Returns dict with:
      - rows, seasons (years), teams count
      - goals distribution stats
      - per-season accuracy of available FTR (counts)
      - files summary if raw dir present (calls validate_dataset)
    """
    if merged_df is None:
        p = Path(processed_path)
        if not p.exists():
            logger.warning("Processed dataset not found at %s", processed_path)
            return {}
        merged_df = pd.read_csv(p, parse_dates=["Date"])

    df = merged_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)

    rows = len(df)
    seasons = sorted(list(df["Date"].dt.year.unique()))
    teams = sorted(set(df["HomeTeam"].dropna().unique()).union(set(df["AwayTeam"].dropna().unique())))

    goals = None
    if "FTHG" in df.columns and "FTAG" in df.columns:
        gf = df["FTHG"].dropna().astype(float)
        ga = df["FTAG"].dropna().astype(float)
        goals = {
            "home_goals_mean": float(gf.mean()) if not gf.empty else None,
            "away_goals_mean": float(ga.mean()) if not ga.empty else None,
            "home_goals_std": float(gf.std()) if not gf.empty else None,
            "away_goals_std": float(ga.std()) if not ga.empty else None,
        }

    # Per-season FTR completeness and distribution
    seasonal = {}
    if "FTR" in df.columns:
        for y, grp in df.groupby(df["Date"].dt.year):
            total = len(grp)
            present = grp["FTR"].notna().sum()
            dist = grp["FTR"].value_counts(dropna=True).to_dict()
            seasonal[int(y)] = {"rows": int(total), "ftr_present": int(present), "distribution": {k: int(v) for k, v in dist.items()}}

    report = {
        "rows": int(rows),
        "seasons": seasons,
        "team_count": len(teams),
        "teams_sample": teams[:10],
        "goals": goals,
        "seasonal_summary": seasonal,
    }
    logger.info("Dataset report: rows=%d seasons=%s teams=%d", rows, seasons, len(teams))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running dataset_manager merge flow")
    merged = merge_datasets()
    rep = dataset_report(merged)
    import json

    print(json.dumps(rep, indent=2))
