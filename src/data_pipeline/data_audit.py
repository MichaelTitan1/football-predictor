"""Read-only audit of repository football data.

The command never downloads, modifies, deletes, or fabricates data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED = {"Date", "HomeTeam", "AwayTeam"}
RESULT_COLS = {"FTHG", "FTAG", "FTR"}


def audit(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> dict:
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    files = sorted(raw.glob("*.csv")) if raw.exists() else []
    report = {
        "raw_directory_exists": raw.exists(),
        "processed_directory_exists": processed.exists(),
        "raw_files": [],
        "processed_files": [],
        "total_rows": 0,
        "finished_rows": 0,
        "scheduled_rows": 0,
        "leagues": [],
        "teams": [],
        "newest_match_date": None,
        "days_stale": None,
    }

    all_frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
            cols = set(df.columns)
            missing = sorted(REQUIRED - cols)
            parsed = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce") if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
            report["raw_files"].append({
                "file": str(path),
                "rows": int(len(df)),
                "missing_required": missing,
                "min_date": parsed.min().date().isoformat() if parsed.notna().any() else None,
                "max_date": parsed.max().date().isoformat() if parsed.notna().any() else None,
            })
            if not missing:
                all_frames.append(df)
        except Exception as exc:
            report["raw_files"].append({"file": str(path), "error": str(exc)})

    for path in sorted(processed.glob("*.csv")) if processed.exists() else []:
        try:
            df = pd.read_csv(path)
            report["processed_files"].append({"file": str(path), "rows": int(len(df)), "columns": list(df.columns)})
        except Exception as exc:
            report["processed_files"].append({"file": str(path), "error": str(exc)})

    if all_frames:
        df = pd.concat(all_frames, ignore_index=True, sort=False)
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        report["total_rows"] = int(len(df))
        report["finished_rows"] = int(df[list(RESULT_COLS & set(df.columns))].notna().all(axis=1).sum()) if RESULT_COLS.issubset(df.columns) else 0
        report["scheduled_rows"] = report["total_rows"] - report["finished_rows"]
        report["leagues"] = sorted(df["League"].dropna().astype(str).unique().tolist()) if "League" in df.columns else []
        report["teams"] = sorted(set(df["HomeTeam"].dropna().astype(str)).union(set(df["AwayTeam"].dropna().astype(str))))
        if dates.notna().any():
            newest = dates.max().date()
            report["newest_match_date"] = newest.isoformat()
            report["days_stale"] = (datetime.now(timezone.utc).date() - newest).days

    return report


def main() -> None:
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
