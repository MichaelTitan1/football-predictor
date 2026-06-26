"""
prepare_merged_dataset.py

One-shot helper: turns raw football-data.co.uk CSV downloads into
    data/processed/merged_dataset.csv

If `data/raw/` is empty, auto-downloads from football-data.co.uk via
`src.data_pipeline.data_downloader`.

Usage:
    python src/data_pipeline/prepare_merged_dataset.py
    python src/data_pipeline/prepare_merged_dataset.py --raw-dir data/raw --out data/processed/merged_dataset.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402

from data_loader import _clean_dataframe, _find_csv_files  # noqa: E402

logger = logging.getLogger("prepare_merged_dataset")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def build_merged_dataset(
    raw_dir: str = "data/raw",
    out_path: str = "data/processed/merged_dataset.csv",
    auto_download: bool = True,
) -> pd.DataFrame:
    files = _find_csv_files(raw_dir, pattern="*.csv")
    if not files:
        if auto_download:
            logger.info("No raw CSVs in %s; bootstrapping via data_downloader...", raw_dir)
            try:
                from src.data_pipeline.data_downloader import download_all_leagues
                download_all_leagues()
            except Exception as e:
                logger.error("Auto-download failed: %s", e)
            files = _find_csv_files(raw_dir, pattern="*.csv")
        if not files:
            raise FileNotFoundError(
                f"No CSV files under {raw_dir!r}. "
                "Drop football-data.co.uk season files into that directory and re-run."
            )

    cleaned = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            logger.error("Skipping %s: failed to read (%s)", f, e)
            continue
        try:
            c = _clean_dataframe(df, source_label=f.name)
        except ValueError as e:
            logger.error("Skipping %s: %s", f, e)
            continue
        cleaned.append(c)

    if not cleaned:
        raise RuntimeError(
            f"No usable CSV files in {raw_dir!r} after cleaning. "
            "Check that each file has Date/HomeTeam/AwayTeam/FTHG/FTAG/FTR."
        )

    full = pd.concat(cleaned, ignore_index=True).sort_values("Date").reset_index(drop=True)
    canonical_first = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    rest = [c for c in full.columns if c not in canonical_first]
    full = full[canonical_first + rest]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, index=False)
    logger.info("Wrote %s with %d rows and %d columns", out, len(full), len(full.columns))
    return full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/processed/merged_dataset.csv")
    args = p.parse_args()
    try:
        df = build_merged_dataset(args.raw_dir, args.out)
        print(f"OK: {len(df)} rows -> {args.out}")
        print("Columns:", list(df.columns))
    except Exception as e:
        logger.error("FAILED: %s", e)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
