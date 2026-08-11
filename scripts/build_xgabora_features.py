from __future__ import annotations

"""Build Fovra's leakage-safe ML feature dataset from xgabora's curated source.

This deliberately replaces the old football-data.co.uk feature bootstrap. It
uses the same audited xgabora source and writes the canonical six match columns
plus Fovra's existing leakage-safe engineered features. No Football-Data
URLs are downloaded by this script.
"""

import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.features.feature_engineer import build_features

SOURCE_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
START_DATE = pd.Timestamp("2010-07-01")
END_DATE = pd.Timestamp("2025-07-01")
EXPECTED_COMPLETED_MATCHES = 168120
OUT_PATH = Path(os.getenv("FOVRA_XGABORA_FEATURE_PATH", "data/processed/xgabora_merged_dataset.csv"))


def _integer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def run() -> dict:
    request = Request(SOURCE_URL, headers={"User-Agent": "Fovra/1.0"})
    with urlopen(request, timeout=180) as response:
        raw = response.read()
    source = pd.read_csv(io.BytesIO(raw))

    required = {"Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"xgabora source missing required columns: {missing}")

    frame = pd.DataFrame({
        "Date": pd.to_datetime(source["MatchDate"], errors="coerce"),
        "HomeTeam": source["HomeTeam"].astype("string").str.strip(),
        "AwayTeam": source["AwayTeam"].astype("string").str.strip(),
        "FTHG": pd.to_numeric(source["FTHome"], errors="coerce"),
        "FTAG": pd.to_numeric(source["FTAway"], errors="coerce"),
        "FTR": source["FTResult"].astype("string").str.strip().str.upper(),
        "League": source["Division"].astype("string").str.strip(),
    })
    frame = frame[(frame["Date"] >= START_DATE) & (frame["Date"] < END_DATE)].copy()
    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    frame = frame[frame["FTR"].isin(["H", "D", "A"])].copy()
    frame["FTHG"] = frame["FTHG"].astype(int)
    frame["FTAG"] = frame["FTAG"].astype(int)
    frame = frame.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"])
    frame = frame.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    if len(frame) != EXPECTED_COMPLETED_MATCHES:
        raise RuntimeError(f"Refusing feature build: expected {EXPECTED_COMPLETED_MATCHES} audited matches, got {len(frame)}")

    # Feature engineering intentionally receives only the canonical historical
    # match columns. This prevents post-match xgabora columns from becoming
    # prediction inputs through accidental passthrough.
    engineered = build_features(frame[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]])
    engineered["League"] = frame["League"].to_numpy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT_PATH.with_suffix(".tmp")
    engineered.to_csv(temp, index=False)
    temp.replace(OUT_PATH)

    result = {
        "source": "xgabora/Club-Football-Match-Data-2000-2025",
        "source_url": SOURCE_URL,
        "bootstrap_window": "2010/11-2024/25",
        "input_completed_matches": len(frame),
        "feature_rows": len(engineered),
        "feature_columns": len(engineered.columns),
        "output": str(OUT_PATH),
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
