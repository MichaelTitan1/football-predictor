"""Build the ML-ready historical feature surface directly from xgabora."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
START_DATE = pd.Timestamp("2010-07-01", tz="UTC")
END_DATE = pd.Timestamp("2025-07-01", tz="UTC")
EXPECTED_COMPLETED_MATCHES = 168120
OUT_PATH = Path(os.getenv("FOVRA_XGABORA_FEATURE_PATH", "data/processed/xgabora_merged_dataset.csv"))

PREMATCH_COLUMNS = [
    "HomeElo", "AwayElo", "Form3Home", "Form3Away", "Form5Home", "Form5Away",
    "OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
    "Over25", "Under25", "MaxOver25", "MaxUnder25", "HandiSize", "HandiHome", "HandiAway",
    "C_LTH", "C_LTA", "C_VHD", "C_VAD", "C_HTB", "C_PHB",
]


def _download() -> pd.DataFrame:
    request = Request(SOURCE_URL, headers={"User-Agent": "Fovra/xgabora-feature-build"})
    with urlopen(request, timeout=180) as response:
        return pd.read_csv(io.BytesIO(response.read()), low_memory=False)


def run() -> dict:
    source = _download()
    required = {"Division", "MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult", *PREMATCH_COLUMNS}
    missing = sorted(required - set(source.columns))
    if missing:
        raise RuntimeError(f"xgabora source missing required columns: {missing}")

    frame = pd.DataFrame({
        "Date": pd.to_datetime(source["MatchDate"], errors="coerce", utc=True),
        "HomeTeam": source["HomeTeam"].astype("string").str.strip(),
        "AwayTeam": source["AwayTeam"].astype("string").str.strip(),
        "League": source["Division"].astype("string").str.strip(),
        "FTHG": pd.to_numeric(source["FTHome"], errors="coerce"),
        "FTAG": pd.to_numeric(source["FTAway"], errors="coerce"),
        "FTR": source["FTResult"].astype("string").str.strip().str.upper(),
        **{column: pd.to_numeric(source[column], errors="coerce") for column in PREMATCH_COLUMNS},
    })

    frame = frame[(frame["Date"] >= START_DATE) & (frame["Date"] < END_DATE)].copy()
    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "League", "FTHG", "FTAG", "FTR"])
    frame = frame[frame["FTR"].isin(["H", "D", "A"])].copy()
    frame["FTHG"] = frame["FTHG"].astype(int)
    frame["FTAG"] = frame["FTAG"].astype(int)
    frame = frame.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "League"], keep="first")
    frame = frame.sort_values(["Date", "HomeTeam", "AwayTeam", "League"]).reset_index(drop=True)

    if len(frame) != EXPECTED_COMPLETED_MATCHES:
        raise RuntimeError(f"Refusing feature build: expected {EXPECTED_COMPLETED_MATCHES} audited completed matches, got {len(frame)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT_PATH.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(OUT_PATH)

    result = {
        "source": "xgabora/Club-Football-Match-Data-2000-2025",
        "source_url": SOURCE_URL,
        "bootstrap_window": "2010/11-2024/25",
        "input_completed_matches": len(frame),
        "feature_rows": len(frame),
        "feature_columns": len(frame.columns),
        "prematch_feature_columns": PREMATCH_COLUMNS,
        "historical_elo_source": "xgabora Matches.csv HomeElo/AwayElo",
        "current_clubelo_refresh": False,
        "output": str(OUT_PATH),
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
