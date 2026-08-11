"""Read-only audit of the xgabora historical football dataset.

This deliberately performs NO Neon writes. It downloads xgabora's published
Matches.csv, filters the proposed 2010/11-2024/25 bootstrap window, and checks
schema, dates, duplicate match identities, result integrity, league coverage,
and pre-match feature availability before any migration is attempted.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import pandas as pd

MATCHES_URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
)
START_DATE = pd.Timestamp("2010-07-01", tz="UTC")
END_DATE = pd.Timestamp("2025-08-01", tz="UTC")
REQUIRED = {
    "Division", "MatchDate", "HomeTeam", "AwayTeam",
    "FTHome", "FTAway", "FTResult",
}
PREMATCH = {
    "HomeElo", "AwayElo", "Form3Home", "Form3Away",
    "Form5Home", "Form5Away", "OddHome", "OddDraw", "OddAway",
}


def load_matches() -> pd.DataFrame:
    request = Request(MATCHES_URL, headers={"User-Agent": "Fovra/xgabora-audit"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload), low_memory=False)


def canonical_identity(frame: pd.DataFrame) -> pd.Series:
    parts = (
        frame["Division"].fillna("").astype(str).str.strip().str.lower()
        + "|" + frame["MatchDate"].astype(str)
        + "|" + frame["HomeTeam"].fillna("").astype(str).str.strip().str.lower()
        + "|" + frame["AwayTeam"].fillna("").astype(str).str.strip().str.lower()
    )
    return parts.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def main() -> int:
    df = load_matches()
    original_rows = len(df)
    missing_columns = sorted(REQUIRED - set(df.columns))
    if missing_columns:
        print(json.dumps({"status": "FAIL", "reason": "missing_columns", "missing": missing_columns}, indent=2))
        return 1

    dates = pd.to_datetime(df["MatchDate"], errors="coerce", utc=True)
    df = df.assign(_date=dates)
    window = df[(df["_date"] >= START_DATE) & (df["_date"] < END_DATE)].copy()

    future_rows = int((window["_date"] > pd.Timestamp.now(tz="UTC")).sum())
    invalid_dates = int(window["_date"].isna().sum())
    missing_scores = int(window[["FTHome", "FTAway", "FTResult"]].isna().any(axis=1).sum())
    bad_results = int(~window["FTResult"].astype(str).str.upper().isin({"H", "D", "A"}).fillna(False).sum())

    identities = canonical_identity(window)
    duplicate_rows = int(identities.duplicated(keep=False).sum())
    unique_matches = int(identities.nunique())

    feature_coverage = {
        column: round(float(window[column].notna().mean()), 6)
        for column in sorted(PREMATCH & set(window.columns))
    }

    report = {
        "status": "PASS" if not (future_rows or invalid_dates or missing_scores or bad_results or duplicate_rows) else "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "xgabora/Club-Football-Match-Data-2000-2025",
        "source_url": MATCHES_URL,
        "original_rows": original_rows,
        "bootstrap_window": "2010/11-2024/25",
        "window_rows": len(window),
        "unique_match_identities": unique_matches,
        "duplicate_rows_by_basic_identity": duplicate_rows,
        "invalid_dates": invalid_dates,
        "future_dated_rows": future_rows,
        "missing_score_or_result_rows": missing_scores,
        "invalid_result_code_rows": bad_results,
        "date_min": None if window.empty else window["_date"].min().isoformat(),
        "date_max": None if window.empty else window["_date"].max().isoformat(),
        "league_count": int(window["Division"].nunique(dropna=True)),
        "leagues": sorted(window["Division"].dropna().astype(str).unique().tolist()),
        "prematch_feature_coverage": feature_coverage,
        "post_match_columns_not_used_as_prediction_features": [
            "FTHome", "FTAway", "FTResult", "HTHome", "HTAway", "HTResult",
            "HomeShots", "AwayShots", "HomeTarget", "AwayTarget",
            "HomeFouls", "AwayFouls", "HomeCorners", "AwayCorners",
            "HomeYellow", "AwayYellow", "HomeRed", "AwayRed",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
