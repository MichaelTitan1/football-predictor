from __future__ import annotations

import pandas as pd

from src.data_pipeline.fast_football_data_ingest_v2 import _normalise_frame


def test_extra_league_schema_is_normalized_to_canonical_columns():
    raw = pd.DataFrame(
        [
            {"League": "ARG", "Date": "01/01/2010", "Time": "15:00", "Home": "Alpha", "Away": "Beta", "HG": 2, "AG": 1, "Res": "H"},
            {"League": "ARG", "Date": "02/01/2025", "Time": "18:00", "Home": "Gamma", "Away": "Delta", "HG": 0, "AG": 0, "Res": "D"},
        ]
    )
    out = _normalise_frame(raw, "ARG")
    assert {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}.issubset(out.columns)
    assert len(out) == 2
    assert out.iloc[0]["HomeTeam"] == "Alpha"
    assert out.iloc[0]["FTHG"] == 2
    assert out.iloc[1]["FTR"] == "D"


def test_extra_league_filter_is_date_based_not_season_download_based():
    raw = pd.DataFrame(
        [
            {"Date": "31/12/2009", "Home": "Old", "Away": "Old2", "HG": 1, "AG": 0, "Res": "H"},
            {"Date": "01/01/2010", "Home": "Keep", "Away": "Keep2", "HG": 1, "AG": 1, "Res": "D"},
        ]
    )
    out = _normalise_frame(raw, "ARG")
    assert len(out) == 1
    assert out.iloc[0]["HomeTeam"] == "Keep"


def test_exact_single_feed_url_set_is_configured():
    from src.data_pipeline.fast_football_data_ingest_v2 import SINGLE_URLS
    expected_codes = {"ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "RUS", "SWE", "SWZ", "USA"}
    assert set(SINGLE_URLS) == expected_codes
    assert all(SINGLE_URLS[c] == f"https://www.football-data.co.uk/new/{c}.csv" for c in expected_codes)
