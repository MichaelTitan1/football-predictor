from __future__ import annotations

import pandas as pd

from src.data_pipeline import fast_football_data_ingest as fast


def test_single_feed_is_one_all_seasons_file_and_filtered_to_2010(monkeypatch, tmp_path):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n15/08/2009,Old,Team,1,0,H\n15/08/2012,Alpha,Beta,2,1,H\n20/08/2025,Gamma,Delta,0,0,D\n"
    monkeypatch.setattr(fast, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fast, "_fetch_single_feed", lambda code: csv)
    monkeypatch.setattr(fast, "_state_path", lambda: tmp_path / "state.json")

    result = fast.refresh_single_leagues()

    assert set(result["refreshed"]) == set(fast.SINGLE_LEAGUES)
    frame = pd.read_csv(tmp_path / "ARG_new.csv")
    assert len(frame) == 2
    assert frame["Date"].min() >= "2012-08-15"
    assert set(frame["League"]) == {"ARG"}


def test_single_feed_failure_is_not_silently_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(fast, "RAW_DIR", tmp_path)
    monkeypatch.setattr(fast, "_state_path", lambda: tmp_path / "state.json")

    def fail(code):
        raise RuntimeError(f"source unavailable: {code}")

    monkeypatch.setattr(fast, "_fetch_single_feed", fail)

    try:
        fast.refresh_single_leagues()
    except RuntimeError as exc:
        assert "Football-Data extra leagues failed" in str(exc)
    else:
        raise AssertionError("single-feed failure must fail the ingestion task")
