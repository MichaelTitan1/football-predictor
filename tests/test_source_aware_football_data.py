from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_pipeline.football_data_provider import FootballDataProvider


def _write(path: Path, rows: list[str]) -> None:
    path.write_text(
        "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,MatchID,League\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_season_filename_is_authoritative_for_season(tmp_path: Path):
    _write(
        tmp_path / "E0_2025.csv",
        ["09/08/2025,15:00,Alpha,Beta,2,1,H,source-1,E0"],
    )
    snapshot = FootballDataProvider(tmp_path, include_remote=False).fetch()
    assert len(snapshot.matches) == 1
    match = snapshot.matches[0]
    assert match.season == "2025-2026"
    assert match.kickoff_utc.startswith("2025-08-09T14:00:00")
    assert match.status == "finished"


def test_date_outside_source_season_is_rejected(tmp_path: Path):
    _write(
        tmp_path / "E0_2025.csv",
        ["01/09/2026,15:00,Alpha,Beta,3,3,H,source-bad,E0"],
    )
    try:
        FootballDataProvider(tmp_path, include_remote=False).fetch()
    except RuntimeError as exc:
        assert "no valid canonical matches" in str(exc)
    else:
        raise AssertionError("a date outside the season source bounds was accepted")


def test_future_result_with_scores_is_not_finished(tmp_path: Path):
    future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)).strftime("%d/%m/%Y")
    _write(
        tmp_path / "E0_2026.csv",
        [f"{future},15:00,Alpha,Beta,3,3,H,source-future,E0"],
    )
    try:
        FootballDataProvider(tmp_path, include_remote=False).fetch()
    except RuntimeError as exc:
        assert "no valid canonical matches" in str(exc)
    else:
        raise AssertionError("a future scored result was accepted as canonical data")


def test_combined_new_feed_can_contain_multiple_seasons(tmp_path: Path):
    _write(
        tmp_path / "ARG_new.csv",
        [
            "10/08/2024,15:00,Alpha,Beta,1,0,H,source-old,ARG",
            "10/08/2025,15:00,Gamma,Delta,0,0,D,source-new,ARG",
        ],
    )
    snapshot = FootballDataProvider(tmp_path, include_remote=False).fetch()
    assert {m.season for m in snapshot.matches} == {"2024-2025", "2025-2026"}


def test_unknown_raw_csv_is_not_merged_into_canonical_source(tmp_path: Path):
    _write(tmp_path / "random.csv", ["10/08/2025,15:00,Alpha,Beta,1,0,H,source-x,E0"])
    try:
        FootballDataProvider(tmp_path, include_remote=False).fetch()
    except RuntimeError as exc:
        assert "no valid canonical matches" in str(exc)
    else:
        raise AssertionError("unclassified CSV was accepted into the canonical source")
