from __future__ import annotations

import pandas as pd

from src.data_pipeline.canonical_data import LeagueRecord, MatchRecord, TeamRecord, connect, initialize, upsert_records
from src.data_pipeline.football_data_provider import FootballDataProvider
from src.data_pipeline.league_config import api_football_id_map, load_enabled_leagues


def test_local_sqlite_upsert_is_idempotent_and_resolves_result():
    conn = connect(":memory:")
    initialize(conn)
    league = LeagueRecord("E0", "English Premier League")
    teams = [TeamRecord("home", "Home", "E0"), TeamRecord("away", "Away", "E0")]
    scheduled = MatchRecord("provider", "E0", "2025-2026", "2026-01-01T12:00:00+00:00", "home", "away", "scheduled")
    finished = MatchRecord("provider", "E0", "2025-2026", "2026-01-01T12:00:00+00:00", "home", "away", "finished", 2, 1)
    upsert_records(conn, [league], teams, [scheduled], "provider")
    upsert_records(conn, [league], teams, [finished], "provider")
    upsert_records(conn, [league], teams, [finished], "provider")
    row = conn.execute("select count(*), status, home_score, away_score from matches").fetchone()
    assert row[0] == 1
    assert row[1:] == ("finished", 2, 1)


def test_provider_normalizes_result_statuses_with_authoritative_codes():
    provider = FootballDataProvider(include_remote=False, raw_dir="/does/not/exist")
    frame = pd.DataFrame([
        {"League": "E0", "Date": "01/01/2026", "Time": "15:00", "HomeTeam": "Home", "AwayTeam": "Away", "FTHG": 2, "FTAG": 1, "FTR": "H"},
        {"League": "E0", "Date": "02/01/2026", "Time": "18:00", "HomeTeam": "Away", "AwayTeam": "Home", "FTHG": None, "FTAG": None, "FTR": None},
    ])
    _, _, matches = provider._normalize([frame])
    assert sorted(m.status for m in matches) == ["finished", "scheduled"]


def test_enabled_leagues_are_authoritative_38_football_data_set():
    leagues = load_enabled_leagues()
    assert len(leagues) == 38
    assert len({league.key for league in leagues}) == 38
    assert all(league.football_data_code for league in leagues)


def test_api_football_mapping_is_optional_for_football_data_only_codes():
    assert isinstance(api_football_id_map(), dict)
