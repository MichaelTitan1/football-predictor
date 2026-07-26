from __future__ import annotations

import pandas as pd

from src.data_pipeline.canonical_data import LeagueRecord, MatchRecord, TeamRecord, connect, initialize, upsert_records
from src.data_pipeline.football_data_provider import FootballDataProvider


def test_match_key_is_deterministic_and_source_id_stable():
    a = MatchRecord("provider", "EPL", "2025-2026", "2026-01-01T15:00:00+00:00", "home", "away", source_id="source-1")
    b = MatchRecord("provider", "EPL", "2025-2026", "2026-01-01T16:00:00+00:00", "home", "away", source_id="source-1")
    assert a.match_key == b.match_key


def test_local_sqlite_upsert_is_idempotent_and_resolves_result():
    conn = connect(":memory:")
    initialize(conn)
    league = LeagueRecord("EPL", "Example League")
    teams = [TeamRecord("home", "Home", "EPL"), TeamRecord("away", "Away", "EPL")]
    scheduled = MatchRecord("provider", "EPL", "2025-2026", "2026-01-01T15:00:00+00:00", "home", "away")
    finished = MatchRecord("provider", "EPL", "2025-2026", "2026-01-01T15:00:00+00:00", "home", "away", "finished", 2, 1)
    upsert_records(conn, [league], teams, [scheduled], "provider")
    upsert_records(conn, [league], teams, [finished], "provider")
    row = conn.execute("select count(*), status, home_score, away_score from matches").fetchone()
    assert tuple(row) == (1, "finished", 2, 1)


def test_provider_normalizes_result_statuses():
    provider = FootballDataProvider(include_remote=False, raw_dir="/does/not/exist")
    frame = pd.DataFrame([
        {"League":"EPL","Date":"01/01/2026","Time":"15:00","HomeTeam":"Home","AwayTeam":"Away","FTHG":2,"FTAG":1,"FTR":"H"},
        {"League":"EPL","Date":"02/01/2026","Time":"18:00","HomeTeam":"Away","AwayTeam":"Home","FTHG":None,"FTAG":None,"FTR":None},
    ])
    _, _, matches = provider._normalize([frame])
    assert sorted(m.status for m in matches) == ["finished", "scheduled"]
