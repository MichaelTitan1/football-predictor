from __future__ import annotations

import pandas as pd

from src.data_pipeline.canonical_data import LeagueRecord, MatchRecord, TeamRecord
from src.data_pipeline.football_data_provider import FootballDataProvider


def test_match_key_is_deterministic_and_source_id_stable():
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

from src.data_pipeline.league_config import load_enabled_leagues, league_config_map


def test_enabled_leagues_are_single_configured_v1_set():
    leagues = load_enabled_leagues()
    assert len(leagues) == 15
    assert len({league.key for league in leagues}) == 15
    assert all(league.football_data_code for league in leagues)
    assert all(league.api_football_id for league in leagues)


def test_downloader_uses_configured_leagues():
    configured = league_config_map()
    from src.data_pipeline import data_downloader
    assert data_downloader.LEAGUE_CONFIG == configured
    assert data_downloader.ALLOWED_LEAGUES == list(configured.keys())

from src.data_pipeline.api_football_provider import APIFootballProvider, APIFootballProviderError


class _FakeAPIFootballProvider(APIFootballProvider):
    def __init__(self):
        self.api_key = "test"
        self.preferred_season = None
        from src.data_pipeline.league_config import api_football_id_map
        self._league_by_api_id = api_football_id_map()
        self._cache = {}
        self._season_cache = {}
        self._season_candidates_cache = {}
        self.match_metadata = {}
        self.calls = []
        self.request_count = 0

    def _get(self, path, **params):
        self.calls.append((path, params))
        self.request_count += 1
        if path == "leagues":
            return [{"seasons": [{"year": 2026, "coverage": {"fixtures": {"events": True}}}, {"year": 2024, "coverage": {"fixtures": {"events": True}}}]}]
        return []


def test_api_football_results_use_date_batches_not_per_league():
    provider = _FakeAPIFootballProvider()
    provider.fetch(mode="results")
    fixture_calls = [(path, params) for path, params in provider.calls if path == "fixtures"]
    assert len(fixture_calls) == 2
    assert all("date" in params and "season" in params and "league" not in params for _, params in fixture_calls)


def test_api_football_fixture_refresh_is_once_per_configured_league():
    provider = _FakeAPIFootballProvider()
    provider.fetch(mode="fixtures")
    fixture_calls = [(path, params) for path, params in provider.calls if path == "fixtures"]
    assert len(fixture_calls) == len(load_enabled_leagues())
    assert all("league" in params and "season" in params for _, params in fixture_calls)


def test_latest_season_check_does_not_bootstrap_without_local_baseline(monkeypatch, tmp_path):
    from src.data_pipeline import data_downloader
    monkeypatch.setattr(data_downloader, "RAW_DIR", tmp_path)
    called = []
    monkeypatch.setattr(data_downloader, "download_season_data", lambda *args, **kwargs: called.append(args) or True)
    result = data_downloader.update_latest_season()
    assert sum(len(v) for v in result.values()) == 0
    assert called == []


def test_api_football_falls_back_when_newest_season_is_not_on_free_plan():
    class FallbackProvider(_FakeAPIFootballProvider):
        def _get(self, path, **params):
            self.calls.append((path, params))
            self.request_count += 1
            if path == "leagues":
                return [{"seasons": [{"year": 2026, "coverage": {"fixtures": {"events": True}}}, {"year": 2024, "coverage": {"fixtures": {"events": True}}}]}]
            if path == "fixtures" and params.get("season") == 2026:
                raise APIFootballProviderError("fixtures", {"plan": "Free plans do not have access to this season, try from 2022 to 2024."})
            return []

    provider = FallbackProvider()
    provider.fetch(mode="fixtures")
    fixture_seasons = [params["season"] for path, params in provider.calls if path == "fixtures"]
    assert 2026 in fixture_seasons
    assert 2024 in fixture_seasons
