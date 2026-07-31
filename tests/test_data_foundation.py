from __future__ import annotations

import pandas as pd

from src.data_pipeline.api_football_provider import APIFootballProvider, APIFootballProviderError
from src.data_pipeline.canonical_data import (
    LeagueRecord,
    MatchRecord,
    TeamRecord,
    connect,
    initialize,
    upsert_records,
)
from src.data_pipeline.football_data_provider import FootballDataProvider
from src.data_pipeline.league_config import api_football_id_map, load_enabled_leagues, league_config_map


def test_match_key_is_deterministic_and_source_id_stable():
    a = MatchRecord(
        "provider",
        "EPL",
        "2025-2026",
        "2026-01-01T15:00:00+00:00",
        "home",
        "away",
        source_id="source-1",
    )
    b = MatchRecord(
        "provider",
        "EPL",
        "2025-2026",
        "2026-01-01T16:00:00+00:00",
        "home",
        "away",
        source_id="source-1",
    )

    assert a.match_key == b.match_key


def test_local_sqlite_upsert_is_idempotent_and_resolves_result():
    conn = connect(":memory:")
    initialize(conn)
    league = LeagueRecord("EPL", "Example League")
    teams = [TeamRecord("home", "Home", "EPL"), TeamRecord("away", "Away", "EPL")]
    scheduled = MatchRecord(
        "provider",
        "EPL",
        "2025-2026",
        "2026-01-01T15:00:00+00:00",
        "home",
        "away",
    )
    finished = MatchRecord(
        "provider",
        "EPL",
        "2025-2026",
        "2026-01-01T15:00:00+00:00",
        "home",
        "away",
        "finished",
        2,
        1,
    )
    upsert_records(conn, [league], teams, [scheduled], "provider")
    upsert_records(conn, [league], teams, [finished], "provider")
    row = conn.execute("select count(*), status, home_score, away_score from matches").fetchone()
    assert tuple(row) == (1, "finished", 2, 1)


def test_provider_normalizes_result_statuses():
    provider = FootballDataProvider(include_remote=False, raw_dir="/does/not/exist")
    frame = pd.DataFrame(
        [
            {
                "League": "EPL",
                "Date": "01/01/2026",
                "Time": "15:00",
                "HomeTeam": "Home",
                "AwayTeam": "Away",
                "FTHG": 2,
                "FTAG": 1,
                "FTR": "H",
            },
            {
                "League": "EPL",
                "Date": "02/01/2026",
                "Time": "18:00",
                "HomeTeam": "Away",
                "AwayTeam": "Home",
                "FTHG": None,
                "FTAG": None,
                "FTR": None,
            },
        ]
    )
    _, _, matches = provider._normalize([frame])
    assert sorted(m.status for m in matches) == ["finished", "scheduled"]


def test_enabled_leagues_are_single_configured_v1_set():
    leagues = load_enabled_leagues()
    assert len(leagues) == 24
    assert len({league.key for league in leagues}) == 24
    assert all(league.football_data_code for league in leagues)
    assert all(league.api_football_id for league in leagues)


def test_downloader_uses_configured_leagues():
    configured = league_config_map()
    from src.data_pipeline import data_downloader

    assert data_downloader.LEAGUE_CONFIG == configured
    assert data_downloader.ALLOWED_LEAGUES == list(configured.keys())


class _FakeAPIFootballProvider(APIFootballProvider):
    def __init__(self):
        self.api_key = "test"
        self.preferred_season = None
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
            return [
                {
                    "seasons": [
                        {"year": 2026, "coverage": {"fixtures": {"events": True}}},
                        {"year": 2024, "coverage": {"fixtures": {"events": True}}},
                    ]
                }
            ]
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


def test_api_football_fixture_refresh_can_run_in_league_batches(monkeypatch):
    monkeypatch.setenv("FOVRA_API_FOOTBALL_BATCH_TOTAL", "3")
    monkeypatch.setenv("FOVRA_API_FOOTBALL_BATCH_INDEX", "1")
    provider = _FakeAPIFootballProvider()
    provider.fetch(mode="fixtures")
    fixture_calls = [(path, params) for path, params in provider.calls if path == "fixtures"]
    expected_leagues = load_enabled_leagues()[1::3]
    assert len(fixture_calls) == len(expected_leagues)
    assert [params["league"] for _, params in fixture_calls] == [league.api_football_id for league in expected_leagues]


def test_api_football_standings_can_run_in_league_batches(monkeypatch):
    monkeypatch.setenv("FOVRA_API_FOOTBALL_BATCH_TOTAL", "3")
    monkeypatch.setenv("FOVRA_API_FOOTBALL_BATCH_INDEX", "2")
    provider = _FakeAPIFootballProvider()
    provider.fetch_standings()
    standings_calls = [(path, params) for path, params in provider.calls if path == "standings"]
    expected_leagues = load_enabled_leagues()[2::3]
    assert len(standings_calls) == len(expected_leagues)
    assert [params["league"] for _, params in standings_calls] == [league.api_football_id for league in expected_leagues]


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
                return [
                    {
                        "seasons": [
                            {"year": 2026, "coverage": {"fixtures": {"events": True}}},
                            {"year": 2024, "coverage": {"fixtures": {"events": True}}},
                        ]
                    }
                ]
            if path == "fixtures" and params.get("season") == 2026:
                raise APIFootballProviderError(
                    "fixtures",
                    {"plan": "Free plans do not have access to this season, try from 2022 to 2024."},
                )
            return []

    provider = FallbackProvider()
    provider.fetch(mode="fixtures")
    fixture_seasons = [params["season"] for path, params in provider.calls if path == "fixtures"]
    assert 2026 in fixture_seasons
    assert 2024 in fixture_seasons


def test_prediction_service_consumes_team_statistics_before_prediction():
    from src.prediction.canonical_service import CanonicalPredictionService

    service = object.__new__(CanonicalPredictionService)
    row = pd.DataFrame(
        [
            {
                "home_elo_prior": 1500.0,
                "away_elo_prior": 1500.0,
                "expected_home_xg": 1.0,
                "expected_away_xg": 1.0,
                "xg_diff": 0.0,
                "elo_diff_home_minus_away": 0.0,
            }
        ]
    )
    context = {
        "home_team_statistics": {"xg": 2.1, "goals": 2.4, "xga": 0.8},
        "away_team_statistics": {"xg": 0.9, "goals": 1.0, "xga": 1.7},
        "home_team_strength": {"elo": 1700.0},
        "away_team_strength": {"elo": 1500.0},
    }

    enriched = service._apply_enrichment_to_row(row, context)

    assert enriched.loc[0, "expected_home_xg"] == 2.1
    assert enriched.loc[0, "expected_away_xg"] == 0.9
    assert enriched.loc[0, "home_elo_prior"] == 1700.0
    assert enriched.loc[0, "away_elo_prior"] == 1500.0
    assert enriched.loc[0, "xg_diff"] == 1.2000000000000002
    assert enriched.loc[0, "elo_diff_home_minus_away"] == 200.0
