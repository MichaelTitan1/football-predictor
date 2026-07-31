from __future__ import annotations

import pandas as pd
import requests

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

class _FakeSupabaseResponse:
    status_code = 200
    text = ""

    def __init__(self, data=None):
        self._data = data
        self.content = b"[]" if data is not None else b""

    def json(self):
        return self._data


class _FakeSupabaseSession:
    def __init__(self, existing=None, fail_match_posts=None):
        self.existing = existing or {}
        self.fail_match_posts = set(fail_match_posts or [])
        self.calls = []
        self.match_post_calls = 0

    def request(self, method, url, **kwargs):
        table = url.rstrip("/").split("/")[-1]
        self.calls.append((method, table, kwargs))
        if method == "GET" and table == "matches":
            keys_param = kwargs["params"]["canonical_key"]
            keys = keys_param.removeprefix("in.(").removesuffix(")").split(",")
            rows = [
                {
                    "canonical_key": key,
                    "updated_at": self.existing[key][0],
                    "source_updated_at": self.existing[key][1],
                }
                for key in keys
                if key in self.existing
            ]
            return _FakeSupabaseResponse(rows)
        if method == "POST" and table == "matches":
            self.match_post_calls += 1
            if self.match_post_calls in self.fail_match_posts:
                raise requests.exceptions.ReadTimeout("boom")
        return _FakeSupabaseResponse()


def _matches(count):
    return [
        MatchRecord(
            "provider",
            "EPL",
            "2025-2026",
            f"2026-01-{index + 1:02d}T15:00:00+00:00",
            f"home-{index}",
            f"away-{index}",
            source_id=f"source-{index}",
        )
        for index in range(count)
    ]


def test_supabase_snapshot_uploads_matches_in_batches(monkeypatch):
    from src.data_pipeline.supabase_store import SupabaseStore

    monkeypatch.setenv("FOVRA_SUPABASE_BATCH_SIZE", "2")
    monkeypatch.setenv("FOVRA_SUPABASE_BATCH_RETRIES", "0")
    session = _FakeSupabaseSession()
    store = SupabaseStore("https://example.supabase.co", "key", session=session)

    uploaded = store.upsert_snapshot([], [], _matches(5), "provider", "2026-07-31T00:00:00+00:00")

    assert uploaded == 5
    match_posts = [call for call in session.calls if call[0] == "POST" and call[1] == "matches"]
    assert [len(call[2]["json"]) for call in match_posts] == [2, 2, 1]
    assert all(call[2]["timeout"] == 300 for call in session.calls)


def test_supabase_snapshot_resume_skips_prior_batches(monkeypatch):
    from src.data_pipeline.supabase_store import SupabaseStore

    monkeypatch.setenv("FOVRA_SUPABASE_BATCH_SIZE", "2")
    monkeypatch.setenv("FOVRA_SUPABASE_MATCH_BATCH_START", "2")
    session = _FakeSupabaseSession()
    store = SupabaseStore("https://example.supabase.co", "key", session=session)

    uploaded = store.upsert_snapshot([], [], _matches(5), "provider", "2026-07-31T00:00:00+00:00")

    assert uploaded == 3
    match_posts = [call for call in session.calls if call[0] == "POST" and call[1] == "matches"]
    assert [len(call[2]["json"]) for call in match_posts] == [2, 1]


def test_supabase_snapshot_skips_unchanged_rows(monkeypatch):
    from src.data_pipeline.supabase_store import SupabaseStore

    fetched_at = "2026-07-31T00:00:00+00:00"
    rows = _matches(3)
    unchanged_key = rows[1].match_key
    monkeypatch.setenv("FOVRA_SUPABASE_BATCH_SIZE", "3")
    session = _FakeSupabaseSession(existing={unchanged_key: (fetched_at, fetched_at)})
    store = SupabaseStore("https://example.supabase.co", "key", session=session)

    uploaded = store.upsert_snapshot([], [], rows, "provider", fetched_at)

    assert uploaded == 2
    match_posts = [call for call in session.calls if call[0] == "POST" and call[1] == "matches"]
    assert len(match_posts) == 1
    assert [row["canonical_key"] for row in match_posts[0][2]["json"]] == [rows[0].match_key, rows[2].match_key]


def test_supabase_snapshot_retries_failed_batch_then_continues(monkeypatch):
    from src.data_pipeline.supabase_store import SupabaseStore

    monkeypatch.setenv("FOVRA_SUPABASE_BATCH_SIZE", "2")
    monkeypatch.setenv("FOVRA_SUPABASE_BATCH_RETRIES", "1")
    monkeypatch.setattr("src.data_pipeline.supabase_store.time.sleep", lambda _: None)
    session = _FakeSupabaseSession(fail_match_posts={1, 3, 4})
    store = SupabaseStore("https://example.supabase.co", "key", session=session)

    uploaded = store.upsert_snapshot([], [], _matches(5), "provider", "2026-07-31T00:00:00+00:00")

    assert uploaded == 3
    match_posts = [call for call in session.calls if call[0] == "POST" and call[1] == "matches"]
    assert [len(call[2]["json"]) for call in match_posts] == [2, 2, 2, 2, 1]

def test_football_data_404_marks_league_unavailable_and_stops(monkeypatch, tmp_path):
    from src.data_pipeline import data_downloader

    class Response404:
        status_code = 404
        content = b""

    calls = []
    monkeypatch.setattr(data_downloader, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_downloader, "UNAVAILABLE_PATH", tmp_path / "football_data_unavailable.json")
    monkeypatch.setattr(data_downloader, "ALLOWED_LEAGUES", ["EPL"])
    monkeypatch.setattr(data_downloader.time, "sleep", lambda _: None)
    monkeypatch.setattr(data_downloader.requests, "get", lambda url, timeout: calls.append(url) or Response404())

    result = data_downloader.download_all_leagues(2010, 2012)

    assert result["EPL"] == []
    assert len(calls) == 1
    assert data_downloader.is_football_data_unavailable("EPL")


def test_football_data_unavailable_league_skips_without_http(monkeypatch, tmp_path):
    from src.data_pipeline import data_downloader

    monkeypatch.setattr(data_downloader, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_downloader, "UNAVAILABLE_PATH", tmp_path / "football_data_unavailable.json")
    data_downloader.mark_football_data_unavailable("EPL", "https://example.test/E0.csv")
    monkeypatch.setattr(data_downloader.requests, "get", lambda *_, **__: (_ for _ in ()).throw(AssertionError("HTTP should not be called")))

    assert data_downloader.download_season_data("EPL", 2011) is False


def test_fbref_mapping_uses_soccerdata_available_league_ids():
    from src.data_pipeline.league_config import LeagueConfig
    from src.data_pipeline.soccerdata_provider import _resolve_soccerdata_league

    class FakeFBref:
        @staticmethod
        def available_leagues():
            return ["ENG-Premier League", "ESP-La Liga"]

    class FakeSoccerData:
        FBref = FakeFBref

    league = LeagueConfig("EPL", "Premier League", 1, fbref_name="Premier League")

    assert _resolve_soccerdata_league(FakeSoccerData, "FBref", league) == "ENG-Premier League"


def test_fbref_unsupported_league_returns_none():
    from src.data_pipeline.league_config import LeagueConfig
    from src.data_pipeline.soccerdata_provider import _resolve_soccerdata_league

    class FakeFBref:
        @staticmethod
        def available_leagues():
            return ["ENG-Premier League"]

    class FakeSoccerData:
        FBref = FakeFBref

    league = LeagueConfig("MLS", "Major League Soccer", 1, fbref_name="Major League Soccer")

    assert _resolve_soccerdata_league(FakeSoccerData, "FBref", league) is None