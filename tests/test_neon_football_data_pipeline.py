from __future__ import annotations

import pytest

from src.data_pipeline import data_downloader
from src.data_pipeline.canonical_data import LeagueRecord, MatchRecord, TeamRecord
from src.data_pipeline.league_config import load_enabled_leagues
from src.data_pipeline.neon_store import NeonStore

SEASON = set("E0 E1 E2 E3 EC SC0 SC1 SC2 SC3 D1 D2 I1 I2 SP1 SP2 F1 F2 N1 B1 P1 T1 G1".split())
SINGLE = set("ARG AUT BRA CHN DNK FIN IRL JPN MEX NOR POL ROU RUS SWE SWZ USA".split())


def test_authoritative_38_league_config():
    leagues = load_enabled_leagues()
    codes = {l.football_data_code for l in leagues}
    by_type = {l.football_data_code: l.football_data_source_type for l in leagues}
    assert len(leagues) == 38
    assert codes == SEASON | SINGLE
    assert {c for c, t in by_type.items() if t == "season"} == SEASON
    assert {c for c, t in by_type.items() if t == "single"} == SINGLE


def test_football_data_url_strategy_for_22_and_16():
    assert data_downloader._candidate_url("E0", 2025).endswith("/mmz4281/2526/E0.csv")
    assert data_downloader._candidate_url("SWE", 2025) == "https://www.football-data.co.uk/new/SWE.csv"
    assert data_downloader._candidate_url("USA", 2025) == "https://www.football-data.co.uk/new/USA.csv"
    assert all("/mmz4281/2526/" in data_downloader._candidate_url(c, 2025) for c in SEASON)
    assert all(data_downloader._candidate_url(c, 2025) == f"https://www.football-data.co.uk/new/{c}.csv" for c in SINGLE)


def test_permanent_404_not_retried_and_persisted(monkeypatch, tmp_path):
    unavailable = tmp_path / "unavailable.json"
    monkeypatch.setattr(data_downloader, "UNAVAILABLE_PATH", unavailable)
    calls = []
    monkeypatch.setattr(data_downloader, "_http_get", lambda url: (calls.append(url) or (None, 404)))
    with pytest.raises(data_downloader.FootballDataUnavailableError):
        data_downloader.download_season_data("E0", 2010, force_refresh=True)
    assert len(calls) == 1
    calls.clear()
    assert data_downloader.download_season_data("E0", 2010, force_refresh=True) is False
    assert calls == []


def test_single_file_leagues_never_attempt_season_urls(monkeypatch, tmp_path):
    monkeypatch.setattr(data_downloader, "UNAVAILABLE_PATH", tmp_path / "unavailable.json")
    seen = []
    monkeypatch.setattr(data_downloader, "_http_get", lambda url: (seen.append(url) or (None, 404)))
    with pytest.raises(data_downloader.FootballDataUnavailableError):
        data_downloader.download_season_data("USA", 2010, force_refresh=True)
    assert seen == ["https://www.football-data.co.uk/new/USA.csv"]
    assert "mmz4281" not in seen[0]


def test_future_season_range_is_computed_from_date():
    from datetime import datetime, timezone
    assert data_downloader.current_season_start(datetime(2026, 8, 4, tzinfo=timezone.utc)) == 2026
    assert data_downloader.START_YEAR == 2010


class FakeNeonStore(NeonStore):
    def __init__(self):
        self.batch_size = 2
        self.batch_start = 1
        self.batch_retries = 1
        self.timeout = 30
        self.records = {}
        self.matches = {}
        self.match_batches = []
        self.fail_once = False
        self.failed = False
    def initialize_schema(self): pass
    def _fetchall(self, sql, params=()):
        if "provider_records" in sql:
            ids = set(params[2:])
            return [{"source_id": k[2], "payload": v["payload"]} for k, v in self.records.items() if k[0] == params[0] and k[1] == params[1] and k[2] in ids]
        return []
    def upsert(self, table, rows, on_conflict):
        if table == "matches":
            self.match_batches.append([r["canonical_key"] for r in rows])
            if self.fail_once and not self.failed:
                self.failed = True
                raise RuntimeError("temporary")
            for r in rows: self.matches[r["canonical_key"]] = r
        if table == "provider_records":
            for r in rows: self.records[(r["provider_key"], r["record_type"], r["source_id"])] = r


def _match(i: int, score: int = 1) -> MatchRecord:
    return MatchRecord("football-data.co.uk", "E0", "2025-2026", f"2025-08-0{i+1}T12:00:00+00:00", f"home{i}", f"away{i}", "finished", score, 0, f"src{i}")


def test_neon_batch_idempotency_insert_skip_update_and_retry():
    store = FakeNeonStore()
    leagues = [LeagueRecord("E0", "E0")]
    teams = [TeamRecord("home0", "Home 0", "E0"), TeamRecord("away0", "Away 0", "E0"), TeamRecord("home1", "Home 1", "E0"), TeamRecord("away1", "Away 1", "E0"), TeamRecord("home2", "Home 2", "E0"), TeamRecord("away2", "Away 2", "E0")]
    assert store.upsert_snapshot(leagues, teams, [_match(0), _match(1), _match(2)], "football-data.co.uk", "2026-08-04T00:00:00+00:00") == 3
    assert len(store.matches) == 3
    assert store.upsert_snapshot(leagues, teams, [_match(0), _match(1), _match(2)], "football-data.co.uk", "2026-08-04T00:00:00+00:00") == 0
    assert len(store.matches) == 3
    assert store.upsert_snapshot(leagues, teams, [_match(0, 2), _match(1), _match(2), _match(3)], "football-data.co.uk", "2026-08-04T00:00:00+00:00") == 2
    assert len(store.matches) == 4
    store.fail_once = True
    assert store.upsert_snapshot(leagues, teams, [_match(4), _match(5), _match(6)], "football-data.co.uk", "2026-08-04T00:00:00+00:00") == 3
    assert store.failed is True


def test_all_16_fixed_file_urls_are_exact():
    expected = {
        "ARG": "https://www.football-data.co.uk/new/ARG.csv",
        "AUT": "https://www.football-data.co.uk/new/AUT.csv",
        "BRA": "https://www.football-data.co.uk/new/BRA.csv",
        "CHN": "https://www.football-data.co.uk/new/CHN.csv",
        "DNK": "https://www.football-data.co.uk/new/DNK.csv",
        "FIN": "https://www.football-data.co.uk/new/FIN.csv",
        "IRL": "https://www.football-data.co.uk/new/IRL.csv",
        "JPN": "https://www.football-data.co.uk/new/JPN.csv",
        "MEX": "https://www.football-data.co.uk/new/MEX.csv",
        "NOR": "https://www.football-data.co.uk/new/NOR.csv",
        "POL": "https://www.football-data.co.uk/new/POL.csv",
        "ROU": "https://www.football-data.co.uk/new/ROU.csv",
        "RUS": "https://www.football-data.co.uk/new/RUS.csv",
        "SWE": "https://www.football-data.co.uk/new/SWE.csv",
        "SWZ": "https://www.football-data.co.uk/new/SWZ.csv",
        "USA": "https://www.football-data.co.uk/new/USA.csv",
    }
    assert {code: data_downloader._candidate_url(code, 2010) for code in SINGLE} == expected
    assert all("/mmz4281/" not in url for url in expected.values())


def test_fixed_file_leagues_download_once_and_parse_all_rows(monkeypatch, tmp_path):
    csv = b"Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/2024,Alpha,Beta,1,0,H\n02/01/2025,Gamma,Delta,2,2,D\n"
    calls = []
    monkeypatch.setattr(data_downloader, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data_downloader, "UNAVAILABLE_PATH", tmp_path / "unavailable.json")
    monkeypatch.setattr(data_downloader, "SOURCE_STATE_PATH", tmp_path / "source_state.json")
    monkeypatch.setattr(data_downloader, "_http_get", lambda url: (calls.append(url) or (csv, 200)))
    assert data_downloader.download_season_data("ARG", data_downloader.START_YEAR, force_refresh=True) is True
    assert calls == ["https://www.football-data.co.uk/new/ARG.csv"]
    provider = __import__("src.data_pipeline.football_data_provider", fromlist=["FootballDataProvider"]).FootballDataProvider(raw_dir=tmp_path, include_remote=False)
    snapshot = provider.fetch()
    assert len(snapshot.matches) == 2
    assert {m.league_key for m in snapshot.matches} == {"ARG"}
    calls.clear()
    assert data_downloader.download_season_data("ARG", data_downloader.START_YEAR, force_refresh=True) is False
    assert calls == ["https://www.football-data.co.uk/new/ARG.csv"]


def test_neon_dsn_adds_ssl_and_ipv4_hostaddr(monkeypatch):
    from src.data_pipeline.neon_store import _safe_dsn_for_connect
    monkeypatch.setattr("src.data_pipeline.neon_store.socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("203.0.113.10", 5432))])
    dsn = _safe_dsn_for_connect("postgresql://user:secret@example.neon.tech/db")
    assert "sslmode=require" in dsn
    assert "hostaddr=203.0.113.10" in dsn
    assert "secret" in dsn


class VerifyCursor:
    description = [("?column?",)]
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, sql, params=()): self.sql = sql
    def fetchone(self): return (1,)


class VerifyConnection:
    def cursor(self): return VerifyCursor()
    def commit(self): self.committed = True


def test_neon_select_1_verification_uses_parameterless_probe():
    store = NeonStore(connection=VerifyConnection())
    store.verify_connection()
    assert store.connection.committed is True


def test_match_key_is_provider_independent_for_cross_source_deduplication():
    base = MatchRecord(
        "football-data.co.uk",
        "E0",
        "2025-2026",
        "2025-08-09T15:00:00+00:00",
        "arsenal",
        "chelsea",
        "finished",
        2,
        1,
        "source-a",
    )
    same_match_other_provider = MatchRecord(
        "another-provider",
        "E0",
        "2025-2026",
        "2025-08-09T16:00:00+01:00",
        "arsenal",
        "chelsea",
        "finished",
        2,
        1,
        "source-b",
    )
    assert base.match_key == same_match_other_provider.match_key


def test_match_key_keeps_distinct_same_day_opponents_separate():
    first = MatchRecord("football-data.co.uk", "E0", "2025-2026", "2025-08-09T12:00:00+00:00", "arsenal", "chelsea")
    second = MatchRecord("another-provider", "E0", "2025-2026", "2025-08-09T18:00:00+00:00", "arsenal", "liverpool")
    assert first.match_key != second.match_key
