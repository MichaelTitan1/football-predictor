from __future__ import annotations

import pytest

from src.data_pipeline.football_data_provider import FootballDataProvider


@pytest.mark.network

def test_real_football_data_provider_downloads_and_normalizes_current_data():
    provider = FootballDataProvider(include_remote=True, raw_dir="data/raw")
    snapshot = provider.fetch()
    assert snapshot.provider == "football-data.co.uk"
    assert snapshot.fetched_at
    assert snapshot.matches, "Football-Data returned no canonical matches"
    assert any(match.status == "finished" for match in snapshot.matches)
    assert all(match.home_team and match.away_team for match in snapshot.matches)
