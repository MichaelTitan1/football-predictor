from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_downloader import (
    ALLOWED_LEAGUES,
    LEAGUE_CONFIG,
    RAW_DIR,
    START_YEAR,
    _load_source_state,
    _save_source_state,
    current_season_start,
    download_season_data,
)
from .football_data_provider import FootballDataProvider
from .neon_store import NeonStore

logger = logging.getLogger(__name__)
SOURCE_FORMAT_VERSION = 2


def _raw_sources_need_refresh() -> bool:
    state = _load_source_state()
    return state.get("_metadata", {}).get("raw_format_version") != SOURCE_FORMAT_VERSION


def _mark_raw_format_current() -> None:
    state = _load_source_state()
    state.setdefault("_metadata", {})["raw_format_version"] = SOURCE_FORMAT_VERSION
    state["_metadata"]["raw_format_checked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save_source_state(state)


def refresh_sources() -> dict[str, Any]:
    """Refresh Football-Data without changing provider dates before parsing.

    Season leagues use one source file per season. Extra leagues use their
    all-seasons /new feed. During the one-time format migration, all historical
    season files are refreshed so previously normalized Date values are replaced
    by the exact provider CSV values. A failed download never deletes a good
    local file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    current = current_season_start()
    migration = _raw_sources_need_refresh()
    refreshed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for league in ALLOWED_LEAGUES:
        source_type = LEAGUE_CONFIG[league].get("source_type")
        years = [START_YEAR] if source_type == "single" else (
            list(range(START_YEAR, current + 1)) if migration else [current]
        )
        for year in years:
            try:
                changed = download_season_data(
                    league,
                    year,
                    force_refresh=True,
                    ignore_unavailable=True,
                )
                key = f"{league}:new" if source_type == "single" else f"{league}:{year}"
                if changed:
                    refreshed.append(key)
                else:
                    skipped.append(key)
            except Exception as exc:
                failed.append(f"{league}:{year}")
                logger.error("Football-Data source refresh failed for %s %s: %s", league, year, exc)

    # Never claim the migration is complete if any source could not be refreshed.
    if not failed:
        _mark_raw_format_current()
    else:
        logger.warning("Raw-source migration remains pending because %d sources failed", len(failed))

    return {
        "source_format_version": SOURCE_FORMAT_VERSION,
        "migration": migration,
        "refreshed": len(refreshed),
        "skipped": len(skipped),
        "failed": failed,
        "current_season": f"{current}-{current + 1}",
    }


def run() -> dict[str, Any]:
    refresh = refresh_sources()
    provider = FootballDataProvider(raw_dir=RAW_DIR, include_remote=False)
    snapshot = provider.fetch()

    expected = set(ALLOWED_LEAGUES)
    actual = {m.league_key for m in snapshot.matches}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError("Local Football-Data snapshot is missing configured leagues: " + ", ".join(missing))

    store = NeonStore()
    store.initialize_schema()
    run_id = store.record_ingestion_start(snapshot.provider)
    try:
        matches = list(snapshot.matches)
        newest = max((m.kickoff_utc for m in matches), default=None)
        upserted = store.upsert_snapshot(
            snapshot.leagues,
            snapshot.teams,
            matches,
            snapshot.provider,
            snapshot.fetched_at,
        )
        store.record_ingestion_finish(
            run_id,
            status="succeeded",
            records_seen=len(matches),
            records_upserted=upserted,
            newest_match_at=newest,
        )
        return {
            "refresh": refresh,
            "leagues": len(snapshot.leagues),
            "teams": len(snapshot.teams),
            "matches": len(matches),
            "upserted": upserted,
        }
    except Exception as exc:
        store.record_ingestion_finish(
            run_id,
            status="failed",
            records_seen=len(snapshot.matches),
            records_upserted=0,
            error_message=str(exc),
        )
        raise


def main() -> None:
    logging.basicConfig(level=os.getenv("FOVRA_LOG_LEVEL", "INFO"))
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
