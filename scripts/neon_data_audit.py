"""Read-only Neon/PostgreSQL health audit for Fovra.

This script never INSERTs, UPDATEs, DELETEs, or ALTERs production data.
It audits the canonical Neon schema before ML/prediction work. Weather is
intentionally excluded because weather ingestion is currently suspended.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from src.data_pipeline.league_config import load_enabled_leagues
from src.data_pipeline.neon_store import NeonStore


CORE_TABLES = {
    "provider_sources",
    "leagues",
    "teams",
    "matches",
    "provider_records",
    "data_sources",
    "ingestion_runs",
    "team_strength",
    "team_statistics",
    "league_standings",
    "model_versions",
    "predictions",
    "prediction_archive",
}

# match_weather deliberately excluded: weather ingestion is suspended.
STALE_RUNNING_HOURS = float(os.getenv("FOVRA_AUDIT_STALE_RUNNING_HOURS", "2"))


def _rows(store: NeonStore, table: str, *, columns: str = "*", where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return store.select(table, columns=columns, where=where, params=params)


def _count(store: NeonStore, table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
    rows = _rows(store, table, columns="count(*) AS n", where=where, params=params)
    return int(rows[0]["n"])


def check_leagues(store: NeonStore, expected: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_keys = {x.key for x in expected}
    rows = _rows(store, "leagues", columns="canonical_key,name,country,source_provider,source_updated_at,updated_at")
    actual_keys = {str(r["canonical_key"]) for r in rows}
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    issues: list[dict[str, Any]] = []
    if missing:
        issues.append({"severity": "CRITICAL", "check": "league_coverage", "message": "Missing configured leagues", "items": missing})
    if unexpected:
        issues.append({"severity": "WARNING", "check": "league_coverage", "message": "Unexpected league keys in Neon", "items": unexpected})
    return {
        "configured": len(expected_keys),
        "in_neon": len(actual_keys),
        "covered": len(expected_keys & actual_keys),
        "missing": missing,
        "unexpected": unexpected,
        "status": "PASS" if not missing else "FAIL",
    }, issues


def check_match_integrity(store: NeonStore, expected: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    expected_keys = {x.key for x in expected}
    league_rows = _rows(store, "leagues", columns="canonical_key")
    actual_keys = {str(r["canonical_key"]) for r in league_rows}

    total = _count(store, "matches")
    bad_status = _count(store, "matches", "status not in ('scheduled','finished','postponed','cancelled')")
    missing_identity = _count(store, "matches", "league_canonical_key is null or home_team_canonical_key is null or away_team_canonical_key is null")
    finished_missing_score = _count(store, "matches", "status='finished' and (home_score is null or away_score is null)")
    negative_scores = _count(store, "matches", "home_score < 0 or away_score < 0")
    same_team = _count(store, "matches", "home_team_canonical_key = away_team_canonical_key")
    orphan_leagues = _count(store, "matches m left join leagues l on l.canonical_key=m.league_canonical_key", "l.canonical_key is null")
    orphan_home = _count(store, "matches m left join teams t on t.canonical_key=m.home_team_canonical_key", "t.canonical_key is null")
    orphan_away = _count(store, "matches m left join teams t on t.canonical_key=m.away_team_canonical_key", "t.canonical_key is null")
    future_finished = _count(store, "matches", "status='finished' and kickoff_at > %s", (now,))

    duplicate_groups = _rows(
        store,
        "matches",
        columns="league_canonical_key,season,kickoff_at,home_team_canonical_key,away_team_canonical_key,source_provider,count(*) AS n",
        where="1=1 group by league_canonical_key,season,kickoff_at,home_team_canonical_key,away_team_canonical_key,source_provider having count(*) > 1",
    )

    checks = {
        "total": total,
        "finished": _count(store, "matches", "status='finished'"),
        "scheduled": _count(store, "matches", "status='scheduled'"),
        "postponed": _count(store, "matches", "status='postponed'"),
        "cancelled": _count(store, "matches", "status='cancelled'"),
        "bad_status": bad_status,
        "missing_identity": missing_identity,
        "finished_missing_score": finished_missing_score,
        "negative_scores": negative_scores,
        "same_team": same_team,
        "orphan_leagues": orphan_leagues,
        "orphan_home_teams": orphan_home,
        "orphan_away_teams": orphan_away,
        "duplicate_groups": len(duplicate_groups),
        "future_finished": future_finished,
        "expected_leagues_with_matches": len(expected_keys & actual_keys),
    }

    for name, value in checks.items():
        if name in {"total", "finished", "scheduled", "postponed", "cancelled", "expected_leagues_with_matches"}:
            continue
        if value:
            severity = "CRITICAL" if name in {"bad_status", "missing_identity", "finished_missing_score", "negative_scores", "same_team", "orphan_leagues", "orphan_home_teams", "orphan_away_teams", "duplicate_groups"} else "WARNING"
            issues.append({"severity": severity, "check": name, "count": value})
    if future_finished:
        issues.append({"severity": "CRITICAL", "check": "future_finished", "count": future_finished, "message": "Finished matches have kickoff_at in the future"})
    checks["status"] = "PASS" if not any(i["severity"] == "CRITICAL" for i in issues) else "FAIL"
    return checks, issues


def check_teams(store: NeonStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    total = _count(store, "teams")
    blank_names = _count(store, "teams", "name is null or btrim(name) = ''")
    orphan = _count(store, "teams t left join leagues l on l.canonical_key=t.league_canonical_key", "l.canonical_key is null")
    return {"total": total, "blank_names": blank_names, "orphan_leagues": orphan, "status": "PASS" if not (blank_names or orphan) else "FAIL"}, [
        *([{"severity": "CRITICAL", "check": "team_blank_names", "count": blank_names}] if blank_names else []),
        *([{"severity": "CRITICAL", "check": "team_orphan_leagues", "count": orphan}] if orphan else []),
    ]


def check_provider_records(store: NeonStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    total = _count(store, "provider_records")
    missing_provider = _count(store, "provider_records pr left join provider_sources ps on ps.provider_key=pr.provider_key", "ps.provider_key is null")
    orphan_canonical = _count(store, "provider_records pr left join matches m on m.canonical_key=pr.canonical_key", "pr.record_type='match' and m.canonical_key is null")
    result = {"total": total, "missing_provider_source": missing_provider, "orphan_match_records": orphan_canonical, "status": "PASS" if not (missing_provider or orphan_canonical) else "FAIL"}
    if missing_provider:
        issues.append({"severity": "CRITICAL", "check": "provider_records_missing_provider", "count": missing_provider})
    if orphan_canonical:
        issues.append({"severity": "CRITICAL", "check": "provider_records_orphan_match", "count": orphan_canonical})
    return result, issues


def check_strength(store: NeonStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    total = _count(store, "team_strength")
    null_elo = _count(store, "team_strength", "elo is null")
    bad_elo = _count(store, "team_strength", "elo <= 0")
    blank_slug = _count(store, "team_strength", "team_slug is null or btrim(team_slug) = ''")
    duplicate_groups = _rows(store, "team_strength", columns="team_slug,count(*) AS n", where="1=1 group by team_slug having count(*) > 1")
    # team_strength currently has one row per team by schema; duplicates are therefore schema/data violations.
    result = {"rows": total, "null_elo": null_elo, "nonpositive_elo": bad_elo, "blank_slug": blank_slug, "duplicate_team_slugs": len(duplicate_groups), "status": "PASS" if not (null_elo or bad_elo or blank_slug or duplicate_groups) else "FAIL"}
    for key, value in (("strength_null_elo", null_elo), ("strength_nonpositive_elo", bad_elo), ("strength_blank_slug", blank_slug), ("strength_duplicate_team_slug", len(duplicate_groups))):
        if value:
            issues.append({"severity": "CRITICAL", "check": key, "count": value})
    return result, issues


def check_ingestion_runs(store: NeonStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    stale_before = now - timedelta(hours=STALE_RUNNING_HOURS)
    invalid_status = _count(store, "ingestion_runs", "status not in ('running','succeeded','failed','partial')")
    stale_running_rows = _rows(store, "ingestion_runs", columns="id,provider_key,status,started_at,records_seen,records_upserted,error_message", where="status='running' and started_at < %s", params=(stale_before,))
    succeeded_with_error = _count(store, "ingestion_runs", "status='succeeded' and error_message is not null")
    failed_without_error = _count(store, "ingestion_runs", "status='failed' and (error_message is null or btrim(error_message)='')")
    succeeded_zero_after_seen = _count(store, "ingestion_runs", "status='succeeded' and records_seen > 0 and records_upserted=0")
    result = {"invalid_status": invalid_status, "stale_running": len(stale_running_rows), "succeeded_with_error": succeeded_with_error, "failed_without_error": failed_without_error, "succeeded_zero_upsert": succeeded_zero_after_seen, "status": "PASS" if not (invalid_status or stale_running_rows or succeeded_with_error or failed_without_error or succeeded_zero_after_seen) else "FAIL"}
    if invalid_status:
        issues.append({"severity": "CRITICAL", "check": "ingestion_invalid_status", "count": invalid_status})
    if stale_running_rows:
        issues.append({"severity": "CRITICAL", "check": "ingestion_stale_running", "count": len(stale_running_rows), "items": stale_running_rows})
    if succeeded_with_error:
        issues.append({"severity": "CRITICAL", "check": "ingestion_succeeded_with_error", "count": succeeded_with_error})
    if failed_without_error:
        issues.append({"severity": "WARNING", "check": "ingestion_failed_without_error", "count": failed_without_error})
    if succeeded_zero_after_seen:
        issues.append({"severity": "CRITICAL", "check": "ingestion_succeeded_zero_upsert", "count": succeeded_zero_after_seen})
    return result, issues


def check_predictions(store: NeonStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Prediction tables may legitimately be empty before the prediction workflow exists.
    issues: list[dict[str, Any]] = []
    total = _count(store, "predictions")
    bad_probability = _count(store, "predictions", "home_probability < 0 or home_probability > 1 or draw_probability < 0 or draw_probability > 1 or away_probability < 0 or away_probability > 1")
    bad_sum = _count(store, "predictions", "abs((coalesce(home_probability,0)+coalesce(draw_probability,0)+coalesce(away_probability,0))-1) > 0.01")
    result = {"rows": total, "bad_probability_rows": bad_probability, "probability_sum_outside_tolerance": bad_sum, "status": "PASS" if not (bad_probability or bad_sum) else "FAIL", "note": "Empty is allowed before prediction workflow is implemented."}
    if bad_probability:
        issues.append({"severity": "CRITICAL", "check": "prediction_probability_range", "count": bad_probability})
    if bad_sum:
        issues.append({"severity": "CRITICAL", "check": "prediction_probability_sum", "count": bad_sum})
    return result, issues


def check_table_presence(store: NeonStore) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _rows(store, "information_schema.tables", columns="table_name", where="table_schema='public'")
    present = {str(r["table_name"]) for r in rows}
    missing = sorted(CORE_TABLES - present)
    result = {"expected": sorted(CORE_TABLES), "missing": missing, "status": "PASS" if not missing else "FAIL", "weather_excluded": True}
    issues = [{"severity": "CRITICAL", "check": "table_presence", "message": "Required table missing", "items": missing}] if missing else []
    return result, issues


def run_audit() -> dict[str, Any]:
    expected = load_enabled_leagues()
    store = NeonStore()
    store.verify_connection()
    checks: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []

    for name, fn, args in [
        ("table_presence", check_table_presence, (store,)),
        ("leagues", check_leagues, (store, expected)),
        ("matches", check_match_integrity, (store, expected)),
        ("teams", check_teams, (store,)),
        ("provider_records", check_provider_records, (store,)),
        ("team_strength", check_strength, (store,)),
        ("ingestion_runs", check_ingestion_runs, (store,)),
        ("predictions", check_predictions, (store,)),
    ]:
        result, found = fn(*args)
        checks[name] = result
        issues.extend(found)

    critical = sum(1 for i in issues if i["severity"] == "CRITICAL")
    warnings = sum(1 for i in issues if i["severity"] == "WARNING")
    return {
        "audit_version": "1.0",
        "database": "Neon PostgreSQL",
        "weather_audit": "excluded_suspended",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "critical_errors": critical,
        "warnings": warnings,
        "data_ready_for_ml": critical == 0,
        "issues": issues,
    }


def main() -> int:
    report = run_audit()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["data_ready_for_ml"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
