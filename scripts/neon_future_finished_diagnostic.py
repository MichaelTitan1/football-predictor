"""Read-only diagnostic for matches marked finished after their kickoff time.

This script NEVER updates, inserts, deletes, or otherwise mutates Neon.
It produces a compact stdout summary plus a full JSON report so a large
set of affected rows (for example 800+) does not need to be pasted into chat.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_pipeline.neon_store import NeonStore

DEFAULT_OUTPUT = Path("neon_future_finished_diagnostic.json")
PAGE_SIZE = 500


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _fetch_all(store: NeonStore, now: datetime) -> list[dict[str, Any]]:
    """Fetch the contradiction set using the actual canonical matches schema.

    NeonStore.select() does not support the filter/order parameter mapping used
    by an earlier version of this diagnostic, and matches has no provider_key
    column. Use the same hard-coded SELECT style as neon_data_audit.py instead.
    This remains strictly read-only.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    sql = """
        select
            canonical_key,
            league_canonical_key,
            season,
            kickoff_at,
            home_team_canonical_key,
            away_team_canonical_key,
            status,
            home_score,
            away_score,
            source_provider
        from matches
        where status = 'finished'
          and kickoff_at > %s
        order by kickoff_at asc, canonical_key asc
        limit %s offset %s
    """

    while True:
        page = store._fetchall(sql, (now, PAGE_SIZE, offset))
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _compact_summary(rows: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    by_league = Counter(str(row.get("league_canonical_key") or "") for row in rows)
    by_provider = Counter(str(row.get("source_provider") or "") for row in rows)
    by_season = Counter(str(row.get("season") or "") for row in rows)
    with_scores = sum(
        row.get("home_score") is not None and row.get("away_score") is not None
        for row in rows
    )
    without_scores = len(rows) - with_scores

    future_days: list[float] = []
    for row in rows:
        kickoff = _parse_dt(row.get("kickoff_at"))
        if kickoff:
            future_days.append((kickoff - now).total_seconds() / 86400)

    return {
        "count": len(rows),
        "audit_now_utc": now.isoformat(),
        "score_presence": {
            "with_both_scores": with_scores,
            "without_both_scores": without_scores,
        },
        "future_days": {
            "min": min(future_days) if future_days else None,
            "max": max(future_days) if future_days else None,
            "avg": (sum(future_days) / len(future_days)) if future_days else None,
        },
        "by_provider": dict(sorted(by_provider.items())),
        "by_league": dict(sorted(by_league.items())),
        "by_season": dict(sorted(by_season.items())),
        "sample_first_25": rows[:25],
    }


def run(output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    # DATABASE_URL is consumed by NeonStore. No write-capable method is called.
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is required")

    now = datetime.now(timezone.utc)
    store = NeonStore()
    rows = _fetch_all(store, now)
    summary = _compact_summary(rows, now)

    report = {
        "diagnostic_version": "1.1",
        "database": "Neon PostgreSQL",
        "read_only": True,
        "query": "status = finished AND kickoff_at > audit_now_utc",
        "generated_at": now.isoformat(),
        "summary": summary,
        "affected_matches": rows,
    }
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "diagnostic_version": report["diagnostic_version"],
                "read_only": True,
                "affected_count": len(rows),
                "report_path": str(output_path),
                "provider_breakdown": summary["by_provider"],
                "league_count": len(summary["by_league"]),
                "season_breakdown": summary["by_season"],
                "score_presence": summary["score_presence"],
                "future_days": summary["future_days"],
                "sample_first_25": summary["sample_first_25"],
            },
            indent=2,
            default=str,
        )
    )
    return report


if __name__ == "__main__":
    output = Path(os.getenv("FOVRA_FUTURE_FINISHED_REPORT", str(DEFAULT_OUTPUT)))
    run(output)
