"""Read-only diagnostic for future-finished matches using stored provider_records."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_pipeline.neon_store import NeonStore

OUTPUT = Path(os.getenv("FOVRA_PROVIDER_RECORD_DIAGNOSTIC_OUTPUT", "neon_future_finished_provider_record_diagnostic.json"))
PAGE_SIZE = 500


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _payload_value(payload: Any, *keys: str) -> Any:
    payload = _json_value(payload)
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return str(a).strip() == str(b).strip()


def _fetch_page(store: NeonStore, now: datetime, after_key: str | None) -> list[dict[str, Any]]:
    where = "m.status = %s and m.kickoff_at > %s"
    params: list[Any] = ["finished", now]
    if after_key is not None:
        where += " and m.canonical_key > %s"
        params.append(after_key)
    where += " and pr.provider_key = %s and pr.record_type = %s"
    params.extend(["football-data.co.uk", "match"])
    sql = f"""
        select
            m.canonical_key,
            m.league_canonical_key,
            m.season,
            m.kickoff_at,
            m.status,
            m.home_score,
            m.away_score,
            m.source_provider,
            m.source_match_id,
            m.source_updated_at,
            pr.provider_key,
            pr.record_type,
            pr.source_id,
            pr.source_updated_at as provider_source_updated_at,
            pr.payload
        from matches m
        left join provider_records pr
          on pr.canonical_key = m.canonical_key
         and pr.provider_key = %s
         and pr.record_type = %s
        where {where.replace('pr.provider_key = %s and pr.record_type = %s', 'true')}
        order by m.canonical_key
        limit %s
    """
    # The provider predicates are already fixed in the JOIN; keep the parameter list
    # explicit and deterministic for psycopg.
    query_params: list[Any] = ["football-data.co.uk", "match", *params[:-2], PAGE_SIZE]
    return store._fetchall(sql, query_params)


def run(output: Path = OUTPUT) -> dict[str, Any]:
    store = NeonStore()
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    after_key: str | None = None

    while True:
        page = _fetch_page(store, now, after_key)
        if not page:
            break
        rows.extend(page)
        next_key = str(page[-1]["canonical_key"])
        if next_key == after_key or len(page) < PAGE_SIZE:
            break
        after_key = next_key

    provider_found = 0
    payload_found = 0
    source_id_agreement = 0
    kickoff_agreement = 0
    score_agreement = 0
    score_disagreement = 0
    status_agreement = 0
    raw_score_present = 0
    raw_result_present = 0
    raw_finished_signal = 0
    raw_no_finished_signal = 0
    breakdown = Counter()
    examples: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for row in rows:
        payload = _json_value(row.get("payload"))
        has_record = row.get("provider_key") is not None
        has_payload = isinstance(payload, dict)
        if has_record:
            provider_found += 1
        if has_payload:
            payload_found += 1

        raw_home = _payload_value(payload, "home_score", "home_goals", "FTHG")
        raw_away = _payload_value(payload, "away_score", "away_goals", "FTAG")
        raw_result = _payload_value(payload, "result", "FTR", "status")
        raw_kickoff = _payload_value(payload, "kickoff_at", "kickoff_utc", "date")
        raw_source_id = _payload_value(payload, "source_match_id", "source_id", "MatchID")
        raw_finished = raw_home is not None and raw_away is not None and _norm(raw_result) in {"H", "D", "A", "finished"}

        if raw_home is not None and raw_away is not None:
            raw_score_present += 1
        if raw_result is not None:
            raw_result_present += 1
        if raw_finished:
            raw_finished_signal += 1
        else:
            raw_no_finished_signal += 1

        if _same(raw_source_id, row.get("source_match_id")) and raw_source_id is not None:
            source_id_agreement += 1
        if _same(raw_kickoff, row.get("kickoff_at")) and raw_kickoff is not None:
            kickoff_agreement += 1
        if raw_home is not None and raw_away is not None:
            if _same(raw_home, row.get("home_score")) and _same(raw_away, row.get("away_score")):
                score_agreement += 1
            else:
                score_disagreement += 1
        if _norm(raw_result) in {"finished", "H", "D", "A"}:
            status_agreement += 1

        if not has_record:
            category = "provider_record_missing"
        elif not has_payload:
            category = "provider_payload_missing_or_invalid"
        elif raw_finished:
            category = "stored_payload_contains_finished_result"
        elif raw_home is not None or raw_away is not None or raw_result is not None:
            category = "stored_payload_has_partial_result_data"
        else:
            category = "stored_payload_has_no_finished_result_signal"
        breakdown[category] += 1

        detail = {
            "canonical_key": row.get("canonical_key"),
            "league_canonical_key": row.get("league_canonical_key"),
            "season": row.get("season"),
            "neon_kickoff_at": str(row.get("kickoff_at")) if row.get("kickoff_at") is not None else None,
            "neon_status": row.get("status"),
            "neon_home_score": row.get("home_score"),
            "neon_away_score": row.get("away_score"),
            "neon_source_match_id": row.get("source_match_id"),
            "provider_record_exists": has_record,
            "provider_payload_exists": has_payload,
            "provider_source_id": row.get("source_id"),
            "provider_payload_kickoff": raw_kickoff,
            "provider_payload_home_score": raw_home,
            "provider_payload_away_score": raw_away,
            "provider_payload_result": raw_result,
            "source_id_agrees": _same(raw_source_id, row.get("source_match_id")) if raw_source_id is not None else None,
            "kickoff_agrees": _same(raw_kickoff, row.get("kickoff_at")) if raw_kickoff is not None else None,
            "scores_agree": (_same(raw_home, row.get("home_score")) and _same(raw_away, row.get("away_score"))) if raw_home is not None and raw_away is not None else None,
            "category": category,
        }
        details.append(detail)
        if len(examples) < 25:
            examples.append(detail)

    report = {
        "diagnostic_version": "1.0",
        "read_only": True,
        "generated_at": now.isoformat(),
        "affected_count": len(rows),
        "provider": "football-data.co.uk",
        "provider_records_found": provider_found,
        "provider_payloads_found": payload_found,
        "source_id_agreement": source_id_agreement,
        "kickoff_agreement": kickoff_agreement,
        "score_agreement": score_agreement,
        "score_disagreement": score_disagreement,
        "raw_score_present": raw_score_present,
        "raw_result_present": raw_result_present,
        "raw_finished_signal": raw_finished_signal,
        "raw_no_finished_signal": raw_no_finished_signal,
        "breakdown": dict(breakdown),
        "sample_first_25": examples,
        "details": details,
    }
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2, default=str))
    print(f"Full report: {output}")
    return report


if __name__ == "__main__":
    run()
