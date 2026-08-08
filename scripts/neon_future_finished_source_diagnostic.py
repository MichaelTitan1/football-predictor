"""Read-only comparison of future-finished Neon matches against Football-Data source rows.

This diagnostic never writes to Neon or to the source provider. It downloads each
relevant Football-Data CSV once, then compares the affected Neon rows against the
raw provider fields used by Fovra's canonicalizer.
"""
from __future__ import annotations

import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.data_pipeline.data_downloader import LEAGUE_CONFIG
from src.data_pipeline.neon_store import NeonStore

PROVIDER = "football-data.co.uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
NEW_URL = "https://www.football-data.co.uk/new"
OUT = Path("neon_future_finished_source_diagnostic.json")
TIMEOUT = 30


def _season_code(season: str) -> str:
    start = int(season.split("-", 1)[0])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def _source_url(league_key: str, season: str) -> str:
    info = LEAGUE_CONFIG[league_key]
    if info.get("source_type") == "single":
        return f"{NEW_URL}/{info['code']}.csv"
    return f"{BASE_URL}/{_season_code(season)}/{info['code']}.csv"


def _norm_team(value: object) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")


def _source_datetime(value: object, time_value: object) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    t = str(time_value or "").strip()
    combined = f"{text} {t}".strip()
    parsed = pd.to_datetime(combined, dayfirst=True, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _load_csv(session: requests.Session, url: str) -> pd.DataFrame:
    response = session.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content))
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    return frame


def _fetch_affected(store: NeonStore) -> list[dict]:
    return store._fetchall(
        """select canonical_key, league_canonical_key, season, kickoff_at,
                  home_team_canonical_key, away_team_canonical_key, status,
                  home_score, away_score, source_provider, source_match_id,
                  source_updated_at
           from matches
          where status='finished' and kickoff_at > %s
          order by kickoff_at, league_canonical_key, canonical_key""",
        (datetime.now(timezone.utc),),
    )


def _find_source_row(frame: pd.DataFrame, neon: dict) -> tuple[dict | None, str]:
    required = {"HomeTeam", "AwayTeam", "Date"}
    if not required.issubset(frame.columns):
        return None, "missing_required_columns"

    work = frame.copy()
    work["_home"] = work["HomeTeam"].map(_norm_team)
    work["_away"] = work["AwayTeam"].map(_norm_team)
    work["_dt"] = [
        _source_datetime(d, t if "Time" in work.columns else "")
        for d, t in zip(work["Date"], work["Time"] if "Time" in work.columns else [""] * len(work))
    ]
    kickoff = pd.Timestamp(neon["kickoff_at"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    else:
        kickoff = kickoff.tz_convert("UTC")

    home = str(neon["home_team_canonical_key"]).split(":", 1)[-1]
    away = str(neon["away_team_canonical_key"]).split(":", 1)[-1]
    candidates = work[(work["_home"] == home) & (work["_away"] == away)].copy()
    if candidates.empty:
        return None, "team_pair_not_found"
    candidates["_delta_seconds"] = candidates["_dt"].map(
        lambda x: abs((x - kickoff).total_seconds()) if pd.notna(x) else 10**12
    )
    row = candidates.sort_values("_delta_seconds").iloc[0]
    if row["_delta_seconds"] > 48 * 3600:
        return None, "team_pair_found_but_date_mismatch"
    return {str(k): (None if pd.isna(v) else v) for k, v in row.items()}, "matched"


def run() -> dict:
    store = NeonStore()
    store.verify_connection()
    affected = _fetch_affected(store)
    session = requests.Session()
    session.headers.update({"User-Agent": "Fovra/1.0 read-only diagnostic"})

    cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    source_counts = Counter()
    comparison_counts = Counter()
    provider_status_counts = Counter()

    for neon in affected:
        league = str(neon["league_canonical_key"])
        season = str(neon["season"])
        url = _source_url(league, season)
        try:
            if url not in cache:
                cache[url] = _load_csv(session, url)
            source, match_status = _find_source_row(cache[url], neon)
            comparison_counts[match_status] += 1
            if source is None:
                rows.append({
                    "canonical_key": neon["canonical_key"],
                    "league_canonical_key": league,
                    "season": season,
                    "kickoff_at": str(neon["kickoff_at"]),
                    "source_url": url,
                    "comparison_status": match_status,
                    "neon": neon,
                })
                continue

            raw_fthg = source.get("FTHG")
            raw_ftag = source.get("FTAG")
            raw_ftr = source.get("FTR")
            raw_finished = pd.notna(pd.to_numeric(pd.Series([raw_fthg]), errors="coerce").iloc[0]) and False
            fthg_num = pd.to_numeric(pd.Series([raw_fthg]), errors="coerce").iloc[0]
            ftag_num = pd.to_numeric(pd.Series([raw_ftag]), errors="coerce").iloc[0]
            raw_finished = bool(pd.notna(fthg_num) and pd.notna(ftag_num) and str(raw_ftr).strip() in {"H", "D", "A"})
            scores_match = (
                (None if pd.isna(fthg_num) else int(fthg_num)) == neon["home_score"]
                and (None if pd.isna(ftag_num) else int(ftag_num)) == neon["away_score"]
            )
            neon_finished_from_source = raw_finished
            if raw_finished:
                provider_status_counts["raw_has_finished_result"] += 1
            else:
                provider_status_counts["raw_does_not_have_finished_result"] += 1
            if scores_match:
                provider_status_counts["scores_match"] += 1
            else:
                provider_status_counts["scores_mismatch"] += 1

            rows.append({
                "canonical_key": neon["canonical_key"],
                "league_canonical_key": league,
                "season": season,
                "kickoff_at": str(neon["kickoff_at"]),
                "source_url": url,
                "comparison_status": "matched",
                "raw": {
                    "Date": source.get("Date"),
                    "Time": source.get("Time"),
                    "HomeTeam": source.get("HomeTeam"),
                    "AwayTeam": source.get("AwayTeam"),
                    "FTHG": source.get("FTHG"),
                    "FTAG": source.get("FTAG"),
                    "FTR": source.get("FTR"),
                    "MatchID": source.get("MatchID"),
                    "raw_finished_by_current_parser_rule": neon_finished_from_source,
                },
                "neon": {
                    "home_score": neon["home_score"],
                    "away_score": neon["away_score"],
                    "status": neon["status"],
                    "source_match_id": neon["source_match_id"],
                    "source_updated_at": str(neon["source_updated_at"]),
                },
                "scores_match": scores_match,
            })
        except Exception as exc:
            comparison_counts["source_fetch_or_parse_error"] += 1
            rows.append({
                "canonical_key": neon["canonical_key"],
                "league_canonical_key": league,
                "season": season,
                "kickoff_at": str(neon["kickoff_at"]),
                "source_url": url,
                "comparison_status": "source_fetch_or_parse_error",
                "error": str(exc),
            })

    source_counts.update(str(r["league_canonical_key"]) for r in affected)
    matched = [r for r in rows if r.get("comparison_status") == "matched"]
    report = {
        "diagnostic_version": "1.0",
        "read_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "affected_count": len(affected),
        "source_urls_loaded": len(cache),
        "league_count": len(source_counts),
        "league_breakdown": dict(sorted(source_counts.items())),
        "comparison_breakdown": dict(sorted(comparison_counts.items())),
        "provider_source_breakdown": dict(sorted(provider_status_counts.items())),
        "matched_rows_with_score_agreement": sum(1 for r in matched if r.get("scores_match")),
        "matched_rows_with_score_disagreement": sum(1 for r in matched if not r.get("scores_match")),
        "raw_source_has_finished_result": sum(1 for r in matched if r.get("raw", {}).get("raw_finished_by_current_parser_rule")),
        "raw_source_has_no_finished_result": sum(1 for r in matched if not r.get("raw", {}).get("raw_finished_by_current_parser_rule")),
        "conclusion": (
            "SOURCE_DATA_HAS_FINISHED_RESULTS" if provider_status_counts["raw_has_finished_result"] == len(matched) and matched
            else "SOURCE_DATA_DOES_NOT_HAVE_FINISHED_RESULTS"
            if provider_status_counts["raw_does_not_have_finished_result"] == len(matched) and matched
            else "MIXED_OR_INCOMPLETE_SOURCE_EVIDENCE"
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, default=str))
    print(f"Full report: {OUT}")
    return report


if __name__ == "__main__":
    run()
