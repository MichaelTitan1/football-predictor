"""Read-only Neon/PostgreSQL health audit for Fovra.

The audit never writes to Neon. It checks the canonical data model before ML.
Weather is deliberately excluded because weather ingestion is suspended.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from src.data_pipeline.league_config import load_enabled_leagues
from src.data_pipeline.neon_store import NeonStore

CORE_TABLES = {
    "provider_sources", "leagues", "teams", "matches", "provider_records",
    "data_sources", "ingestion_runs", "team_strength", "team_statistics",
    "league_standings", "model_versions", "predictions", "prediction_archive",
}
STALE_RUNNING_HOURS = float(os.getenv("FOVRA_AUDIT_STALE_RUNNING_HOURS", "2"))


def sql_rows(store: NeonStore, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Execute only hard-coded SELECT statements; this module never writes."""
    return store._fetchall(sql, params)


def count(store: NeonStore, table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
    sql = f"select count(*) as n from {table}"
    if where:
        sql += f" where {where}"
    return int(sql_rows(store, sql, params)[0]["n"])


def check_tables(store: NeonStore):
    present = {r["table_name"] for r in sql_rows(store, "select table_name from information_schema.tables where table_schema='public'")}
    missing = sorted(CORE_TABLES - present)
    return {"expected": sorted(CORE_TABLES), "missing": missing, "weather_excluded": True, "status": "PASS" if not missing else "FAIL"}, ([{"severity":"CRITICAL","check":"table_presence","items":missing}] if missing else [])


def check_leagues(store: NeonStore, expected):
    expected_keys = {x.key for x in expected}
    rows = sql_rows(store, "select canonical_key,name,country,source_provider,source_updated_at,updated_at from leagues")
    actual = {str(r["canonical_key"]) for r in rows}
    missing = sorted(expected_keys - actual)
    unexpected = sorted(actual - expected_keys)
    match_counts = sql_rows(store, "select league_canonical_key,count(*) as matches from matches group by league_canonical_key")
    by_league = {str(r["league_canonical_key"]): int(r["matches"]) for r in match_counts}
    no_matches = sorted(k for k in expected_keys if by_league.get(k, 0) == 0)
    blank_names = sorted(str(r["canonical_key"]) for r in rows if not str(r.get("name") or "").strip())
    issues = []
    if missing: issues.append({"severity":"CRITICAL","check":"league_coverage","items":missing})
    if no_matches: issues.append({"severity":"CRITICAL","check":"league_match_coverage","items":no_matches})
    if blank_names: issues.append({"severity":"CRITICAL","check":"league_blank_names","items":blank_names})
    if unexpected: issues.append({"severity":"WARNING","check":"unexpected_leagues","items":unexpected})
    return {
        "configured": len(expected_keys), "in_neon": len(actual), "covered": len(expected_keys & actual),
        "missing": missing, "leagues_with_zero_matches": no_matches,
        "match_counts": {k: by_league.get(k, 0) for k in sorted(expected_keys)},
        "unexpected": unexpected, "blank_names": blank_names,
        "status": "PASS" if not (missing or no_matches or blank_names) else "FAIL",
    }, issues


def check_matches(store: NeonStore):
    now = datetime.now(timezone.utc)
    checks = {
        "total": count(store,"matches"),
        "finished": count(store,"matches","status='finished'"),
        "scheduled": count(store,"matches","status='scheduled'"),
        "postponed": count(store,"matches","status='postponed'"),
        "cancelled": count(store,"matches","status='cancelled'"),
        "bad_status": count(store,"matches","status not in ('scheduled','finished','postponed','cancelled')"),
        "missing_identity": count(store,"matches","league_canonical_key is null or home_team_canonical_key is null or away_team_canonical_key is null"),
        "finished_missing_score": count(store,"matches","status='finished' and (home_score is null or away_score is null)"),
        "negative_scores": count(store,"matches","home_score < 0 or away_score < 0"),
        "same_team": count(store,"matches","home_team_canonical_key=away_team_canonical_key"),
        "orphan_leagues": len(sql_rows(store,"select m.canonical_key from matches m left join leagues l on l.canonical_key=m.league_canonical_key where l.canonical_key is null")),
        "orphan_home_teams": len(sql_rows(store,"select m.canonical_key from matches m left join teams t on t.canonical_key=m.home_team_canonical_key where t.canonical_key is null")),
        "orphan_away_teams": len(sql_rows(store,"select m.canonical_key from matches m left join teams t on t.canonical_key=m.away_team_canonical_key where t.canonical_key is null")),
        "future_finished": count(store,"matches","status='finished' and kickoff_at > %s",(now,)),
    }
    duplicates = sql_rows(store,"""select league_canonical_key,season,kickoff_at,home_team_canonical_key,away_team_canonical_key,source_provider,count(*) as n
        from matches group by league_canonical_key,season,kickoff_at,home_team_canonical_key,away_team_canonical_key,source_provider having count(*)>1""")
    checks["duplicate_groups"] = len(duplicates)
    critical_names = {"bad_status","missing_identity","finished_missing_score","negative_scores","same_team","orphan_leagues","orphan_home_teams","orphan_away_teams","future_finished","duplicate_groups"}
    issues = [{"severity":"CRITICAL","check":k,"count":v} for k,v in checks.items() if k in critical_names and v]
    return {**checks,"duplicate_examples":duplicates[:20],"status":"PASS" if not issues else "FAIL"}, issues


def check_teams(store: NeonStore):
    blank = count(store,"teams","name is null or btrim(name)=''")
    orphan = len(sql_rows(store,"select t.canonical_key from teams t left join leagues l on l.canonical_key=t.league_canonical_key where l.canonical_key is null"))
    issues=[]
    if blank: issues.append({"severity":"CRITICAL","check":"team_blank_names","count":blank})
    if orphan: issues.append({"severity":"CRITICAL","check":"team_orphan_leagues","count":orphan})
    return {"total":count(store,"teams"),"blank_names":blank,"orphan_leagues":orphan,"status":"PASS" if not issues else "FAIL"},issues


def check_provider_integrity(store: NeonStore):
    missing_source = len(sql_rows(store,"select pr.provider_key from provider_records pr left join provider_sources ps on ps.provider_key=pr.provider_key where ps.provider_key is null"))
    orphan_match = len(sql_rows(store,"select pr.canonical_key from provider_records pr left join matches m on m.canonical_key=pr.canonical_key where pr.record_type='match' and m.canonical_key is null"))
    issues=[]
    if missing_source: issues.append({"severity":"CRITICAL","check":"provider_record_missing_source","count":missing_source})
    if orphan_match: issues.append({"severity":"CRITICAL","check":"provider_record_orphan_match","count":orphan_match})
    return {"total":count(store,"provider_records"),"missing_provider_source":missing_source,"orphan_match_records":orphan_match,"status":"PASS" if not issues else "FAIL"},issues


def check_sources(store: NeonStore):
    missing_provider_sources = len(sql_rows(store,"select d.provider_key from data_sources d left join provider_sources p on p.provider_key=d.provider_key where p.provider_key is null"))
    error_sources = sql_rows(store,"select provider_key,last_success_at,last_data_at,last_success_rows,last_error from data_sources where last_error is not null")
    issues=[]
    if missing_provider_sources: issues.append({"severity":"CRITICAL","check":"data_source_missing_provider_source","count":missing_provider_sources})
    if error_sources: issues.append({"severity":"WARNING","check":"data_sources_with_last_error","count":len(error_sources),"items":error_sources[:20]})
    return {"rows":count(store,"data_sources"),"missing_provider_sources":missing_provider_sources,"sources_with_last_error":len(error_sources),"status":"PASS" if not missing_provider_sources else "FAIL"},issues


def check_strength(store: NeonStore):
    rows=count(store,"team_strength")
    null_elo=count(store,"team_strength","elo is null")
    nonpositive=count(store,"team_strength","elo<=0")
    blank_slug=count(store,"team_strength","team_slug is null or btrim(team_slug)=''")
    duplicates=sql_rows(store,"select team_slug,count(*) as n from team_strength group by team_slug having count(*)>1")
    issues=[]
    for name,value in (("strength_null_elo",null_elo),("strength_nonpositive_elo",nonpositive),("strength_blank_slug",blank_slug),("strength_duplicate_team_slug",len(duplicates))):
        if value: issues.append({"severity":"CRITICAL","check":name,"count":value})
    return {"rows":rows,"null_elo":null_elo,"nonpositive_elo":nonpositive,"blank_slug":blank_slug,"duplicate_team_slugs":len(duplicates),"status":"PASS" if not issues else "FAIL"},issues


def check_statistics_and_standings(store: NeonStore):
    stats_orphan_leagues=len(sql_rows(store,"select s.league_canonical_key from team_statistics s left join leagues l on l.canonical_key=s.league_canonical_key where l.canonical_key is null"))
    standings_orphan_leagues=len(sql_rows(store,"select s.league_canonical_key from league_standings s left join leagues l on l.canonical_key=s.league_canonical_key where l.canonical_key is null"))
    bad_stats=count(store,"team_statistics","season is null or btrim(season)='' or team_slug is null or btrim(team_slug)=''")
    bad_standings=count(store,"league_standings","rank<=0 or team is null or btrim(team)=''")
    issues=[]
    for name,value in (("team_statistics_orphan_leagues",stats_orphan_leagues),("team_statistics_invalid_identity",bad_stats),("league_standings_orphan_leagues",standings_orphan_leagues),("league_standings_invalid_rows",bad_standings)):
        if value: issues.append({"severity":"CRITICAL","check":name,"count":value})
    return {"team_statistics_rows":count(store,"team_statistics"),"league_standings_rows":count(store,"league_standings"),"stats_orphan_leagues":stats_orphan_leagues,"stats_invalid_identity":bad_stats,"standings_orphan_leagues":standings_orphan_leagues,"standings_invalid_rows":bad_standings,"status":"PASS" if not issues else "FAIL"},issues


def check_ingestion_runs(store: NeonStore):
    stale_before=datetime.now(timezone.utc)-timedelta(hours=STALE_RUNNING_HOURS)
    invalid=count(store,"ingestion_runs","status not in ('running','succeeded','failed','partial')")
    stale=sql_rows(store,"select id,provider_key,status,started_at,records_seen,records_upserted,error_message from ingestion_runs where status='running' and started_at < %s",(stale_before,))
    succeeded_error=count(store,"ingestion_runs","status='succeeded' and error_message is not null")
    succeeded_zero=count(store,"ingestion_runs","status='succeeded' and records_seen>0 and records_upserted=0")
    failed_no_error=count(store,"ingestion_runs","status='failed' and (error_message is null or btrim(error_message)='')")
    issues=[]
    for name,value,severity in (("ingestion_invalid_status",invalid,"CRITICAL"),("ingestion_stale_running",len(stale),"CRITICAL"),("ingestion_succeeded_with_error",succeeded_error,"CRITICAL"),("ingestion_succeeded_zero_upsert",succeeded_zero,"CRITICAL"),("ingestion_failed_without_error",failed_no_error,"WARNING")):
        if value: issues.append({"severity":severity,"check":name,"count":value,"items":stale if name=="ingestion_stale_running" else None})
    return {"invalid_status":invalid,"stale_running":len(stale),"succeeded_with_error":succeeded_error,"succeeded_zero_upsert":succeeded_zero,"failed_without_error":failed_no_error,"stale_rows":stale,"status":"PASS" if not any(i["severity"]=="CRITICAL" for i in issues) else "FAIL"},issues


def check_prediction_tables(store: NeonStore):
    total=count(store,"predictions")
    bad_range=count(store,"predictions","home_probability<0 or home_probability>1 or draw_probability<0 or draw_probability>1 or away_probability<0 or away_probability>1")
    bad_sum=count(store,"predictions","abs((coalesce(home_probability,0)+coalesce(draw_probability,0)+coalesce(away_probability,0))-1)>0.01")
    orphan_archive=len(sql_rows(store,"select a.prediction_key from prediction_archive a left join matches m on m.canonical_key=a.match_canonical_key where a.match_canonical_key is not null and m.canonical_key is null"))
    issues=[]
    if bad_range: issues.append({"severity":"CRITICAL","check":"prediction_probability_range","count":bad_range})
    if bad_sum: issues.append({"severity":"CRITICAL","check":"prediction_probability_sum","count":bad_sum})
    if orphan_archive: issues.append({"severity":"CRITICAL","check":"prediction_archive_orphan_match","count":orphan_archive})
    return {"rows":total,"bad_probability_rows":bad_range,"probability_sum_outside_tolerance":bad_sum,"orphan_archive_matches":orphan_archive,"status":"PASS" if not issues else "FAIL","empty_allowed_before_prediction_workflow":True},issues


def run_audit() -> dict[str,Any]:
    expected=load_enabled_leagues()
    store=NeonStore()
    store.verify_connection()
    checks={}
    issues=[]
    for name,fn,args in [
        ("tables",check_tables,(store,)),("leagues",check_leagues,(store,expected)),
        ("matches",check_matches,(store,)),("teams",check_teams,(store,)),
        ("provider_integrity",check_provider_integrity,(store,)),("data_sources",check_sources,(store,)),
        ("team_strength",check_strength,(store,)),("statistics_and_standings",check_statistics_and_standings,(store,)),
        ("ingestion_runs",check_ingestion_runs,(store,)),("predictions",check_prediction_tables,(store,)),
    ]:
        result,found=fn(*args); checks[name]=result; issues.extend(found)
    critical=sum(i["severity"]=="CRITICAL" for i in issues)
    warnings=sum(i["severity"]=="WARNING" for i in issues)
    return {"audit_version":"1.2","database":"Neon PostgreSQL","weather_audit":"excluded_suspended","generated_at":datetime.now(timezone.utc).isoformat(),"checks":checks,"critical_errors":critical,"warnings":warnings,"data_ready_for_ml":critical==0,"issues":issues}


def main()->int:
    report=run_audit()
    print(json.dumps(report,indent=2,default=str))
    return 0 if report["data_ready_for_ml"] else 1

if __name__=="__main__":
    raise SystemExit(main())
