"""Supabase PostgreSQL persistence for the canonical Fovra data layer.

Production source of truth: the existing Supabase project. The service key is
used only by trusted backend/update jobs; browser clients never receive it.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Any, Iterable
import requests
from .canonical_data import LeagueRecord, MatchRecord, TeamRecord

class SupabaseStoreError(RuntimeError): pass

class SupabaseStore:
    def __init__(self,url:str|None=None,key:str|None=None,timeout:int=30):
        self.url=(url or os.getenv("SUPABASE_URL","")).rstrip("/"); self.key=key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY",""); self.timeout=timeout
        if not self.url or not self.key: raise SupabaseStoreError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SECRET_KEY) are required")
    @property
    def headers(self): return {"apikey":self.key,"Authorization":f"Bearer {self.key}","Content-Type":"application/json","Prefer":"return=minimal,resolution=merge-duplicates"}
    def _request(self,method,table,*,params=None,payload=None,prefer=None):
        headers=dict(self.headers)
        if prefer: headers["Prefer"]=prefer
        response=requests.request(method,f"{self.url}/rest/v1/{table}",headers=headers,params=params,json=payload,timeout=self.timeout)
        if response.status_code>=400: raise SupabaseStoreError(f"Supabase {method} {table} failed ({response.status_code}): {response.text[:1000]}")
        if not response.content:return None
        try:return response.json()
        except ValueError:return response.text
    def upsert(self,table,rows,on_conflict):
        if rows:self._request("POST",table,params={"on_conflict":on_conflict},payload=rows)
    def record_ingestion_start(self,provider):
        result=self._request("POST","ingestion_runs",params={"select":"id"},payload={"provider_key":provider,"status":"running","started_at":datetime.now(timezone.utc).isoformat()},prefer="return=representation")
        if not result: raise SupabaseStoreError("Supabase did not return an ingestion run id")
        return str(result[0]["id"])
    def record_ingestion_finish(self,run_id,*,status,records_seen,records_upserted,newest_match_at=None,error_message=None):
        self._request("PATCH","ingestion_runs",params={"id":f"eq.{run_id}"},payload={"finished_at":datetime.now(timezone.utc).isoformat(),"status":status,"records_seen":records_seen,"records_upserted":records_upserted,"newest_match_at":newest_match_at,"error_message":error_message[:2000] if error_message else None})
    def upsert_snapshot(self,leagues:Iterable[LeagueRecord],teams:Iterable[TeamRecord],matches:Iterable[MatchRecord],provider:str,fetched_at:str)->int:
        now=datetime.now(timezone.utc).isoformat(); match_list=list(matches)
        self.upsert("leagues",[{"canonical_key":l.key,"name":l.name,"source_provider":provider,"source_updated_at":fetched_at,"updated_at":now} for l in leagues],"canonical_key")
        self.upsert("teams",[{"canonical_key":f"{t.league_key}:{t.key}","league_canonical_key":t.league_key,"name":t.name,"source_provider":provider,"source_updated_at":fetched_at,"updated_at":now} for t in teams],"canonical_key")
        self.upsert("matches",[{"canonical_key":m.match_key,"league_canonical_key":m.league_key,"season":m.season,"kickoff_at":m.kickoff_utc,"status":m.status,"home_team_canonical_key":f"{m.league_key}:{m.home_team}","away_team_canonical_key":f"{m.league_key}:{m.away_team}","home_score":m.home_score,"away_score":m.away_score,"source_provider":provider,"source_match_id":m.source_id,"source_updated_at":fetched_at,"updated_at":now} for m in match_list],"canonical_key")
        return len(match_list)
    def resolve_predictions(self, matches: Iterable[MatchRecord]) -> int:
    """
    Resolve only predictions that exist in prediction_archive.

    Production optimization:
    - avoids scanning every historical match
    - avoids one Supabase request per match
    - processes only matches with unresolved predictions
    """

    finished_matches = {
        match.match_key: match
        for match in matches
        if match.status == "finished"
        and match.home_score is not None
        and match.away_score is not None
    }

    if not finished_matches:
        return 0

    resolved = 0

    # Get unresolved predictions only once
    rows = self._request(
        "GET",
        "prediction_archive",
        params={
            "is_correct": "is.null",
            "select": "prediction_key,match_canonical_key,selected_prediction"
        },
    ) or []

    if not rows:
        return 0

    updates = []

    now = datetime.now(timezone.utc).isoformat()

    for row in rows:
        match_key = row.get("match_canonical_key")

        match = finished_matches.get(match_key)

        if not match:
            continue

        actual = (
            "H"
            if match.home_score > match.away_score
            else "A"
            if match.away_score > match.home_score
            else "D"
        )

        updates.append(
            {
                "prediction_key": row["prediction_key"],
                "actual_result": actual,
                "actual_home_score": match.home_score,
                "actual_away_score": match.away_score,
                "resolved_at": now,
                "is_correct": row.get("selected_prediction") == actual,
            }
        )

    # Apply updates
    for update in updates:
        self._request(
            "PATCH",
            "prediction_archive",
            params={
                "prediction_key": f"eq.{update['prediction_key']}"
            },
            payload={
                "actual_result": update["actual_result"],
                "actual_home_score": update["actual_home_score"],
                "actual_away_score": update["actual_away_score"],
                "resolved_at": update["resolved_at"],
                "is_correct": update["is_correct"],
            },
        )

        resolved += 1

    return resolved
        resolved=0
        for match in matches:
            if match.status!="finished" or match.home_score is None or match.away_score is None: continue
            actual="H" if match.home_score>match.away_score else "A" if match.away_score>match.home_score else "D"
            rows=self._request("GET","prediction_archive",params={"match_canonical_key":f"eq.{match.match_key}","is_correct":"is.null","select":"prediction_key,selected_prediction"}) or []
            for row in rows:
                self._request("PATCH","prediction_archive",params={"prediction_key":f"eq.{row['prediction_key']}"},payload={"actual_result":actual,"actual_home_score":match.home_score,"actual_away_score":match.away_score,"resolved_at":datetime.now(timezone.utc).isoformat(),"is_correct":str(row.get("selected_prediction",""))==actual})
                resolved+=1
        return resolved
