"""Supabase PostgreSQL persistence for the canonical Fovra data layer.

Production source of truth: the existing Supabase project. The service key is
used only by trusted backend/update jobs; browser clients never receive it.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence
import math
import time
import requests
from .canonical_data import LeagueRecord, MatchRecord, TeamRecord

class SupabaseStoreError(RuntimeError): pass

def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SupabaseStoreError(f"{name} must be an integer") from exc
    if value < minimum:
        raise SupabaseStoreError(f"{name} must be >= {minimum}")
    return value

class SupabaseStore:
    def __init__(self,url:str|None=None,key:str|None=None,timeout:int|None=None,session:requests.Session|None=None):
        self.url=(url or os.getenv("SUPABASE_URL","")).rstrip("/"); self.key=key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SECRET_KEY","")
        self.timeout=timeout if timeout is not None else _env_int("FOVRA_SUPABASE_TIMEOUT",300)
        self.batch_size=_env_int("FOVRA_SUPABASE_BATCH_SIZE",500)
        self.batch_start=_env_int("FOVRA_SUPABASE_MATCH_BATCH_START",1)
        self.batch_retries=_env_int("FOVRA_SUPABASE_BATCH_RETRIES",3,minimum=0)
        self.session=session or requests.Session()
        if not self.url or not self.key: raise SupabaseStoreError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_SECRET_KEY) are required")
    @property
    def headers(self): return {"apikey":self.key,"Authorization":f"Bearer {self.key}","Content-Type":"application/json","Prefer":"return=minimal,resolution=merge-duplicates"}
    @staticmethod
    def _jsonable(value):
        if isinstance(value, dict):
            return {k: SupabaseStore._jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SupabaseStore._jsonable(v) for v in value]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    def _request(self,method,table,*,params=None,payload=None,prefer=None):
        headers=dict(self.headers)
        payload = self._jsonable(payload)
        if prefer: headers["Prefer"]=prefer
        response=self.session.request(method,f"{self.url}/rest/v1/{table}",headers=headers,params=params,json=payload,timeout=self.timeout)
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

    @staticmethod
    def _chunks(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        for row in rows:
            batch.append(row)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _format_remaining(started_at: float, completed: int, total: int | None) -> str:
        if not total or completed <= 0:
            return "unknown"
        remaining = max(total - completed, 0)
        seconds = (time.monotonic() - started_at) / completed * remaining
        return f"{seconds / 60:.1f} minutes"

    def _existing_match_timestamps(self, rows: Sequence[dict[str, Any]]) -> dict[str, tuple[str | None, str | None]]:
        existing: dict[str, tuple[str | None, str | None]] = {}
        # Keep lookup URLs small while the upload batch itself remains 500 rows by default.
        for chunk in self._chunks(({"canonical_key": row["canonical_key"]} for row in rows), 75):
            keys = ",".join(row["canonical_key"] for row in chunk)
            result = self._request(
                "GET",
                "matches",
                params={"canonical_key": f"in.({keys})", "select": "canonical_key,updated_at,source_updated_at"},
            ) or []
            for row in result:
                existing[row["canonical_key"]] = (row.get("updated_at"), row.get("source_updated_at"))
        return existing

    def _filter_changed_matches(self, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        existing = self._existing_match_timestamps(rows)
        changed = []
        skipped = 0
        for row in rows:
            timestamps = existing.get(row["canonical_key"])
            if timestamps == (row.get("updated_at"), row.get("source_updated_at")):
                skipped += 1
            else:
                changed.append(row)
        return changed, skipped

    def _upsert_match_batch(self, batch_number: int, rows: list[dict[str, Any]]) -> int:
        for attempt in range(self.batch_retries + 1):
            try:
                changed, skipped = self._filter_changed_matches(rows)
                if changed:
                    self.upsert("matches", changed, "canonical_key")
                return skipped
            except Exception:
                if attempt >= self.batch_retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
        return 0

    def upsert_snapshot(self,leagues:Iterable[LeagueRecord],teams:Iterable[TeamRecord],matches:Iterable[MatchRecord],provider:str,fetched_at:str)->int:
        started_at=time.monotonic(); now=fetched_at or datetime.now(timezone.utc).isoformat()
        self.upsert("leagues",[{"canonical_key":l.key,"name":l.name,"source_provider":provider,"source_updated_at":fetched_at,"updated_at":now} for l in leagues],"canonical_key")
        self.upsert("teams",[{"canonical_key":f"{t.league_key}:{t.key}","league_canonical_key":t.league_key,"name":t.name,"source_provider":provider,"source_updated_at":fetched_at,"updated_at":now} for t in teams],"canonical_key")
        total_matches = len(matches) if hasattr(matches, "__len__") else None
        total_batches = math.ceil(total_matches / self.batch_size) if total_matches is not None else None
        downloaded=skipped=uploaded=0; failed_batches: list[int] = []
        for batch_number, batch in enumerate(self._chunks(({"canonical_key":m.match_key,"league_canonical_key":m.league_key,"season":m.season,"kickoff_at":m.kickoff_utc,"status":m.status,"home_team_canonical_key":f"{m.league_key}:{m.home_team}","away_team_canonical_key":f"{m.league_key}:{m.away_team}","home_score":m.home_score,"away_score":m.away_score,"source_provider":provider,"source_match_id":m.source_id,"source_updated_at":fetched_at,"updated_at":now} for m in matches), self.batch_size), start=1):
            downloaded += len(batch)
            if batch_number < self.batch_start:
                continue
            label_total = str(total_batches) if total_batches is not None else "?"
            print(f"Uploading batch {batch_number}/{label_total}")
            try:
                batch_skipped = self._upsert_match_batch(batch_number, batch)
                skipped += batch_skipped
                uploaded += len(batch) - batch_skipped
            except Exception as exc:
                failed_batches.append(batch_number)
                print(f"Batch {batch_number} failed after {self.batch_retries + 1} attempt(s): {exc}")
            print(f"Matches uploaded:\n{uploaded} / {total_matches if total_matches is not None else downloaded}")
            print(f"Estimated remaining:\n{self._format_remaining(started_at, batch_number, total_batches)}")
            del batch
        elapsed = time.monotonic() - started_at
        print(f"Downloaded:\n{downloaded}")
        print(f"Skipped:\n{skipped}")
        print(f"Uploaded:\n{uploaded}")
        print(f"Failed batches:\n{failed_batches}")
        print(f"Elapsed time:\n{elapsed:.1f} seconds")
        return uploaded
    def resolve_predictions(self, matches: Iterable[MatchRecord]) -> int:
        """
        Resolve only predictions that exist in prediction_archive.
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

        rows = self._request(
            "GET",
            "prediction_archive",
            params={
                "is_correct": "is.null",
                "select": "prediction_key,match_canonical_key,selected_prediction",
            },
        ) or []

        if not rows:
            return 0

        now = datetime.now(timezone.utc).isoformat()

        for row in rows:
            match = finished_matches.get(
                row.get("match_canonical_key")
            )

            if not match:
                continue

            actual = (
                "H"
                if match.home_score > match.away_score
                else "A"
                if match.away_score > match.home_score
                else "D"
            )

            self._request(
                "PATCH",
                "prediction_archive",
                params={
                    "prediction_key": f"eq.{row['prediction_key']}"
                },
                payload={
                    "actual_result": actual,
                    "actual_home_score": match.home_score,
                    "actual_away_score": match.away_score,
                    "resolved_at": now,
                    "is_correct": row.get("selected_prediction") == actual,
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
