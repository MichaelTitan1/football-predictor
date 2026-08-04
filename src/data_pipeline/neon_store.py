"""Direct Neon PostgreSQL persistence for canonical football data."""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Sequence

from dotenv import load_dotenv

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord

load_dotenv()


class NeonStoreError(RuntimeError):
    pass


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise NeonStoreError(f"{name} must be an integer") from exc
    if value < minimum:
        raise NeonStoreError(f"{name} must be >= {minimum}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


SCHEMA_SQL = """
create table if not exists provider_sources(provider_key text primary key, display_name text not null, source_type text not null, updated_at timestamptz not null default now());
create table if not exists leagues(canonical_key text primary key, name text not null, country text, source_provider text, source_updated_at timestamptz, updated_at timestamptz not null default now());
create table if not exists teams(canonical_key text primary key, league_canonical_key text references leagues(canonical_key), name text not null, source_provider text, source_updated_at timestamptz, updated_at timestamptz not null default now());
create table if not exists matches(canonical_key text primary key, league_canonical_key text not null references leagues(canonical_key), season text, kickoff_at timestamptz not null, status text not null, home_team_canonical_key text not null references teams(canonical_key), away_team_canonical_key text not null references teams(canonical_key), home_score integer, away_score integer, source_provider text not null, source_match_id text, source_updated_at timestamptz, created_at timestamptz not null default now(), updated_at timestamptz not null default now(), unique(league_canonical_key, season, kickoff_at, home_team_canonical_key, away_team_canonical_key, source_provider));
create table if not exists provider_records(provider_key text not null, record_type text not null, source_id text not null, canonical_key text not null, payload jsonb, source_updated_at timestamptz, updated_at timestamptz not null default now(), primary key(provider_key, record_type, source_id));
create table if not exists data_sources(provider_key text primary key, display_name text, last_attempt_at timestamptz, last_success_at timestamptz, last_data_at timestamptz, last_success_rows integer, last_error text, freshness_policy text);
create table if not exists ingestion_runs(id bigserial primary key, provider_key text not null, status text not null, started_at timestamptz not null default now(), finished_at timestamptz, records_seen integer default 0, records_upserted integer default 0, newest_match_at timestamptz, error_message text);
create table if not exists team_strength(team_slug text primary key, team_name text, elo double precision, rank integer, country text, level text, from_date text, to_date text, source_provider text default 'clubelo-soccerdata', updated_at timestamptz not null default now());
create table if not exists team_statistics(league_canonical_key text not null, team_slug text not null, season text not null, xg double precision, xga double precision, goals double precision, updated_at timestamptz not null default now(), primary key(league_canonical_key,team_slug,season));
create table if not exists league_standings(league_canonical_key text not null, rank integer not null, team text not null, played integer, wins integer, draws integer, losses integer, gf integer, ga integer, points integer, updated_at timestamptz not null default now(), primary key(league_canonical_key,rank));
create table if not exists match_weather(match_canonical_key text primary key references matches(canonical_key), forecast_at timestamptz, temperature_c double precision, rain_mm double precision, wind_mps double precision, condition text, source_provider text default 'met-norway', updated_at timestamptz not null default now());
create table if not exists model_versions(version text primary key, model_family text, artifact_path text, artifact_sha256 text, feature_schema_version text, calibration_method text, is_active boolean default false);
create table if not exists predictions(match_canonical_key text primary key references matches(canonical_key), predicted_at timestamptz not null, model_version text, model_artifact_hash text, home_probability double precision, draw_probability double precision, away_probability double precision, selected_prediction text, confidence double precision, market text not null default 'match_winner_1x2', data_freshness_at text, updated_at timestamptz not null default now());
create table if not exists prediction_archive(prediction_key text primary key, match_id text, match_canonical_key text references matches(canonical_key), predicted_at timestamptz not null, model_version text, model_artifact_hash text, feature_schema_version text, home_probability double precision, draw_probability double precision, away_probability double precision, selected_prediction text, confidence double precision, home_elo double precision, away_elo double precision, home_xg double precision, away_xg double precision, home_form jsonb, away_form jsonb, weather jsonb, feature_values jsonb, actual_result text, actual_home_score integer, actual_away_score integer, resolved_at timestamptz, is_correct boolean);
"""


class NeonStore:
    def __init__(self, dsn: str | None = None, connection: Any | None = None):
        self.dsn = dsn or os.getenv("DATABASE_URL", "")
        self.batch_size = _env_int("FOVRA_NEON_BATCH_SIZE", _env_int("FOVRA_DB_BATCH_SIZE", 500))
        self.batch_start = _env_int("FOVRA_NEON_MATCH_BATCH_START", 1)
        self.batch_retries = _env_int("FOVRA_NEON_BATCH_RETRIES", 3, minimum=0)
        self.timeout = _env_int("FOVRA_NEON_TIMEOUT", 300)
        self.connection = connection
        if self.connection is None:
            if not self.dsn:
                raise NeonStoreError("DATABASE_URL is required for Neon PostgreSQL")
            try:
                import psycopg
            except ImportError as exc:
                raise NeonStoreError("psycopg is required for Neon PostgreSQL") from exc
            self.connection = psycopg.connect(self.dsn, connect_timeout=self.timeout)
            self.connection.autocommit = False

    def initialize_schema(self) -> None:
        with self.connection.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        self.connection.commit()

    @staticmethod
    def _chunks(rows: Iterable[Any], size: int) -> Iterator[list[Any]]:
        batch=[]
        for row in rows:
            batch.append(row)
            if len(batch)==size:
                yield batch; batch=[]
        if batch: yield batch

    def _execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self.connection.cursor() as cur:
            cur.execute(sql, params or ())

    def _fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self.connection.cursor() as cur:
            cur.execute(sql, params or ())
            cols=[d[0] for d in (cur.description or [])]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update(self, table: str, payload: dict[str, Any], where: str, params: Sequence[Any]) -> None:
        cols=list(payload.keys())
        sql=f"update {table} set " + ", ".join(f"{c}=%s" for c in cols) + f" where {where}"
        values=tuple(json.dumps(_jsonable(payload[c])) if isinstance(_jsonable(payload[c]), (dict, list)) else _jsonable(payload[c]) for c in cols)
        self._execute(sql, (*values, *params)); self.connection.commit()

    def select(self, table: str, where: str = "", params: Sequence[Any] = (), order: str = "", limit: int | None = None, columns: str = "*") -> list[dict[str, Any]]:
        if not table.replace("_", "").isalnum():
            raise NeonStoreError("invalid table name")
        sql=f"select {columns} from {table}"
        if where: sql += f" where {where}"
        if order: sql += f" order by {order}"
        if limit is not None: sql += " limit %s"; params=(*params, limit)
        return self._fetchall(sql, params)

    def upsert(self, table: str, rows: list[dict[str, Any]], on_conflict: str) -> None:
        if not rows: return
        cols=list(rows[0].keys())
        placeholders=", ".join(["%s"]*len(cols))
        updates=", ".join(f"{c}=excluded.{c}" for c in cols if c not in [x.strip() for x in on_conflict.split(',')])
        sql=f"insert into {table} ({', '.join(cols)}) values ({placeholders}) on conflict ({on_conflict}) do update set {updates}"
        with self.connection.cursor() as cur:
            for row in rows:
                cur.execute(sql, tuple(json.dumps(_jsonable(row[c])) if isinstance(_jsonable(row[c]), (dict, list)) else _jsonable(row[c]) for c in cols))
        self.connection.commit()

    def record_ingestion_start(self, provider: str) -> str:
        rows=self._fetchall("insert into ingestion_runs(provider_key,status,started_at) values(%s,'running',%s) returning id", (provider, datetime.now(timezone.utc)))
        self.connection.commit(); return str(rows[0]["id"])

    def record_ingestion_finish(self, run_id: str, *, status: str, records_seen: int, records_upserted: int, newest_match_at: str | None = None, error_message: str | None = None) -> None:
        self._execute("update ingestion_runs set finished_at=%s,status=%s,records_seen=%s,records_upserted=%s,newest_match_at=%s,error_message=%s where id=%s", (datetime.now(timezone.utc), status, records_seen, records_upserted, newest_match_at, error_message, run_id)); self.connection.commit()

    def _existing_provider_payloads(self, provider: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        ids = [r.get("source_match_id") or r["canonical_key"] for r in rows]
        placeholders = ", ".join(["%s"] * len(ids))
        found = self._fetchall(
            f"select source_id,payload from provider_records where provider_key=%s and record_type=%s and source_id in ({placeholders})",
            (provider, "match", *ids),
        )
        return {str(r["source_id"]): r.get("payload") for r in found}

    def upsert_snapshot(self, leagues: Iterable[LeagueRecord], teams: Iterable[TeamRecord], matches: Iterable[MatchRecord], provider: str, fetched_at: str) -> int:
        now=datetime.now(timezone.utc).isoformat(); self.initialize_schema()
        self.upsert("provider_sources", [{"provider_key": provider, "display_name": provider, "source_type": "match-data", "updated_at": now}], "provider_key")
        self.upsert("leagues", [{"canonical_key":l.key,"name":l.name,"country":l.country,"source_provider":provider,"source_updated_at":fetched_at,"updated_at":now} for l in leagues], "canonical_key")
        self.upsert("teams", [{"canonical_key":f"{t.league_key}:{t.key}","league_canonical_key":t.league_key,"name":t.name,"source_provider":provider,"source_updated_at":fetched_at,"updated_at":now} for t in teams], "canonical_key")
        match_rows=[{"canonical_key":m.match_key,"league_canonical_key":m.league_key,"season":m.season,"kickoff_at":m.kickoff_utc,"status":m.status,"home_team_canonical_key":f"{m.league_key}:{m.home_team}","away_team_canonical_key":f"{m.league_key}:{m.away_team}","home_score":m.home_score,"away_score":m.away_score,"source_provider":provider,"source_match_id":m.source_id,"source_updated_at":fetched_at,"updated_at":now} for m in matches]
        uploaded=skipped=failed=0; total=len(match_rows); batches=math.ceil(total/self.batch_size) if total else 0; started=time.monotonic()
        for batch_no,batch in enumerate(self._chunks(match_rows,self.batch_size),1):
            if batch_no < self.batch_start: skipped += len(batch); continue
            for attempt in range(self.batch_retries+1):
                try:
                    existing = self._existing_provider_payloads(provider, batch)
                    changed = []
                    batch_unchanged = 0
                    for r in batch:
                        source_id = r.get("source_match_id") or r["canonical_key"]
                        existing_payload = existing.get(source_id)
                        if isinstance(existing_payload, str):
                            try:
                                existing_payload = json.loads(existing_payload)
                            except json.JSONDecodeError:
                                pass
                        comparable_existing = {k: v for k, v in existing_payload.items() if k != "updated_at"} if isinstance(existing_payload, dict) else existing_payload
                        comparable_row = {k: v for k, v in r.items() if k != "updated_at"}
                        if comparable_existing == comparable_row:
                            batch_unchanged += 1
                        else:
                            changed.append(r)
                    skipped += batch_unchanged
                    if changed:
                        self.upsert("matches", changed, "canonical_key")
                        records=[{"provider_key":provider,"record_type":"match","source_id":r.get("source_match_id") or r["canonical_key"],"canonical_key":r["canonical_key"],"payload":r,"source_updated_at":fetched_at,"updated_at":now} for r in changed]
                        self.upsert("provider_records", records, "provider_key, record_type, source_id")
                    uploaded += len(changed); break
                except Exception as exc:
                    getattr(self, "connection", None) and self.connection.rollback()
                    if attempt >= self.batch_retries:
                        failed += 1; print(f"Batch {batch_no} failed: {exc}")
                    else: time.sleep(min(2**attempt,30))
            print(f"current batch={batch_no}/{batches} total rows={total} new/updated rows={uploaded} unchanged rows={skipped} uploaded={uploaded} skipped={skipped} failed batches={failed} remaining batches={max(batches-batch_no,0)} elapsed={time.monotonic()-started:.1f}s")
        return uploaded

    def resolve_predictions(self, matches: Iterable[MatchRecord]) -> int:
        finished = {m.match_key: m for m in matches if m.status == "finished" and m.home_score is not None and m.away_score is not None}
        if not finished:
            return 0
        rows = self.select("prediction_archive", "is_correct is null", columns="prediction_key,match_canonical_key,selected_prediction")
        resolved = 0
        for row in rows:
            match = finished.get(str(row.get("match_canonical_key")))
            if not match:
                continue
            actual = "H" if match.home_score > match.away_score else "A" if match.away_score > match.home_score else "D"
            self.update("prediction_archive", {"actual_result": actual, "actual_home_score": match.home_score, "actual_away_score": match.away_score, "resolved_at": datetime.now(timezone.utc).isoformat(), "is_correct": row.get("selected_prediction") == actual}, "prediction_key = %s", (row["prediction_key"],))
            resolved += 1
        return resolved
