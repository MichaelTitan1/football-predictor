"""Provider-independent canonical football records.

Production persistence is Supabase PostgreSQL. SQLite support in this module is
only for isolated local testing and is never the production source of truth.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DB_DEFAULT = Path("data/processed/fovra_data.sqlite3")
STATUSES = {"scheduled", "finished", "postponed", "cancelled"}

@dataclass(frozen=True)
class LeagueRecord:
    key: str
    name: str
    country: Optional[str] = None

@dataclass(frozen=True)
class TeamRecord:
    key: str
    name: str
    league_key: str

@dataclass(frozen=True)
class MatchRecord:
    provider: str
    league_key: str
    season: Optional[str]
    kickoff_utc: str
    home_team: str
    away_team: str
    status: str = "scheduled"
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    source_id: Optional[str] = None

    @property
    def match_key(self) -> str:
        # Prefer a provider's stable source ID so kickoff corrections do not
        # create duplicate matches. Fall back to deterministic match identity.
        raw = "|".join([self.provider, "source", self.source_id]) if self.source_id else "|".join([self.provider, self.league_key, self.season or "", self.kickoff_utc, self.home_team, self.away_team])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def utc_now() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")

def connect(path: str | Path = DB_DEFAULT) -> sqlite3.Connection:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); conn=sqlite3.connect(p); conn.row_factory=sqlite3.Row; conn.execute("PRAGMA foreign_keys = ON"); return conn

def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    create table if not exists leagues(league_key text primary key,name text not null,country text,updated_at text not null);
    create table if not exists teams(team_key text not null,league_key text not null,name text not null,updated_at text not null,primary key(league_key,team_key),foreign key(league_key) references leagues(league_key));
    create table if not exists matches(match_key text primary key,provider text not null,source_id text,league_key text not null,season text,kickoff_utc text not null,home_team_key text not null,away_team_key text not null,status text not null,home_score integer,away_score integer,first_seen_at text not null,updated_at text not null,foreign key(league_key) references leagues(league_key),foreign key(league_key,home_team_key) references teams(league_key,team_key),foreign key(league_key,away_team_key) references teams(league_key,team_key));
    create index if not exists idx_local_matches_kickoff on matches(kickoff_utc);
    create table if not exists source_updates(provider text primary key,last_success_at text,last_attempt_at text not null,records_seen integer not null default 0,records_upserted integer not null default 0,error text);
    """); conn.commit()

def _validate_match(match: MatchRecord) -> None:
    if not match.league_key or not match.home_team or not match.away_team or match.home_team==match.away_team: raise ValueError("invalid match identity")
    if match.status not in STATUSES: raise ValueError(f"unsupported match status: {match.status}")
    if match.status=="finished" and (match.home_score is None or match.away_score is None): raise ValueError("finished matches require both scores")
    if any(score is not None and score<0 for score in (match.home_score,match.away_score)): raise ValueError("scores cannot be negative")

def upsert_records(conn: sqlite3.Connection, leagues: Iterable[LeagueRecord], teams: Iterable[TeamRecord], matches: Iterable[MatchRecord], provider: str) -> int:
    now=utc_now(); count=0
    with conn:
        for league in leagues: conn.execute("insert into leagues values(?,?,?,?) on conflict(league_key) do update set name=excluded.name,country=excluded.country,updated_at=excluded.updated_at",(league.key,league.name,league.country,now))
        for team in teams: conn.execute("insert into teams values(?,?,?,?) on conflict(league_key,team_key) do update set name=excluded.name,updated_at=excluded.updated_at",(team.key,team.league_key,team.name,now))
        for match in matches:
            _validate_match(match); conn.execute("insert into matches(match_key,provider,source_id,league_key,season,kickoff_utc,home_team_key,away_team_key,status,home_score,away_score,first_seen_at,updated_at) values(?,?,?,?,?,?,?,?,?,?,?,?,?) on conflict(match_key) do update set source_id=coalesce(excluded.source_id,matches.source_id),kickoff_utc=excluded.kickoff_utc,status=case when excluded.status='finished' then 'finished' else excluded.status end,home_score=coalesce(excluded.home_score,matches.home_score),away_score=coalesce(excluded.away_score,matches.away_score),updated_at=excluded.updated_at",(match.match_key,match.provider,match.source_id,match.league_key,match.season,match.kickoff_utc,match.home_team,match.away_team,match.status,match.home_score,match.away_score,now,now)); count+=1
        conn.execute("insert into source_updates values(?,?,?,?,?,null) on conflict(provider) do update set last_success_at=excluded.last_success_at,last_attempt_at=excluded.last_attempt_at,records_seen=excluded.records_seen,records_upserted=excluded.records_upserted,error=null",(provider,now,now,count,count))
    return count

def record_source_error(conn: sqlite3.Connection, provider: str, error: str, records_seen: int=0) -> None:
    now=utc_now()
    with conn: conn.execute("insert into source_updates(provider,last_success_at,last_attempt_at,records_seen,records_upserted,error) values(?,null,?,?,0,?) on conflict(provider) do update set last_attempt_at=excluded.last_attempt_at,records_seen=excluded.records_seen,error=excluded.error",(provider,now,records_seen,error[:1000]))

def summary(conn: sqlite3.Connection)->dict:
    one=lambda q:int(conn.execute(q).fetchone()[0])
    return {"leagues":one("select count(*) from leagues"),"teams":one("select count(*) from teams"),"matches":one("select count(*) from matches"),"finished_matches":one("select count(*) from matches where status='finished'"),"upcoming_matches":one("select count(*) from matches where status='scheduled'"),"newest_finished_match":conn.execute("select max(kickoff_utc) from matches where status='finished'").fetchone()[0],"next_scheduled_match":conn.execute("select min(kickoff_utc) from matches where status='scheduled'").fetchone()[0]}
