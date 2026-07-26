"""Canonical Fovra V1 football data layer.

This module is intentionally independent of any football-data provider. Providers
return canonical records; this module validates and upserts them into a small
SQLite store. SQLite keeps the V1 foundation dependency-free while leaving the
provider boundary replaceable later.
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
        raw = "|".join(
            [
                self.provider,
                self.league_key,
                self.season or "",
                self.kickoff_utc,
                self.home_team,
                self.away_team,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path = DB_DEFAULT) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS leagues (
            league_key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS teams (
            team_key TEXT NOT NULL,
            league_key TEXT NOT NULL,
            name TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (league_key, team_key),
            FOREIGN KEY (league_key) REFERENCES leagues(league_key)
        );

        CREATE TABLE IF NOT EXISTS matches (
            match_key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            source_id TEXT,
            league_key TEXT NOT NULL,
            season TEXT,
            kickoff_utc TEXT NOT NULL,
            home_team_key TEXT NOT NULL,
            away_team_key TEXT NOT NULL,
            status TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            first_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (league_key) REFERENCES leagues(league_key),
            FOREIGN KEY (league_key, home_team_key) REFERENCES teams(league_key, team_key),
            FOREIGN KEY (league_key, away_team_key) REFERENCES teams(league_key, team_key)
        );

        CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches(kickoff_utc);
        CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_key);
        CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);

        CREATE TABLE IF NOT EXISTS source_updates (
            provider TEXT PRIMARY KEY,
            last_success_at TEXT,
            last_attempt_at TEXT NOT NULL,
            records_seen INTEGER NOT NULL DEFAULT 0,
            records_upserted INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        """
    )
    conn.commit()


def _validate_match(match: MatchRecord) -> None:
    if not match.league_key or not match.home_team or not match.away_team:
        raise ValueError("match requires league, home team and away team")
    if match.home_team == match.away_team:
        raise ValueError("home and away teams must differ")
    if match.status not in STATUSES:
        raise ValueError(f"unsupported match status: {match.status}")
    if match.status == "finished" and (match.home_score is None or match.away_score is None):
        raise ValueError("finished matches require both scores")
    if match.home_score is not None and match.home_score < 0:
        raise ValueError("home score cannot be negative")
    if match.away_score is not None and match.away_score < 0:
        raise ValueError("away score cannot be negative")


def upsert_records(
    conn: sqlite3.Connection,
    leagues: Iterable[LeagueRecord],
    teams: Iterable[TeamRecord],
    matches: Iterable[MatchRecord],
    provider: str,
) -> int:
    now = utc_now()
    count = 0
    with conn:
        for league in leagues:
            conn.execute(
                """INSERT INTO leagues(league_key,name,country,updated_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(league_key) DO UPDATE SET
                     name=excluded.name, country=excluded.country, updated_at=excluded.updated_at""",
                (league.key, league.name, league.country, now),
            )

        for team in teams:
            if not team.key or not team.name or not team.league_key:
                raise ValueError("team requires key, name and league")
            conn.execute(
                """INSERT INTO teams(team_key,league_key,name,updated_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(league_key,team_key) DO UPDATE SET
                     name=excluded.name, updated_at=excluded.updated_at""",
                (team.key, team.league_key, team.name, now),
            )

        for match in matches:
            _validate_match(match)
            conn.execute(
                """INSERT INTO matches(
                       match_key,provider,source_id,league_key,season,kickoff_utc,
                       home_team_key,away_team_key,status,home_score,away_score,
                       first_seen_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(match_key) DO UPDATE SET
                     source_id=COALESCE(excluded.source_id,matches.source_id),
                     status=CASE WHEN excluded.status='finished' THEN 'finished' ELSE excluded.status END,
                     home_score=COALESCE(excluded.home_score,matches.home_score),
                     away_score=COALESCE(excluded.away_score,matches.away_score),
                     updated_at=excluded.updated_at""",
                (
                    match.match_key,
                    match.provider,
                    match.source_id,
                    match.league_key,
                    match.season,
                    match.kickoff_utc,
                    match.home_team,
                    match.away_team,
                    match.status,
                    match.home_score,
                    match.away_score,
                    now,
                    now,
                ),
            )
            count += 1

        conn.execute(
            """INSERT INTO source_updates(provider,last_success_at,last_attempt_at,records_seen,records_upserted,error)
               VALUES(?,?,?,?,?,NULL)
               ON CONFLICT(provider) DO UPDATE SET
                 last_success_at=excluded.last_success_at,
                 last_attempt_at=excluded.last_attempt_at,
                 records_seen=excluded.records_seen,
                 records_upserted=excluded.records_upserted,
                 error=NULL""",
            (provider, now, now, count, count),
        )
    return count


def record_source_error(conn: sqlite3.Connection, provider: str, error: str, records_seen: int = 0) -> None:
    now = utc_now()
    with conn:
        conn.execute(
            """INSERT INTO source_updates(provider,last_success_at,last_attempt_at,records_seen,records_upserted,error)
               VALUES(?,NULL,?,?,0,?)
               ON CONFLICT(provider) DO UPDATE SET
                 last_attempt_at=excluded.last_attempt_at,
                 records_seen=excluded.records_seen,
                 error=excluded.error""",
            (provider, now, now, records_seen, error[:1000]),
        )


def summary(conn: sqlite3.Connection) -> dict:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    latest = conn.execute("SELECT MAX(kickoff_utc) FROM matches WHERE status='finished'").fetchone()[0]
    upcoming = conn.execute("SELECT MIN(kickoff_utc) FROM matches WHERE status='scheduled'").fetchone()[0]
    return {
        "leagues": one("SELECT COUNT(*) FROM leagues"),
        "teams": one("SELECT COUNT(*) FROM teams"),
        "matches": one("SELECT COUNT(*) FROM matches"),
        "finished_matches": one("SELECT COUNT(*) FROM matches WHERE status='finished'"),
        "upcoming_matches": one("SELECT COUNT(*) FROM matches WHERE status='scheduled'"),
        "newest_finished_match": latest,
        "next_scheduled_match": upcoming,
    }
