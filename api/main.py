"""Fovra V1 FastAPI API.

The API reads persisted canonical data and predictions from Supabase. It never
fetches football data on frontend requests and never exposes the service key.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.repository import FovraRepository

logger = logging.getLogger("fovra.api")
app = FastAPI(title="Fovra Football Intelligence API", version="1.0.0")
origins = [x.strip() for x in os.getenv("FOVRA_CORS_ORIGINS", "http://localhost:8000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET"], allow_headers=["*"])
repo: FovraRepository | None = None


def get_repo() -> FovraRepository:
    global repo
    if repo is None:
        repo = FovraRepository()
    return repo


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        return {"status": "ok", **get_repo().health()}
    except Exception as exc:
        logger.exception("health check failed")
        return {"status": "degraded", "database": "unavailable", "error": str(exc)}


@app.get("/api/v1/matches/today")
def todays_matches(limit: int = Query(100, ge=1, le=200)) -> list[dict[str, Any]]:
    return get_repo().matches_today(limit)


@app.get("/api/v1/matches/upcoming")
def upcoming_matches(limit: int = Query(100, ge=1, le=200)) -> list[dict[str, Any]]:
    return get_repo().upcoming(limit)


@app.get("/api/v1/matches/{canonical_key}")
def match_details(canonical_key: str) -> dict[str, Any]:
    match = get_repo().match(canonical_key)
    if not match:
        raise HTTPException(404, "Match not found")
    prediction = get_repo().predictions(canonical_key, 1)
    return {"match": match, "prediction": prediction[0] if prediction else None}


@app.get("/api/v1/matches/{canonical_key}/prediction")
def match_prediction(canonical_key: str) -> dict[str, Any]:
    prediction = get_repo().predictions(canonical_key, 1)
    if not prediction:
        raise HTTPException(404, "No persisted prediction is available for this match")
    return prediction[0]


@app.get("/api/v1/predictions")
def predictions(limit: int = Query(100, ge=1, le=200)) -> list[dict[str, Any]]:
    return get_repo().predictions(limit=limit)


@app.get("/api/v1/best-picks")
def best_picks(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
    rows = get_repo().predictions(limit=200)
    return sorted(rows, key=lambda x: float(x.get("confidence") or 0), reverse=True)[:limit]


@app.get("/api/v1/prediction-archive")
def prediction_archive(limit: int = Query(100, ge=1, le=200)) -> list[dict[str, Any]]:
    return get_repo().archive(limit)


@app.get("/api/v1/teams")
def teams(league: str | None = None) -> list[dict[str, Any]]:
    return get_repo().teams(league)


@app.get("/api/v1/teams/{team_key}/form")
def team_form(team_key: str, limit: int = Query(5, ge=1, le=20)) -> list[dict[str, Any]]:
    return get_repo().team_matches(team_key, max(limit, 20))[:limit]


@app.get("/api/v1/teams/{team_key}/h2h/{opponent_key}")
def team_h2h(team_key: str, opponent_key: str, limit: int = Query(10, ge=1, le=50)) -> list[dict[str, Any]]:
    return get_repo().h2h(team_key, opponent_key, limit)


@app.get("/api/v1/leagues")
def leagues() -> list[dict[str, Any]]:
    return get_repo().leagues()


@app.get("/api/v1/leagues/{league_key}/standings")
def standings(league_key: str) -> list[dict[str, Any]]:
    return get_repo().standings(league_key)


@app.get("/api/v1/leagues/{league_key}/fixtures")
def fixtures(league_key: str, limit: int = Query(100, ge=1, le=200)) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    return get_repo()._select("matches", params={"league_canonical_key": f"eq.{league_key}", "kickoff_at": f"gte.{now}", "order": "kickoff_at.asc", "limit": str(limit)})


@app.get("/api/v1/data-freshness")
def data_freshness() -> list[dict[str, Any]]:
    return get_repo().freshness()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "fovra", "version": "1.0.0", "app": "/app/"}


app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")
