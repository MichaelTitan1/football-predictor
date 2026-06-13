"""
api/main.py

FastAPI backend for the football prediction system (football-ai-system).

Endpoints:
- GET /health -> basic service and model status
- POST /predict -> single match full prediction
- POST /best-picks -> accept list of matches and return only high-confidence picks

Design:
- Loads model and feature dataset at startup and caches them on app.state
- Uses modules: src.prediction.engine (core intelligence), src.prediction.filter_engine (filter),
  data_loader.load_all_data, src.features.feature_engineer.build_features
- Defensive error handling and logging
- Structured JSON responses via Pydantic

Run with:
    uvicorn api.main:app --reload

"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Local imports from package
from data_loader import load_all_data
from src.features.feature_engineer import build_features
from src.prediction.engine import load_prediction_model, prepare_match_features, predict_match
from src.prediction.filter_engine import filter_predictions

logger = logging.getLogger("football_api")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

app = FastAPI(title="football-ai-system API", version="1.0")


MODEL_PATH = os.environ.get("FOOTBALL_MODEL_PATH", "models/football_model.cbm")


# Pydantic request/response models
class MatchRequest(BaseModel):
    home_team: str
    away_team: str


class BestPicksRequest(BaseModel):
    matches: List[MatchRequest]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    feature_rows: int


@app.on_event("startup")
async def startup_event() -> None:
    """Load model and feature dataset into app.state for reuse."""
    logger.info("Starting football-ai-system API startup: loading data and model")

    # Load raw data and build features (if available). Wrap in try/except to let service start even if data not present
    try:
        raw = load_all_data()
        features = build_features(raw)
        app.state.feature_data = features
        logger.info("Loaded feature_data with %d rows", len(features))
    except Exception as e:
        app.state.feature_data = None
        logger.exception("Failed to load feature data at startup: %s", e)

    # Load CatBoost model
    try:
        model = load_prediction_model(MODEL_PATH)
        app.state.model = model
        app.state.model_path = MODEL_PATH
        logger.info("Loaded model from %s", MODEL_PATH)
    except Exception as e:
        app.state.model = None
        app.state.model_path = None
        logger.exception("Failed to load model at startup: %s", e)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health, whether model and feature data are loaded."""
    model_loaded = app.state.model is not None
    feature_rows = len(app.state.feature_data) if getattr(app.state, "feature_data", None) is not None else 0
    status = "ok" if model_loaded and feature_rows > 0 else "degraded"
    resp = HealthResponse(status=status, model_loaded=model_loaded, model_path=getattr(app.state, "model_path", None), feature_rows=feature_rows)
    return resp


@app.post("/predict")
async def predict_endpoint(req: MatchRequest, request: Request) -> Dict[str, Any]:
    """Produce full prediction markets for a single match.

    Uses cached model and feature_data loaded at startup. Returns detailed markets produced by the core engine.
    """
    logger.info("/predict request from %s: %s vs %s", request.client.host if request.client else "unknown", req.home_team, req.away_team)

    if app.state.model is None:
        logger.error("Predict requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if app.state.feature_data is None:
        logger.error("Predict requested but feature data not loaded")
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    try:
        # Prepare feature row (handles unknown teams safely)
        feature_row = prepare_match_features(req.home_team, req.away_team, app.state.feature_data, model=app.state.model)
        # Predict full markets
        prediction = predict_match(app.state.model, feature_row, app.state.feature_data)
        return prediction
    except Exception as e:
        logger.exception("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/best-picks")
async def best_picks_endpoint(req: BestPicksRequest, request: Request) -> Dict[str, Any]:
    """Return only high-confidence picks for a list of matches.

    For each requested match, compute full prediction then filter with filter_engine.filter_predictions.
    Returns a list of match-level best picks and an aggregated best_picks list.
    """
    logger.info("/best-picks request from %s: %d matches", request.client.host if request.client else "unknown", len(req.matches))

    if app.state.model is None:
        logger.error("Best-picks requested but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    if app.state.feature_data is None:
        logger.error("Best-picks requested but feature data not loaded")
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    results = []
    aggregated_picks = []

    for m in req.matches:
        try:
            feature_row = prepare_match_features(m.home_team, m.away_team, app.state.feature_data, model=app.state.model)
            prediction = predict_match(app.state.model, feature_row, app.state.feature_data)
            # filter picks for this match
            filtered = filter_predictions(prediction, high_threshold=0.70, medium_threshold=0.55, include_medium=False)
            # attach filtered picks (may be empty)
            results.append({"match": {"home_team": m.home_team, "away_team": m.away_team}, "best_picks": filtered.get("best_picks", [])})
            # aggregate
            for pick in filtered.get("best_picks", []):
                aggregated_picks.append({"match": {"home_team": m.home_team, "away_team": m.away_team}, "market": pick["market"], "probability": pick["probability"], "confidence": pick["confidence"]})
        except Exception as e:
            logger.exception("Failed to compute best picks for %s vs %s: %s", m.home_team, m.away_team, e)
            # include an error entry for this match
            results.append({"match": {"home_team": m.home_team, "away_team": m.away_team}, "error": str(e)})

    # Sort aggregated picks by probability desc
    aggregated_picks = sorted(aggregated_picks, key=lambda x: x.get("probability", 0.0), reverse=True)

    return {"results": results, "aggregated_best_picks": aggregated_picks}


# Basic root path
@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "football-ai-system API - see /docs for usage"}
