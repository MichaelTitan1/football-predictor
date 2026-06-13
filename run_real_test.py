#!/usr/bin/env python3
"""
run_real_test.py

Strict real-run script. Requirements enforced:
- No simulation or fallbacks
- Either load existing model at models/football_model.cbm OR train from scratch using data/processed/clean_dataset.csv
- Use advanced features and model prediction only (no heuristics)
- Fail fast: any missing dependency, missing data, missing model, or prediction error raises and stops execution
- Output: exactly one JSON object to stdout with keys: home_win, draw, away_win, confidence, risk

Usage: python run_real_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run from repository root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

# Strict behavior: do not catch exceptions except to re-raise
# Import pipeline components
from src.models.train_advanced import train_advanced_model
from src.features.advanced_features import build_advanced_features
from src.prediction.engine import load_prediction_model, prepare_match_features, predict_match


CLEAN_PATH = Path("data/processed/clean_dataset.csv")
MODEL_PATH = Path("models/football_model.cbm")

HOME_TEAM = "Arsenal"
AWAY_TEAM = "Chelsea"

# Fail fast checks
if not CLEAN_PATH.exists():
    raise FileNotFoundError(f"Required dataset missing at {CLEAN_PATH}")

# Load clean dataset
df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"]) if CLEAN_PATH.exists() else None
if df is None or df.empty:
    raise RuntimeError(f"Clean dataset at {CLEAN_PATH} is empty or could not be loaded")

# Basic schema validation
required_cols = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
missing = required_cols - set(df.columns)
if missing:
    raise RuntimeError(f"Clean dataset is missing required columns: {missing}")

# Determine model: load or train
model = None
if MODEL_PATH.exists():
    # Load real model from disk
    model = load_prediction_model(str(MODEL_PATH))
else:
    # Train a real model from scratch using train_advanced_model
    # This will raise if dependencies missing or training fails
    train_res = train_advanced_model(df)
    # Verify training produced a saved model file
    mp = Path(train_res.get("model_path")) if isinstance(train_res, dict) else None
    if mp is None or not mp.exists():
        raise RuntimeError("Training completed but model file not found or not saved")
    # Load the saved model from disk to ensure consistency
    model = load_prediction_model(str(mp))

# Ensure model is present
if model is None:
    raise RuntimeError("Model not available after loading or training")

# Build advanced features (deterministic, no leakage)
adv = build_advanced_features(df)
if adv is None or adv.empty:
    raise RuntimeError("Advanced feature engineering produced empty dataset")

# Prepare feature row using model and advanced features
feature_row = prepare_match_features(HOME_TEAM, AWAY_TEAM, adv, model)
if feature_row is None or not hasattr(feature_row, "shape") or feature_row.shape[0] != 1:
    raise RuntimeError("prepare_match_features failed to produce a single-row feature vector")

# Predict using real model and feature_row
output = predict_match(model, feature_row, adv)
if not isinstance(output, dict):
    raise RuntimeError("predict_match did not return expected dict output")

# Extract required fields
try:
    main = output["main_result"]
    home_win = float(main["home_win"]) if "home_win" in main else None
    draw = float(main["draw"]) if "draw" in main else None
    away_win = float(main["away_win"]) if "away_win" in main else None
    confidence = float(output.get("confidence_score") if "confidence_score" in output else output.get("confidence", None))
    risk = str(output.get("risk_level") if "risk_level" in output else output.get("risk", None))
except Exception as e:
    raise RuntimeError(f"Prediction output missing required fields: {e}")

# Final strict validation: no None values allowed
if any(v is None for v in (home_win, draw, away_win, confidence, risk)):
    raise RuntimeError("Prediction contained None or missing values; aborting")

result = {
    "home_win": home_win,
    "draw": draw,
    "away_win": away_win,
    "confidence": confidence,
    "risk": risk,
}

# Print exactly one JSON object to stdout
print(json.dumps(result))
