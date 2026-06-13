"""
prediction_pipeline.py

Unified prediction pipeline that ties together advanced feature engineering,
ensemble prediction, probability calibration, and a filter engine for high-confidence picks.

Public API:
- predict_match_pipeline(home_team: str, away_team: str, df: pd.DataFrame) -> dict

Behavior:
1. Builds prior-to-match advanced features using src.features.advanced_features.build_advanced_features
   by appending a placeholder match row at a future date (so features are computed using only past matches).
2. Loads ensemble models from the default model directory and runs predict_ensemble to get raw probabilities.
3. Attempts to load a saved calibration mapping (models/calibration_*.pkl). If found, applies calibration; otherwise leaves probabilities unchanged.
4. Computes confidence (max probability) and risk level; applies a filter that only returns best picks when confidence >= threshold.
5. Returns a structured result containing match info, final_prediction, confidence, risk_level, and best_picks.

Notes:
- This function is the single source of truth for producing match predictions in production.
- It is deterministic and logs each step.
- It is defensive: handles missing models, missing calibrator, and missing data safely.

"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.advanced_features import build_advanced_features
from src.models import ensemble_model
from src.models import calibration as calibration_module

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(h)
LOGGER.setLevel(logging.INFO)


MODEL_DIR = os.environ.get("MODEL_DIR", "models")
CALIBRATION_PATHS = [
    Path(MODEL_DIR) / "calibration_platt.pkl",
    Path(MODEL_DIR) / "calibration_isotonic.pkl",
]
# Confidence threshold for a 'best pick'
BEST_PICK_CONFIDENCE_THRESHOLD = float(os.environ.get("BEST_PICK_CONF_THRESH", 0.70))


def _load_calibrator() -> calibration_module.CalibrationModel | None:
    """Attempt to load a saved calibrator from MODEL_DIR. Returns CalibrationModel or None."""
    for p in CALIBRATION_PATHS:
        try:
            if p.exists():
                LOGGER.info("Loading calibrator from %s", p)
                with open(p, "rb") as fh:
                    cal = pickle.load(fh)
                if isinstance(cal, calibration_module.CalibrationModel):
                    return cal
                else:
                    LOGGER.warning("Loaded object from %s is not a CalibrationModel", p)
        except Exception as e:
            LOGGER.exception("Failed to load calibrator %s: %s", p, e)
    LOGGER.info("No calibrator found in %s", MODEL_DIR)
    return None


def _prepare_single_match_features(home_team: str, away_team: str, df: pd.DataFrame) -> pd.DataFrame:
    """Construct a single-row DataFrame of advanced features for the provided match.

    Method:
    - Create a placeholder match row with Date = max(df.Date) + 1 day (or today if no Date present)
    - FTHG, FTAG, FTR set to NaN so feature builder treats it as a future/unplayed match
    - Append to historical df and call build_advanced_features
    - Extract the last row's advanced feature columns (excluding identifiers)
    """
    # Validate df columns minimally
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Input dataframe must contain columns: {sorted(required)}")

    df_copy = df.copy().reset_index(drop=True)
    # Determine next date
    try:
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])
        if df_copy["Date"].notna().any():
            next_date = df_copy["Date"].max() + timedelta(days=1)
        else:
            next_date = pd.Timestamp.now().normalize()
    except Exception:
        next_date = pd.Timestamp.now().normalize()

    placeholder = {
        "Date": next_date,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "FTHG": np.nan,
        "FTAG": np.nan,
        "FTR": np.nan,
    }
    # Preserve optional League if present
    if "League" in df_copy.columns:
        placeholder["League"] = df_copy["League"].mode().iloc[0] if not df_copy["League"].mode().empty else ""

    df_app = pd.concat([df_copy, pd.DataFrame([placeholder])], ignore_index=True)

    LOGGER.info("Computing advanced features with placeholder match for %s vs %s on %s", home_team, away_team, str(next_date.date()))
    adv = build_advanced_features(df_app)

    # Exclude raw identifiers
    exclude = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"}
    feature_cols = [c for c in adv.columns if c not in exclude]

    single_row = adv.iloc[[-1]][feature_cols].copy()
    # Ensure numeric columns are filled with medians for safety
    for c in single_row.columns:
        if pd.api.types.is_numeric_dtype(single_row[c]):
            if pd.isna(single_row[c].iloc[0]):
                # fallback median from historical rows
                hist = adv.loc[:-2, c] if adv.shape[0] > 1 else pd.Series([])
                med = hist.median() if not hist.empty else 0.0
                single_row[c] = single_row[c].fillna(med)
        else:
            single_row[c] = single_row[c].fillna("UNK")

    return single_row


def _compute_risk_level(confidence: float) -> str:
    """Map confidence to risk level. Higher confidence -> lower risk."""
    if confidence >= 0.75:
        return "LOW"
    if confidence >= 0.6:
        return "MEDIUM"
    return "HIGH"


def predict_match_pipeline(home_team: str, away_team: str, df: pd.DataFrame) -> Dict:
    """Main pipeline entrypoint.

    Args:
      home_team: Home team name (must match naming in historical df)
      away_team: Away team name
      df: historical merged DataFrame used to compute prior features (must include standard columns)

    Returns: structured dict with prediction, confidence, risk_level, and best_picks
    """
    LOGGER.info("Starting prediction pipeline for %s vs %s", home_team, away_team)

    # 1) Build features for the single match
    try:
        X_row = _prepare_single_match_features(home_team, away_team, df)
    except Exception as e:
        LOGGER.exception("Failed to prepare features: %s", e)
        # Return safe empty prediction
        return {
            "match": {"home_team": home_team, "away_team": away_team},
            "final_prediction": {"home_win": 0.33, "draw": 0.34, "away_win": 0.33},
            "confidence": 0.34,
            "risk_level": "HIGH",
            "best_picks": [],
        }

    # 2) Load ensemble models
    try:
        models = ensemble_model.load_ensemble_models(MODEL_DIR)
        LOGGER.info("Loaded ensemble models: keys=%s", [k for k, v in models.items() if v is not None])
    except Exception as e:
        LOGGER.exception("Failed to load ensemble models: %s", e)
        models = {}

    # 3) Run ensemble prediction
    try:
        probs_dict = ensemble_model.predict_ensemble(models, X_row)
        raw_probs = np.array([probs_dict["home_win"], probs_dict["draw"], probs_dict["away_win"]]).reshape(1, -1)
        LOGGER.info("Raw ensemble probabilities: %s", raw_probs.flatten().tolist())
    except Exception as e:
        LOGGER.exception("Ensemble prediction failed: %s", e)
        # fallback uniform
        raw_probs = np.array([[0.3333333, 0.3333333, 0.3333333]])

    # 4) Load calibrator if available and apply
    cal = _load_calibrator()
    if cal is not None:
        try:
            calibrated = calibration_module.calibrated_predict(cal, raw_probs)
            calibrated = np.asarray(calibrated)
            LOGGER.info("Applied calibration to probabilities: %s", calibrated.flatten().tolist())
        except Exception as e:
            LOGGER.exception("Calibration apply failed: %s", e)
            calibrated = raw_probs
    else:
        calibrated = raw_probs

    # Ensure sums to 1 and numeric stability
    calibrated = np.clip(calibrated.astype(float), 1e-8, 1.0)
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

    # 5) Compute confidence and risk
    confidence = float(np.max(calibrated))
    risk_level = _compute_risk_level(confidence)

    # 6) Apply filter engine: only return best_picks if confidence above threshold
    best_picks: List[Dict[str, float]] = []
    top_idx = int(np.argmax(calibrated[0]))
    markets = ["home_win", "draw", "away_win"]
    top_market = markets[top_idx]
    top_prob = float(calibrated[0, top_idx])
    if top_prob >= BEST_PICK_CONFIDENCE_THRESHOLD:
        best_picks.append({"market": top_market, "probability": top_prob})

    result = {
        "match": {"home_team": home_team, "away_team": away_team},
        "final_prediction": {"home_win": float(calibrated[0, 0]), "draw": float(calibrated[0, 1]), "away_win": float(calibrated[0, 2])},
        "confidence": confidence,
        "risk_level": risk_level,
        "best_picks": best_picks,
    }

    LOGGER.info("Prediction pipeline complete for %s vs %s: confidence=%.3f risk=%s", home_team, away_team, confidence, risk_level)
    return result


# Minimal demo run when executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: require merged dataset at data/processed/merged_dataset.csv
    merged = Path("data/processed/merged_dataset.csv")
    if not merged.exists():
        LOGGER.error("Merged dataset not found at data/processed/merged_dataset.csv — cannot run demo")
        raise SystemExit(1)
    df_hist = pd.read_csv(merged, parse_dates=["Date"])
    out = predict_match_pipeline("Sample FC", "Example United", df_hist)
    import json

    print(json.dumps(out, indent=2))
