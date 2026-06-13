"""
main_pipeline.py

Unified single-entry orchestration for the football prediction system.

Public API:
- run_system(retrain: bool = False)
- predict_game(home_team: str, away_team: str) -> Dict
- get_system_status() -> Dict

Notes:
- Uses existing modules only: src.pipeline.clean_training_pipeline, src.features.advanced_features,
  src.features.prediction_features, src.models.train_advanced, src.prediction.engine
- Safe: checks for missing files and dependencies and raises informative errors.
- Deterministic: avoids randomness unless training is explicitly requested (train_advanced_model uses a seed).

This file is intended to be the single entrypoint for end-to-end runs and simple programmatic usage.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Local imports (existing modules)
from src.pipeline.clean_training_pipeline import build_clean_dataset
from src.features.prediction_features import build_features_for_match
from src.models.train_advanced import train_advanced_model
from src.prediction.engine import load_prediction_model, prepare_match_features

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# Configurable paths
CLEAN_DATA_PATH = Path(os.environ.get("CLEAN_DATA_PATH", "data/processed/clean_dataset.csv"))
MODEL_PATH = Path(os.environ.get("FOOTBALL_MODEL_PATH", "models/football_model.cbm"))
LAST_TRAIN_INFO_PATH = Path("models/last_train_info.txt")


class SystemState:
    """Simple in-memory state cache."""

    def __init__(self):
        self.dataset: Optional[pd.DataFrame] = None
        self.model = None
        self.last_training: Optional[datetime] = None


_STATE = SystemState()


def _load_dataset() -> pd.DataFrame:
    """Load or build the clean dataset. If CLEAN_DATA_PATH exists, load it; otherwise attempt to build via pipeline."""
    if CLEAN_DATA_PATH.exists():
        logger.info("Loading existing clean dataset from %s", CLEAN_DATA_PATH)
        df = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["Date"]) if CLEAN_DATA_PATH.exists() else pd.DataFrame()
        _STATE.dataset = df
        return df

    logger.info("Clean dataset not found at %s — attempting to build via pipeline", CLEAN_DATA_PATH)
    df = build_clean_dataset()
    if df is None or df.empty:
        raise RuntimeError("Clean dataset empty after pipeline. Ensure data/raw has valid CSVs and PROJECT_STATUS.md is configured.")
    # Ensure processed path exists
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    _STATE.dataset = df
    return df


def _load_model() -> Optional[object]:
    """Load trained model if available. Returns model or None."""
    if MODEL_PATH.exists():
        try:
            model = load_prediction_model(str(MODEL_PATH))
            _STATE.model = model
            logger.info("Loaded model from %s", MODEL_PATH)
            return model
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            _STATE.model = None
            return None
    logger.info("Model not found at %s", MODEL_PATH)
    _STATE.model = None
    return None


def run_system(retrain: bool = False, *, train_kwargs: Optional[Dict] = None) -> Dict:
    """Run the full pipeline.

    Steps:
    1. Load or build clean dataset
    2. Validate dataset has required columns
    3. Run advanced feature engineering (implicitly via train_advanced_model or prepare_match_features as needed)
    4. Train model if retrain=True or model missing
    5. Load trained model into memory

    Returns a dict with status and artifact paths.
    """
    df = _load_dataset()

    # Basic validation
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Clean dataset missing required columns: {missing}")

    # Determine whether to train
    model_needed = retrain or not MODEL_PATH.exists()
    train_info = None
    if model_needed:
        logger.info("Starting training (retrain=%s)", retrain)
        train_kwargs = train_kwargs or {}
        # train_advanced_model returns a dict including 'model_path'
        res = train_advanced_model(df, **train_kwargs)
        # Save last training info
        _STATE.last_training = datetime.utcnow()
        try:
            LAST_TRAIN_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_TRAIN_INFO_PATH, "w", encoding="utf-8") as fh:
                fh.write(f"last_training={_STATE.last_training.isoformat()}\n")
                fh.write(f"model_path={res.get('model_path')}\n")
            logger.info("Wrote last training info to %s", LAST_TRAIN_INFO_PATH)
        except Exception:
            logger.exception("Failed to write last training info")
        train_info = res

    # Attempt to load model
    model = _load_model()

    return {"dataset_rows": len(df), "model_loaded": model is not None, "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None, "train_info": train_info}


def predict_game(home_team: str, away_team: str) -> Dict:
    """Produce a prediction for a single match using the loaded model and historical data.

    Returns dict: {home_win, draw, away_win, confidence, risk}
    - confidence: model's top probability
    - risk: heuristic based on agreement/confidence (LOW/MEDIUM/HIGH)
    """
    # Ensure dataset loaded
    df = _STATE.dataset if _STATE.dataset is not None else (_load_dataset())

    # Ensure model loaded
    model = _STATE.model if _STATE.model is not None else (_load_model())
    if model is None:
        raise RuntimeError("Model not loaded. Run run_system(retrain=True) or ensure a trained model exists at FOOTBALL_MODEL_PATH")

    # Build single-match features using historical clean dataset
    features = build_features_for_match(home_team, away_team, df)

    # Convert to DataFrame matching model expectations using prepare_match_features if available
    # We'll try to use src.prediction.engine.prepare_match_features, which expects the full feature_data (advanced features).
    try:
        # First compute advanced features for dataset so prepare_match_features can extract required columns
        from src.features.advanced_features import build_advanced_features

        adv = build_advanced_features(df)
        feature_row = prepare_match_features(home_team, away_team, adv, model)
    except Exception as e:
        # Fallback: use our lightweight features as feature_row with names matching what the model may expect
        logger.warning("prepare_match_features failed (%s). Falling back to minimal feature set.", e)
        feature_row = pd.DataFrame([features])

    # Ensure single-row DataFrame
    if isinstance(feature_row, pd.Series):
        feature_row = feature_row.to_frame().T

    # Ensure numeric coercion
    for c in feature_row.columns:
        if pd.api.types.is_numeric_dtype(feature_row[c]):
            feature_row[c] = pd.to_numeric(feature_row[c], errors="coerce").fillna(0.0)

    # Predict using model
    try:
        # CatBoost model interface
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feature_row)
        else:
            # Some wrappers return raw margin; try predict
            pred = model.predict(feature_row)
            # fallback create one-hot-ish probs
            probs = []
        # probs expected shape (n, k)
        import numpy as np

        probs = np.asarray(probs)
        if probs.ndim == 1:
            # binary case
            if probs.size == 2:
                # treat as two-class probabilities
                home_p = float(probs[0])
                draw_p = 0.0
                away_p = float(probs[1])
            else:
                home_p = float(probs[0])
                draw_p = 0.0
                away_p = 1.0 - home_p
        else:
            # attempt to map classes H/D/A; many models use order ['H','D','A'] or label indices
            # We'll try to get model.classes_ when available
            classes = None
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
            # default canonical
            canonical = ["H", "D", "A"]
            order = canonical
            if classes is not None:
                order = classes
            # create mapping from order to indices
            mapping = {lab: i for i, lab in enumerate(order)}
            def _get_idx(label):
                return mapping.get(label, None)

            row = probs[0]
            # pick indices
            h_idx = _get_idx("H")
            d_idx = _get_idx("D")
            a_idx = _get_idx("A")
            # If mapping is incomplete, fall back to 0/1/2
            if h_idx is None or d_idx is None or a_idx is None or max(h_idx,d_idx,a_idx) >= row.size:
                # fallback: take first three columns or pad
                padded = np.zeros(3)
                padded[: min(3, row.size)] = row[: min(3, row.size)]
                home_p, draw_p, away_p = float(padded[0]), float(padded[1]), float(padded[2])
            else:
                home_p = float(row[h_idx])
                draw_p = float(row[d_idx])
                away_p = float(row[a_idx])

        # clamp and normalize
        totals = home_p + draw_p + away_p
        if totals <= 0:
            home_p, draw_p, away_p = 0.33, 0.34, 0.33
        else:
            home_p, draw_p, away_p = home_p / totals, draw_p / totals, away_p / totals

        top = max(home_p, draw_p, away_p)
        confidence = float(top)
        # risk heuristic: high confidence => low risk, but if teams unfamiliar raise risk
        if confidence >= 0.70:
            risk = "LOW"
        elif confidence >= 0.55:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return {
            "home_win": home_p,
            "draw": draw_p,
            "away_win": away_p,
            "confidence": confidence,
            "risk": risk,
        }
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise RuntimeError(f"Prediction failed: {e}")


def get_system_status() -> Dict:
    """Return system status: dataset size, number of leagues, model loaded, last training date."""
    df = _STATE.dataset if _STATE.dataset is not None else (CLEAN_DATA_PATH.exists() and pd.read_csv(CLEAN_DATA_PATH, parse_dates=["Date"]))
    dataset_rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    leagues = 0
    if isinstance(df, pd.DataFrame) and "League" in df.columns:
        leagues = int(df["League"].nunique())
    model_loaded = _STATE.model is not None
    last_train = None
    if LAST_TRAIN_INFO_PATH.exists():
        try:
            with open(LAST_TRAIN_INFO_PATH, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            d = dict([l.split("=", 1) for l in lines if "=" in l])
            last_train = d.get("last_training")
        except Exception:
            last_train = None
    return {"dataset_rows": dataset_rows, "leagues": leagues, "model_loaded": model_loaded, "last_training": last_train}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running full system (this may train a model if none exists).")
    res = run_system(retrain=False)
    print("System run result:", res)
    # Example prediction
    try:
        pred = predict_game("Arsenal", "Chelsea")
        print("Sample prediction:", pred)
    except Exception as e:
        print("Prediction example failed:", e)
