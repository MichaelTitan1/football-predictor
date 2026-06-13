"""
football_ai_engine.py

Unified football AI engine: single entrypoint for training, prediction and evaluation.

Core functions
- run_full_training(): orchestrate clean dataset build, feature engineering, ensemble training, calibration, and save final artifact.
- predict_match(home_team, away_team): produce calibrated probabilities and confidence for a single match.
- filter_high_confidence_predictions(preds, threshold=0.65): keep only predictions above threshold.

Engineering rules implemented
- Uses only data/processed/clean_dataset.csv as the single training input.
- Never bypasses validation pipeline (reads PROJECT_STATUS.md via clean_training_pipeline behavior).
- Logs every step and fails early if data quality insufficient.
- Saves models and calibration objects under models/final/ for reproducibility.

Notes
- This module uses best-effort imports for optional components:
  - src.pipeline.clean_training_pipeline (required)
  - src.features.advanced_features (preferred)
  - src.features.prediction_features (preferred for single-match feature building)
  - src.models.ensemble_model (required for ensemble training/prediction)
  - sklearn for calibration if available
- If a required component is missing at prediction time, predict_match will raise an informative error.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Paths
CLEAN_PATH = Path("data/processed/clean_dataset.csv")
FINAL_MODEL_DIR = Path("models/final")
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATOR_PATH = FINAL_MODEL_DIR / "calibrator.pkl"
ENSEMBLE_MODEL_DIR = Path("models")  # ensemble_model will save models here by default

# Best-effort imports
try:
    from src.pipeline.clean_training_pipeline import build_clean_dataset, run_training_pipeline  # type: ignore
except Exception:
    build_clean_dataset = None
    run_training_pipeline = None

try:
    from src.features.advanced_features import build_advanced_features  # type: ignore
except Exception:
    build_advanced_features = None

# prediction_features is optional and preferred for single-match feature creation
try:
    from src.features.prediction_features import build_features_for_match  # type: ignore
except Exception:
    build_features_for_match = None

try:
    from src.models import ensemble_model as ensemble_model_module  # type: ignore
except Exception:
    ensemble_model_module = None

# Calibration utilities
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
    train_test_split = None


def _ensure_clean_dataset_exists() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        logger.error("Clean dataset not found at %s. Run the training pipeline first.", CLEAN_PATH)
        raise FileNotFoundError("Clean dataset not found. Run run_full_training() to build data/processed/clean_dataset.csv")
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"]) if CLEAN_PATH.exists() else pd.DataFrame()
    return df


def _save_calibrator(calibrator: Any) -> None:
    try:
        with open(CALIBRATOR_PATH, "wb") as fh:
            pickle.dump(calibrator, fh)
        logger.info("Saved calibrator to %s", CALIBRATOR_PATH)
    except Exception as e:
        logger.exception("Failed to save calibrator: %s", e)


def _load_calibrator() -> Optional[Any]:
    if not CALIBRATOR_PATH.exists():
        return None
    try:
        with open(CALIBRATOR_PATH, "rb") as fh:
            obj = pickle.load(fh)
        return obj
    except Exception:
        logger.exception("Failed to load calibrator from %s", CALIBRATOR_PATH)
        return None


def run_full_training(min_league_score: int = 70, min_rows: int = 1000) -> Dict[str, Any]:
    """Orchestrate full training:
    1. Build clean dataset via clean_training_pipeline.build_clean_dataset
    2. Run advanced feature engineering
    3. Train ensemble_model.train_ensemble on features
    4. Fit a simple calibrator on validation predictions
    5. Save models + calibrator under models/final

    Returns a dict with training artifacts info.
    """
    logger.info("Starting full training run")

    # Step 1: Build clean dataset (this also validates PROJECT_STATUS.md via the pipeline)
    if build_clean_dataset is None:
        logger.error("clean_training_pipeline.build_clean_dataset is not available. Aborting")
        raise RuntimeError("Missing pipeline module: src.pipeline.clean_training_pipeline")

    clean_df = build_clean_dataset(min_league_score=min_league_score)
    if clean_df is None or clean_df.empty or len(clean_df) < min_rows:
        logger.error("DATA NOT READY FOR TRAINING")
        raise RuntimeError("DATA NOT READY FOR TRAINING")

    logger.info("Clean dataset prepared (%d rows). Proceeding to feature engineering", len(clean_df))

    # Step 2: Feature engineering
    if build_advanced_features is None:
        logger.error("Feature engineering module src.features.advanced_features not available. Aborting")
        raise RuntimeError("Missing feature engineering module")

    features_df = build_advanced_features(clean_df)
    if features_df is None or features_df.empty:
        logger.error("Feature engineering produced no data. Aborting")
        raise RuntimeError("Feature engineering failure")

    logger.info("Feature engineering complete (%d rows, %d cols)", len(features_df), features_df.shape[1])

    # Step 3: Train ensemble
    if ensemble_model_module is None:
        logger.error("Ensemble model module not available (src.models.ensemble_model). Aborting")
        raise RuntimeError("Missing ensemble model module")

    logger.info("Training ensemble models (this may take significant time and resources)")
    train_res = ensemble_model_module.train_ensemble(features_df, model_dir=str(ENSEMBLE_MODEL_DIR))
    logger.info("Ensemble training finished. Saved model paths: %s", train_res.get("paths"))

    # Step 4: Calibration
    # We'll create a simple calibrator using validation set predictions if possible
    calibrator = None
    try:
        # Attempt to load trained models
        models = ensemble_model_module.load_ensemble_models(str(ENSEMBLE_MODEL_DIR))
        # Create validation split from features_df for calibration
        if train_test_split is not None:
            # Prepare X/y using ensemble_model helper if available, else attempt to reuse features_df with FTR target
            if "FTR" not in features_df.columns:
                logger.warning("Features dataframe missing FTR target; skipping calibration")
            else:
                X = features_df.drop(columns=[c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"] if c in features_df.columns], errors='ignore')
                y = features_df["FTR"].astype(str)
                X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
                # Get ensemble predictions (probabilities) on X_cal
                try:
                    preds = ensemble_model_module.predict_ensemble(models, X_cal)
                    # preds is list of dicts with home_win/draw/away_win and model_agreement_score
                    # We will calibrate the max-probability per-row using logistic regression on whether the predicted top label was correct
                    top_probs = []
                    top_correct = []
                    for i, p in enumerate(preds if isinstance(preds, list) else [preds]):
                        probs = np.array([p["home_win"], p["draw"], p["away_win"]])
                        top_idx = int(np.argmax(probs))
                        top_label = ["H", "D", "A"][top_idx]
                        top_prob = float(probs[top_idx])
                        true = 1 if str(y_cal.iloc[i]) == top_label else 0
                        top_probs.append([top_prob])
                        top_correct.append(true)
                    # Fit a simple logistic regressor mapping top_prob -> P(correct)
                    if LogisticRegression is not None and len(top_probs) > 10:
                        lr = LogisticRegression()
                        lr.fit(np.array(top_probs), np.array(top_correct))
                        calibrator = {"type": "logistic", "model": lr}
                        logger.info("Fitted logistic calibrator on validation set")
                    else:
                        logger.info("Not enough data or sklearn missing for calibration; skipping")
                except Exception as e:
                    logger.exception("Calibration prediction step failed: %s", e)
        else:
            logger.info("sklearn.train_test_split not available; skipping calibration")
    except Exception as e:
        logger.exception("Calibration step failed: %s", e)

    # Step 5: Save calibrator and finalize
    if calibrator is not None:
        _save_calibrator(calibrator)
    else:
        logger.info("No calibrator produced; continuing without calibration")

    # Copy ensemble model files to final dir (for reproducibility we keep original trained models in models/)
    try:
        # Save a small metadata file
        meta = {"ensemble_paths": train_res.get("paths"), "weights": train_res.get("weights"), "metrics": train_res.get("metrics")}
        with open(FINAL_MODEL_DIR / "metadata.pkl", "wb") as fh:
            pickle.dump(meta, fh)
        logger.info("Saved training metadata to %s", FINAL_MODEL_DIR / "metadata.pkl")
    except Exception:
        logger.exception("Failed to save training metadata")

    logger.info("Full training run completed successfully")
    return {"train_res": train_res, "calibrator_saved": calibrator is not None}


def _load_models_for_predict() -> Dict[str, Optional[object]]:
    if ensemble_model_module is None:
        raise RuntimeError("Ensemble model module not available")
    models = ensemble_model_module.load_ensemble_models(str(ENSEMBLE_MODEL_DIR))
    return models


def _apply_calibration(raw_probs: Dict[str, float], calibrator: Optional[Any]) -> Dict[str, float]:
    """Apply calibration to probabilities. Expects raw_probs with keys home_win/draw/away_win.

    If calibrator is a logistic model mapping top_prob->P(correct), we scale the top_prob and renormalize.
    Otherwise return raw_probs unchanged.
    """
    if calibrator is None:
        return raw_probs
    try:
        if isinstance(calibrator, dict) and calibrator.get("type") == "logistic":
            lr = calibrator.get("model")
            probs = np.array([raw_probs["home_win"], raw_probs["draw"], raw_probs["away_win"]])
            top_idx = int(np.argmax(probs))
            top_prob = probs[top_idx]
            p_correct = float(lr.predict_proba([[top_prob]])[0, 1])
            # Scale top probability towards p_correct while keeping ratios: new_top = p_correct, distribute remaining mass proportionally
            other_idx = [i for i in range(3) if i != top_idx]
            remaining = 1.0 - p_correct
            other_sum = probs[other_idx].sum() if probs[other_idx].sum() > 0 else 1.0
            new_probs = np.zeros(3)
            new_probs[top_idx] = p_correct
            for oi in other_idx:
                new_probs[oi] = remaining * (probs[oi] / other_sum)
            return {"home_win": float(new_probs[0]), "draw": float(new_probs[1]), "away_win": float(new_probs[2])}
    except Exception:
        logger.exception("Calibration application failed; returning raw probabilities")
    return raw_probs


def _risk_level_from_confidence(conf: float) -> str:
    if conf >= 0.8:
        return "LOW"
    if conf >= 0.65:
        return "MEDIUM"
    return "HIGH"


def predict_match(home_team: str, away_team: str, date: Optional[str] = None) -> Dict[str, object]:
    """Predict probabilities for a single match.

    Steps:
    - Load clean_dataset.csv (training data only) to access team history and global stats
    - Use src.features.prediction_features.build_features_for_match if available to construct the feature row
    - Otherwise, attempt to raise informative error (preparing proper single-match features requires project-specific logic)
    - Run ensemble_model.predict_ensemble to get raw probabilities and agreement
    - Apply calibrator if available
    - Return structured predictions including confidence and risk_level
    """
    logger.info("Predicting match: %s vs %s", home_team, away_team)

    # Ensure clean dataset exists
    _ = _ensure_clean_dataset_exists()

    # Build feature row for the match
    if build_features_for_match is None:
        logger.error("No single-match feature builder found (src.features.prediction_features.build_features_for_match).")
        raise RuntimeError("Prediction feature builder not available. Implement src.features.prediction_features.build_features_for_match")

    # build_features_for_match should accept (clean_df, home_team, away_team, date) and return a single-row DataFrame of features
    try:
        clean_df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"]) if CLEAN_PATH.exists() else pd.DataFrame()
        feature_row = build_features_for_match(clean_df, home_team, away_team, date=date)
    except Exception as e:
        logger.exception("Feature building for match failed: %s", e)
        raise

    if feature_row is None or feature_row.empty:
        logger.error("Feature builder returned no features for match %s vs %s", home_team, away_team)
        raise RuntimeError("No features available for prediction")

    # Load ensemble models
    models = _load_models_for_predict()

    # Predict ensemble probabilities
    try:
        pred = ensemble_model_module.predict_ensemble(models, feature_row)
        # predict_ensemble returns dict for single-row
        raw_probs = {
            "home_win": float(pred.get("home_win", 0.0)),
            "draw": float(pred.get("draw", 0.0)),
            "away_win": float(pred.get("away_win", 0.0)),
        }
        agreement = float(pred.get("model_agreement_score", 0.0))
    except Exception as e:
        logger.exception("Ensemble prediction failed: %s", e)
        raise

    # Apply calibrator if present
    calibrator = _load_calibrator()
    calibrated = _apply_calibration(raw_probs, calibrator)

    # Compute confidence: use max prob * agreement as conservative measure
    maxp = max([calibrated["home_win"], calibrated["draw"], calibrated["away_win"]])
    confidence = float(maxp * (0.5 + 0.5 * agreement)) if agreement is not None else float(maxp)
    # ensure 0-1
    confidence = max(0.0, min(1.0, confidence))

    risk = _risk_level_from_confidence(confidence)

    out = {
        "home_win": calibrated["home_win"],
        "draw": calibrated["draw"],
        "away_win": calibrated["away_win"],
        "confidence": confidence,
        "risk_level": risk,
    }
    logger.info("Prediction result: %s", out)
    return out


def filter_high_confidence_predictions(predictions: List[Dict[str, Any]], threshold: float = 0.65) -> List[Dict[str, Any]]:
    """Return only predictions where confidence > threshold."""
    filtered = [p for p in predictions if float(p.get("confidence", 0.0)) > float(threshold)]
    logger.info("Filtered %d -> %d predictions with threshold %s", len(predictions), len(filtered), threshold)
    return filtered


if __name__ == "__main__":
    # Example quick-run: attempt to run full training if executed directly
    try:
        run_full_training()
    except Exception as e:
        logger.exception("run_full_training failed: %s", e)
