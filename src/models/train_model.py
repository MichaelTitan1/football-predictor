"""
train_model.py

Production-ready training module for football match outcome prediction.

Provides:
- train_model(df: pd.DataFrame, *, model_path: str = "models/football_model.cbm", random_seed: int = 42, cat_features: Optional[list] = None)
    - Splits data (80/20), trains a CatBoostClassifier, evaluates on validation set (accuracy + classification report),
      then re-trains on full dataset and saves final model to `model_path`.
- load_model(path: str) -> CatBoostClassifier
- predict_match(model, features) -> Dict[str, float]  (probabilities for H, D, A)

Requirements:
- catboost
- scikit-learn
- pandas

The function is defensive: checks for required columns, handles missing values, logs progress, and ensures reproducibility.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Delayed imports for optional dependencies
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - handled at runtime
    CatBoostClassifier = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except Exception:  # pragma: no cover - handled at runtime
    train_test_split = None
    classification_report = None
    accuracy_score = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


REQUIRED_INPUT_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


def _check_deps():
    missing = []
    if CatBoostClassifier is None:
        missing.append("catboost")
    if train_test_split is None or classification_report is None or accuracy_score is None:
        missing.append("scikit-learn")
    if missing:
        raise ImportError(
            "Missing required packages: {}. Install with `pip install catboost scikit-learn`".format(
                ", ".join(missing)
            )
        )


def _prepare_features(df: pd.DataFrame, cat_features: Optional[List[str]] = None) -> (pd.DataFrame, pd.Series, List[str]):
    """Prepare X, y and list of categorical feature names for CatBoost.

    - Validates required columns
    - Maps target FTR -> {H,D,A}
    - Uses numeric features automatically; retains HomeTeam/AwayTeam as categorical if present in cat_features
    - Fills missing numeric values with -999 and categorical with 'UNK'
    Returns X, y, categorical_feature_names
    """
    missing = REQUIRED_INPUT_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing)}")

    data = df.copy()

    # Target
    y = data["FTR"].astype(str).copy()

    # Drop Date and raw goal columns from features by default (we keep them out; models use engineered features)
    # But if downstream wants to use them, they can be included explicitly before calling train_model
    drop_cols = ["Date", "FTR"]
    feature_df = data.drop(columns=[c for c in drop_cols if c in data.columns])

    # Identify candidate features: drop core identifiers if present (HomeTeam, AwayTeam kept optionally)
    # Keep numeric columns
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()

    # Categorical features: default to HomeTeam and AwayTeam
    if cat_features is None:
        cat_features = [c for c in ["HomeTeam", "AwayTeam"] if c in feature_df.columns]

    # Ensure categorical columns exist and are treated as strings
    for c in cat_features:
        if c in feature_df.columns:
            feature_df[c] = feature_df[c].fillna("UNK").astype(str)

    # Fill numeric NaNs
    for c in numeric_cols:
        feature_df[c] = feature_df[c].fillna(-999).astype(float)

    # Final X is feature_df with categorical columns left as object/string types
    X = feature_df

    return X, y, cat_features


def train_model(
    df: pd.DataFrame,
    *,
    model_path: str = "models/football_model.cbm",
    random_seed: int = 42,
    cat_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    cb_params: Optional[Dict] = None,
) -> Dict[str, object]:
    """
    Train a CatBoostClassifier on the provided engineered dataset.

    Steps:
    - Validate and prepare features
    - Split into train/validation (80/20)
    - Train with early stopping using validation set
    - Evaluate and print accuracy + classification report
    - Retrain on full dataset (train+val) for final model and save to `model_path`

    Returns a dict with keys: model (final CatBoostClassifier), metrics (dict), model_path
    """
    _check_deps()

    if cb_params is None:
        cb_params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",  # robust multi-class metric
            "random_seed": random_seed,
            "early_stopping_rounds": 50,
            "verbose": 100,
        }

    X, y, cat_feats = _prepare_features(df, cat_features)

    # Map labels to ensure consistent ordering
    classes = ["H", "D", "A"]
    # Ensure only expected labels are present
    unique_labels = sorted(y.unique())
    logger.info("Target labels present: %s", unique_labels)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(**cb_params)

    # Fit with eval_set as validation
    logger.info("Training CatBoost on %d samples, validating on %d samples", len(X_train), len(X_val))

    # Prepare cat_features indices for CatBoost (names or indices accepted). We'll pass names.
    cat_features_list = cat_feats

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features_list,
        use_best_model=True,
    )

    # Evaluate
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)

    acc = float(accuracy_score(y_val, y_val_pred))
    report = classification_report(y_val, y_val_pred, digits=4)

    logger.info("Validation Accuracy: %.4f", acc)
    logger.info("Classification Report:\n%s", report)

    metrics = {"validation_accuracy": acc, "classification_report": report}

    # Retrain on full data for final model
    logger.info("Retraining on full dataset (%d samples) for final model", len(X))
    final_model = CatBoostClassifier(**{k: v for k, v in cb_params.items() if k != "early_stopping_rounds"})
    final_model.fit(X, y, cat_features=cat_features_list, verbose=100)

    # Ensure model path directory exists
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model in CatBoost binary format
    final_model.save_model(str(model_path))
    logger.info("Saved final model to %s", model_path)

    return {"model": final_model, "metrics": metrics, "model_path": str(model_path)}


def load_model(path: str):
    """Load a CatBoost model from disk and return the CatBoostClassifier instance."""
    _check_deps()
    pathp = Path(path)
    if not pathp.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    m = CatBoostClassifier()
    m.load_model(str(pathp))
    logger.info("Loaded model from %s", path)
    return m


def predict_match(model, features: pd.DataFrame | Dict) -> Dict[str, float]:
    """
    Predict probabilities for a single match (or batch) using the provided model.

    Args:
        model: CatBoostClassifier (loaded or trained)
        features: either a single-row DataFrame or dict mapping feature names -> values

    Returns:
        If single row input: dict {"H": p_home_win, "D": p_draw, "A": p_away_win}
        If batch (DataFrame): returns a numpy array of shape (n_rows, 3) where columns ordered as model.classes_
    """
    _check_deps()

    if isinstance(features, dict):
        X = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        X = features.copy()
    else:
        raise ValueError("features must be a dict or pandas DataFrame")

    # Fill missing numeric values
    for c in X.select_dtypes(include=["number"]).columns:
        X[c] = X[c].fillna(-999)

    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].fillna("UNK").astype(str)

    proba = model.predict_proba(X)

    # Map classes to probabilities
    classes = list(model.classes_)
    # If single row return dict
    if len(proba.shape) == 1 or proba.shape[0] == 1:
        probs = proba.ravel() if proba.ndim > 1 else np.array([proba])
        # Ensure order H, D, A
        out = {label: 0.0 for label in ["H", "D", "A"]}
        for lbl, p in zip(classes, probs):
            out[str(lbl)] = float(p)
        return out
    return proba


if __name__ == "__main__":
    # Quick run example (requires local dataset and dependencies)
    try:
        from data_loader import load_all_data
        from src.features.feature_engineer import build_features

        raw = load_all_data()
        feats = build_features(raw)
        result = train_model(feats)
        print("Trained model saved to:", result["model_path"])
    except Exception as e:
        logger.exception("Train script failed: %s", e)
