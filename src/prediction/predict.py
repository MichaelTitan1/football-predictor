"""
predict.py

Live prediction engine for football match outcome probabilities.

This module provides a single entry point to make predictions for upcoming matches using a
trained CatBoost model and the latest engineered features produced by feature_engineer.build_features.

Functions:
- load_prediction_model(path: str) -> CatBoostClassifier
- prepare_match_features(home_team: str, away_team: str, feature_data: pd.DataFrame, model=None) -> pd.DataFrame
- predict_match_result(model, feature_row: pd.DataFrame) -> Dict[str, object]

Design goals:
- Prevent feature mismatch errors by aligning feature columns with those seen at training time
  (model.feature_names_ if available) or falling back to the feature dataset schema.
- Handle unknown teams safely by falling back to global medians / sensible defaults.
- Use the latest available per-team statistics from feature_data (most recent match rows).
- Include logging at each step.

Note: feature_data should be the dataset returned by feature_engineer.build_features(...)

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Delayed import for optional dependency
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - dependency checked at runtime
    CatBoostClassifier = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _check_catboost():
    if CatBoostClassifier is None:
        raise ImportError("catboost is required for prediction. Install with `pip install catboost`.")


def load_prediction_model(path: str):
    """Load a trained CatBoost model from disk and return it.

    Args:
        path: path to the .cbm CatBoost model file

    Returns:
        CatBoostClassifier
    """
    _check_catboost()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = CatBoostClassifier()
    model.load_model(str(p))
    logger.info("Loaded CatBoost model from %s", p)
    return model


def _get_training_feature_names(model: Optional[object], feature_data: pd.DataFrame) -> List[str]:
    """Infer the feature columns expected by the model.

    Priority:
      1. If model has attribute `feature_names_` or method `get_feature_names`, use it.
      2. Fall back to feature_data columns excluding core identifiers.

    Returns a list of feature names (strings).
    """
    # Try to get from model
    if model is not None:
        # Several CatBoost versions expose feature names differently; try common attrs/methods
        names = None
        if hasattr(model, "feature_names_"):
            names = list(getattr(model, "feature_names_"))
        elif hasattr(model, "feature_names"):
            names = list(getattr(model, "feature_names"))
        elif hasattr(model, "get_feature_names"):
            try:
                names = list(model.get_feature_names())
            except Exception:
                names = None

        if names:
            logger.debug("Using feature names from model with %d features", len(names))
            return names

    # Fallback: infer from feature_data
    # Exclude core columns that are not model features
    exclude = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    inferred = [c for c in feature_data.columns if c not in exclude]
    logger.debug("Inferred %d feature columns from feature_data", len(inferred))
    return inferred


def _latest_team_row(team: str, feature_data: pd.DataFrame, role: str) -> Optional[pd.Series]:
    """Return the most recent match row for a given team in the given role.

    role: 'home' or 'away' indicates whether to look for rows where the team was HomeTeam or AwayTeam.
    Returns a pandas Series (row) or None if not found.
    """
    if role not in {"home", "away"}:
        raise ValueError("role must be 'home' or 'away'")
    col = "HomeTeam" if role == "home" else "AwayTeam"
    mask = feature_data[col] == team
    if not mask.any():
        return None
    row = feature_data.loc[mask].sort_values("Date").iloc[-1]
    return row


def prepare_match_features(
    home_team: str,
    away_team: str,
    feature_data: pd.DataFrame,
    model: Optional[object] = None,
) -> pd.DataFrame:
    """Prepare a single-row feature DataFrame for an upcoming match.

    Strategy:
    - Determine the feature columns the model expects (see _get_training_feature_names).
    - For columns prefixed with 'home_' take the value from the most recent row where HomeTeam==home_team.
    - For columns prefixed with 'away_' take value from most recent row where AwayTeam==away_team.
    - For columns prefixed with 'h2h_' attempt to get last head-to-head row between the two teams.
    - If a source value is missing (team never seen, no h2h), fall back to the column's global median (numeric) or 'UNK' (categorical).
    - Ensure the returned DataFrame columns are ordered to match training features if model provides them.

    Args:
        home_team: name of home team (string)
        away_team: name of away team (string)
        feature_data: DataFrame returned by feature_engineer.build_features()
        model: optional trained CatBoost model (used to infer feature names)

    Returns:
        pandas.DataFrame with a single row corresponding to feature vector for the upcoming match.
    """
    # Input validation
    if not isinstance(feature_data, pd.DataFrame):
        raise ValueError("feature_data must be a pandas DataFrame")

    feature_cols = _get_training_feature_names(model, feature_data)

    # Compute global defaults (median for numeric, mode or 'UNK' for categorical)
    defaults: Dict[str, object] = {}
    for col in feature_cols:
        if col in feature_data.columns and pd.api.types.is_numeric_dtype(feature_data[col]):
            defaults[col] = float(feature_data[col].median(skipna=True)) if feature_data[col].notna().any() else 0.0
        else:
            defaults[col] = "UNK"

    # Find the most recent rows for home/away teams
    home_row = _latest_team_row(home_team, feature_data, role="home")
    away_row = _latest_team_row(away_team, feature_data, role="away")

    # For head-to-head, look for any past match between the two teams
    mask_h2h = (
        ((feature_data["HomeTeam"] == home_team) & (feature_data["AwayTeam"] == away_team))
        | ((feature_data["HomeTeam"] == away_team) & (feature_data["AwayTeam"] == home_team))
    )
    h2h_row = None
    if mask_h2h.any():
        h2h_row = feature_data.loc[mask_h2h].sort_values("Date").iloc[-1]

    # Build the feature dictionary
    feature_dict: Dict[str, object] = {}
    for col in feature_cols:
        # prefer home/away/h2h-specific sources when available
        if col.startswith("home_"):
            if home_row is not None and col in home_row.index:
                feature_dict[col] = home_row.get(col, defaults[col])
            else:
                # fallback to global median/default
                feature_dict[col] = defaults[col]
        elif col.startswith("away_"):
            if away_row is not None and col in away_row.index:
                feature_dict[col] = away_row.get(col, defaults[col])
            else:
                feature_dict[col] = defaults[col]
        elif col.startswith("h2h_"):
            if h2h_row is not None and col in h2h_row.index:
                feature_dict[col] = h2h_row.get(col, defaults[col])
            else:
                feature_dict[col] = defaults[col]
        elif col in {"HomeTeam", "AwayTeam"}:
            # keep the team names as-is
            feature_dict[col] = home_team if col == "HomeTeam" else away_team
        else:
            # Generic: try to use last overall metric for the home team first, then away, else default
            # Some features may be unprefixed per-match aggregates; try to find them in home_row then away_row
            val = None
            if home_row is not None and col in home_row.index:
                val = home_row.get(col)
            elif away_row is not None and col in away_row.index:
                val = away_row.get(col)
            feature_dict[col] = val if val is not None else defaults.get(col, "UNK")

    # Convert dict to single-row DataFrame and ensure column order matches feature_cols
    feature_row = pd.DataFrame([feature_dict], columns=feature_cols)

    # Coerce numeric columns to numeric dtypes
    for c in feature_row.columns:
        if c in feature_data.columns and pd.api.types.is_numeric_dtype(feature_data[c]):
            feature_row[c] = pd.to_numeric(feature_row[c], errors="coerce").fillna(defaults.get(c, 0.0)).astype(float)

    logger.info("Prepared feature row for %s vs %s", home_team, away_team)
    return feature_row


def predict_match_result(model, feature_row: pd.DataFrame) -> Dict[str, object]:
    """Predict probabilities for a single upcoming match.

    Args:
        model: trained CatBoostClassifier (or compatible object with predict_proba and classes_)
        feature_row: single-row DataFrame of features (as returned by prepare_match_features)

    Returns:
        Dict with keys: home_team, away_team, probabilities (mapping HomeWin/Draw/AwayWin), recommended_outcome (H/D/A)
    """
    _check_catboost()

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not have predict_proba method")

    if not isinstance(feature_row, pd.DataFrame):
        raise ValueError("feature_row must be a pandas DataFrame")

    if feature_row.shape[0] != 1:
        raise ValueError("feature_row must contain exactly one row")

    # Ensure features are in the same column order expected by the model if possible
    model_feature_names = None
    if hasattr(model, "feature_names_"):
        model_feature_names = list(getattr(model, "feature_names_"))
    elif hasattr(model, "get_feature_names"):
        try:
            model_feature_names = list(model.get_feature_names())
        except Exception:
            model_feature_names = None

    if model_feature_names is not None:
        # align and add missing cols with zeros/defaults
        missing = [c for c in model_feature_names if c not in feature_row.columns]
        for c in missing:
            logger.debug("Adding missing model feature %s with default 0.0", c)
            feature_row[c] = 0.0
        feature_row = feature_row[model_feature_names]

    # Predict probabilities
    proba = model.predict_proba(feature_row)
    # proba shape: (1, n_classes)
    proba = np.asarray(proba).reshape(-1)

    # Map classes to probabilities
    classes = list(model.classes_)
    # Build mapping with consistent keys
    mapping = {str(lbl): float(0.0) for lbl in classes}
    for lbl, p in zip(classes, proba):
        mapping[str(lbl)] = float(p)

    # Standardize to requested output keys
    probs_out = {
        "HomeWin": mapping.get("H", 0.0),
        "Draw": mapping.get("D", 0.0),
        "AwayWin": mapping.get("A", 0.0),
    }

    # Recommended outcome
    # pick argmax among H,D,A (if some absent, highest among available)
    best = max(probs_out.items(), key=lambda kv: kv[1])[0]
    # convert best back to short label
    rec = {"HomeWin": "H", "Draw": "D", "AwayWin": "A"}[best]

    # Extract team names if present as columns
    home_team = feature_row["HomeTeam"].iloc[0] if "HomeTeam" in feature_row.columns else "Unknown"
    away_team = feature_row["AwayTeam"].iloc[0] if "AwayTeam" in feature_row.columns else "Unknown"

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "probabilities": probs_out,
        "recommended_outcome": rec,
    }

    logger.info(
        "Predicted %s vs %s -> Home: %.3f Draw: %.3f Away: %.3f; recommend %s",
        home_team,
        away_team,
        probs_out["HomeWin"],
        probs_out["Draw"],
        probs_out["AwayWin"],
        rec,
    )

    return result


# Example CLI-style quick test (not executed on import)
if __name__ == "__main__":
    try:
        # Quick manual test if model and feature_data exist locally
        model = load_prediction_model("models/football_model.cbm")
        # feature_data should be a DataFrame from feature_engineer.build_features
        from data_loader import load_all_data
        from src.features.feature_engineer import build_features

        raw = load_all_data()
        features = build_features(raw)
        home = features["HomeTeam"].iloc[-1]
        away = features["AwayTeam"].iloc[-1]
        feat_row = prepare_match_features(home, away, features, model=model)
        print(predict_match_result(model, feat_row))
    except Exception as e:
        logger.exception("Live predict test failed: %s", e)
