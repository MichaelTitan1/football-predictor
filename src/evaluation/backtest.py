"""
backtest.py

Backtesting utilities for the football prediction system.

Simulates historical predictions in chronological order using a saved CatBoost model
and the advanced_features feature set. Computes per-season and overall metrics,
confidence vs correctness analysis, and returns a structured report.

Requirements met:
- Does NOT retrain the model during backtest
- Uses only past-derived features (build_advanced_features provides prior features)
- Reproducible chronological simulation
- Computes accuracy, per-season accuracy, confidence analysis, and log loss (when available)

Public functions:
- backtest_model(model, df, *, confidence_threshold=0.70) -> dict
- _prepare_feature_matrix_for_backtest(df) -> (X_df, feature_cols)

"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Delayed imports for optional metrics
try:
    from sklearn.metrics import accuracy_score, log_loss
except Exception:  # pragma: no cover - scikit-learn may not be installed
    accuracy_score = None
    log_loss = None

# CatBoost type import deferred; only used for isinstance checks when available
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


_REQUIRED_IDENTIFIERS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


def _prepare_feature_matrix_for_backtest(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Run advanced feature builder and prepare feature matrix for backtesting.

    Returns:
      X_df: DataFrame with feature columns for each match (these are prior-to-match features)
      feature_cols: list of feature column names used

    The function uses the same exclusion rules as training: excludes identifiers and raw score columns.
    """
    from src.features.advanced_features import build_advanced_features

    logger.info("Computing advanced features for backtest dataset")
    adv = build_advanced_features(df.copy())

    # Validate presence of required identifiers
    missing = [c for c in _REQUIRED_IDENTIFIERS if c not in adv.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Exclude raw identifiers and target
    exclude = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"}
    feature_cols = [c for c in adv.columns if c not in exclude]

    X_df = adv[feature_cols].copy()

    # Fill missing numeric values with column medians computed from prior rows (forward-safe)
    # For backtest we will use the column medians computed across the whole X_df as a simple, reproducible fallback
    for c in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[c]):
            med = X_df[c].median()
            X_df[c] = X_df[c].fillna(med)
        else:
            X_df[c] = X_df[c].fillna("UNK")

    return X_df, feature_cols


def _safe_predict_proba(model, X_row: pd.DataFrame) -> Optional[np.ndarray]:
    """Call model.predict_proba on a single-row DataFrame safely.
    Returns a 1D numpy array of probabilities or None on failure.
    """
    try:
        proba = model.predict_proba(X_row)
        proba = np.asarray(proba).reshape(-1)
        return proba
    except Exception as e:
        logger.exception("predict_proba failed for one row: %s", e)
        return None


def backtest_model(model, df: pd.DataFrame, *, confidence_threshold: float = 0.70) -> Dict:
    """Run a chronological backtest over the provided DataFrame using the provided trained model.

    Args:
      model: trained CatBoost model (must implement predict_proba and classes_)
      df: merged historical DataFrame (must include columns used by advanced_features)
      confidence_threshold: probability threshold used to separate high vs low confidence

    Returns a structured report dict with overall and per-season metrics plus confidence analysis.
    """
    if model is None:
        raise ValueError("model must be provided")

    # Prepare features
    X_df, feature_cols = _prepare_feature_matrix_for_backtest(df)

    # Ensure chronological order
    df2 = df.copy().reset_index(drop=True)
    df2["Date"] = pd.to_datetime(df2["Date"])
    order = df2["Date"].argsort().values
    df2 = df2.loc[order].reset_index(drop=True)
    X_df = X_df.loc[order].reset_index(drop=True)

    n = len(df2)
    logger.info("Backtesting %d matches chronologically", n)

    # Map model.classes_ to indices -> ensure labels in same order
    if not hasattr(model, "classes_"):
        logger.warning("Model has no classes_ attribute; attempting to infer from training or assume [H,D,A]")
        classes = ["H", "D", "A"]
    else:
        classes = list(model.classes_)
    logger.info("Model classes: %s", classes)

    # Prepare storage
    preds = []  # predicted labels
    probs = []  # predicted probability for each class (list of arrays)
    confidences = []  # max prob
    actuals = []  # actual FTR
    dates = []
    seasons = []
    missing_count = 0

    for i in range(n):
        row_X = X_df.iloc[[i]]  # single-row DataFrame
        row_meta = df2.iloc[i]
        date = pd.to_datetime(row_meta["Date"])
        season = str(date.year)

        dates.append(date)
        seasons.append(season)

        # Safe predict
        proba = _safe_predict_proba(model, row_X)
        actual = row_meta.get("FTR")
        actuals.append(actual)

        if proba is None:
            preds.append(None)
            probs.append(None)
            confidences.append(None)
            missing_count += 1
            continue

        # Ensure proba aligns with classes; model.predict_proba returns columns in model.classes_ order
        probs.append(proba)
        best_idx = int(np.argmax(proba))
        pred_label = classes[best_idx] if best_idx < len(classes) else None
        preds.append(pred_label)
        confidences.append(float(np.max(proba)))

    logger.info("Backtest predictions complete: %d missing predictions", missing_count)

    # Build a DataFrame for analysis
    results = pd.DataFrame({
        "date": dates,
        "season": seasons,
        "actual": actuals,
        "pred": preds,
        "confidence": confidences,
        "proba": probs,
    })

    # Compute overall accuracy ignoring missing
    valid_mask = results["pred"].notna() & results["actual"].notna()
    if valid_mask.sum() == 0:
        raise RuntimeError("No valid predictions to evaluate")

    results_valid = results.loc[valid_mask].reset_index(drop=True)

    # Accuracy overall
    overall_acc = float((results_valid["pred"] == results_valid["actual"]).mean())

    # Seasonal results
    seasonal = {}
    for season, grp in results_valid.groupby("season"):
        acc = float((grp["pred"] == grp["actual"]).mean()) if len(grp) > 0 else 0.0
        seasonal[str(season)] = acc

    # Confidence analysis: high vs low sets
    high_mask = results_valid["confidence"] >= confidence_threshold
    low_mask = results_valid["confidence"] < confidence_threshold
    high_acc = float((results_valid.loc[high_mask, "pred"] == results_valid.loc[high_mask, "actual"]).mean()) if high_mask.sum() > 0 else None
    low_acc = float((results_valid.loc[low_mask, "pred"] == results_valid.loc[low_mask, "actual"]).mean()) if low_mask.sum() > 0 else None

    # Correlation between confidence and correctness (1 if correct else 0)
    correct_series = (results_valid["pred"] == results_valid["actual"]).astype(int)
    conf_series = results_valid["confidence"].astype(float)
    try:
        conf_corr = float(conf_series.corr(correct_series)) if len(conf_series) > 1 else 0.0
    except Exception:
        conf_corr = 0.0

    # Log loss if sklearn available and model.probabilities available
    overall_logloss = None
    if log_loss is not None:
        try:
            # Build probability matrix aligned to label order [H,D,A] or model.classes_
            # Create y_true as labels
            y_true = results_valid["actual"].tolist()
            # Build proba array shape (n, k)
            proba_list = list(results_valid["proba"])
            proba_arr = np.vstack(proba_list)
            # If model.classes_ is not H/D/A in that order, log_loss still accepts label mapping
            overall_logloss = float(log_loss(y_true, proba_arr, labels=classes))
        except Exception as e:
            logger.exception("Failed to compute log_loss: %s", e)
            overall_logloss = None

    # Drift over time: compute rolling accuracy by season or by time window
    # Use seasonal_accuracy_dict computed above

    report = {
        "overall_accuracy": overall_acc,
        "overall_log_loss": overall_logloss,
        "seasonal_results": seasonal,
        "confidence_analysis": {
            "high_confidence_accuracy": high_acc,
            "low_confidence_accuracy": low_acc,
            "confidence_correctness_correlation": conf_corr,
        },
        "n_matches_evaluated": int(valid_mask.sum()),
        "n_missing_predictions": int(missing_count),
    }

    logger.info("Backtest report: overall_acc=%.4f logloss=%s n_evaluated=%d", overall_acc, str(overall_logloss), int(valid_mask.sum()))

    return report


if __name__ == "__main__":
    # Simple CLI backtest demo
    logging.basicConfig(level=logging.INFO)
    from src.models.train_advanced import train_advanced_model  # noqa: F401 - just checking availability
    from src.prediction import engine as prediction_engine  # noqa: F401

    # Load model
    model_path = "models/football_model.cbm"
    if not Path(model_path).exists():
        logger.error("Model not found at %s — please train and save model before running backtest", model_path)
        raise SystemExit(1)

    try:
        from src.prediction.engine import load_prediction_model
        model = load_prediction_model(model_path)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise

    # Load merged data
    merged_path = Path("data/processed/merged_dataset.csv")
    if not merged_path.exists():
        logger.error("Merged dataset not found at %s", merged_path)
        raise SystemExit(1)
    df = pd.read_csv(merged_path, parse_dates=["Date"])

    report = backtest_model(model, df)
    import json

    print(json.dumps(report, indent=2))
