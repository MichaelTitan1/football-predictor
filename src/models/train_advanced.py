"""
train_advanced.py

Professional training pipeline for the football prediction system using advanced engineered features only.

Public functions:
- train_advanced_model(df: pd.DataFrame, *, model_path: str = "models/football_model.cbm", val_start_date: Optional[pd.Timestamp] = None, time_gap_days: int = 0, random_seed: int = 42, cb_params: Optional[Dict] = None) -> Dict
- evaluate_model(model, X_val, y_val) -> Dict

Key behaviors:
- Uses only build_advanced_features(df) as input features (no basic features allowed)
- Splits by time (date-based split) to prevent leakage
- Trains CatBoostClassifier with early stopping and overfitting controls
- Produces model file, feature importance CSV, and a training report text file

Engineering:
- Reproducible (random_seed)
- Safe handling of missing values
- Logs all major steps
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# delayed imports
try:
    from catboost import CatBoostClassifier, Pool
except Exception:  # pragma: no cover - dependency handled at runtime
    CatBoostClassifier = None
    Pool = None

try:
    from sklearn.metrics import accuracy_score, log_loss, classification_report
    from sklearn.preprocessing import LabelEncoder
except Exception:
    accuracy_score = None
    log_loss = None
    classification_report = None
    LabelEncoder = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _check_deps():
    missing = []
    if CatBoostClassifier is None:
        missing.append("catboost")
    if accuracy_score is None:
        missing.append("scikit-learn")
    if missing:
        raise ImportError(f"Missing dependencies: {', '.join(missing)}. Install via pip.")


def _prepare_data_advanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Run build_advanced_features and prepare X, y. Returns X, y, feature_names list.

    Only uses advanced features produced by build_advanced_features.
    """
    from src.features.advanced_features import build_advanced_features

    adv = build_advanced_features(df)

    # Target
    if "FTR" not in adv.columns:
        raise ValueError("Advanced features frame must contain target column 'FTR'")

    y = adv["FTR"].astype(str).reset_index(drop=True)

    # Exclude original identifiers and raw goals from features
    exclude = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"}
    feature_cols = [c for c in adv.columns if c not in exclude]

    X = adv[feature_cols].reset_index(drop=True)

    # Handle missing values: numeric -> median, object -> 'UNK'
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            med = X[c].median()
            X[c] = X[c].fillna(med)
        else:
            X[c] = X[c].fillna("UNK").astype(str)

    return X, y, feature_cols


def _time_train_val_split(df: pd.DataFrame, val_start_date: Optional[pd.Timestamp] = None, time_gap_days: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by date into train (older) and val (newer).

    If val_start_date provided (pd.Timestamp or parseable string) use it. Otherwise use 80/20 quantile cutoff.
    time_gap_days: if >0, exclude matches within this many days before validation start from training to create a gap.
    Returns train_df, val_df (both are slices of original df)
    """
    d = df.copy().sort_values("Date").reset_index(drop=True)
    if val_start_date is None:
        # pick 80th percentile date
        cutoff = d["Date"].quantile(0.80)
    else:
        cutoff = pd.to_datetime(val_start_date)

    # Optionally create a time gap
    if time_gap_days and time_gap_days > 0:
        gap_start = pd.to_datetime(cutoff) - pd.Timedelta(days=int(time_gap_days))
        train_mask = d["Date"] < gap_start
    else:
        train_mask = d["Date"] < cutoff

    val_mask = d["Date"] >= cutoff

    train_df = d.loc[train_mask].reset_index(drop=True)
    val_df = d.loc[val_mask].reset_index(drop=True)

    logger.info("Time split: train rows=%d val rows=%d cutoff=%s", len(train_df), len(val_df), str(cutoff))
    return train_df, val_df


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
    """Evaluate model and return metrics dict. Prints classification report.

    Metrics: accuracy, log_loss, classification_report
    """
    _check_deps()
    if hasattr(model, "predict"):
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
    else:
        raise ValueError("Model does not have predict/predict_proba methods")

    # ensure labels aligned
    acc = float(accuracy_score(y_val, y_pred))
    ll = float(log_loss(y_val, y_proba, labels=model.classes_))
    report = classification_report(y_val, y_pred, digits=4)

    logger.info("Validation Accuracy: %.4f", acc)
    logger.info("Validation LogLoss: %.4f", ll)
    logger.info("Classification Report:\n%s", report)

    return {"accuracy": acc, "log_loss": ll, "classification_report": report}


def train_advanced_model(
    df: pd.DataFrame,
    *,
    model_path: str = "models/football_model.cbm",
    feature_importance_path: str = "models/feature_importance.csv",
    training_report_path: str = "models/training_report.txt",
    val_start_date: Optional[str] = None,
    time_gap_days: int = 0,
    random_seed: int = 42,
    cb_params: Optional[Dict] = None,
) -> Dict:
    """Train a CatBoost model using advanced features only and time-based validation.

    Args:
      df: raw merged dataframe with Date + match columns
      model_path: path to save final model
      val_start_date: optional date string to start validation period; if None use 80th percentile
      time_gap_days: number of days to exclude before validation start from training (temporal gap)
      random_seed: RNG seed
      cb_params: optional CatBoost params override

    Returns:
      dict with keys: model, metrics, model_path, feature_importance_path, training_report_path
    """
    _check_deps()

    np.random.seed(random_seed)

    # Build advanced features and obtain X/y
    logger.info("Building advanced features and preparing data")
    from src.features.advanced_features import build_advanced_features

    adv = build_advanced_features(df)

    # Ensure Date column is datetime
    adv["Date"] = pd.to_datetime(adv["Date"])

    # Time split
    train_df, val_df = _time_train_val_split(adv, val_start_date, time_gap_days)

    if train_df.empty or val_df.empty:
        raise ValueError("Train or validation split is empty; adjust val_start_date or check input data span")

    # Prepare X, y for train and val
    X_all_cols, y_all, feature_cols = _prepare_data_advanced(adv)

    # Now map rows for train_df and val_df to indices in adv to split consistently
    # We'll use Date + HomeTeam + AwayTeam as keys
    adv_keys = adv[ ["Date", "HomeTeam", "AwayTeam"] ].astype(str).agg("__".join, axis=1)
    train_keys = train_df[ ["Date", "HomeTeam", "AwayTeam"] ].astype(str).agg("__".join, axis=1)
    val_keys = val_df[ ["Date", "HomeTeam", "AwayTeam"] ].astype(str).agg("__".join, axis=1)

    key_to_index = {k: i for i, k in enumerate(adv_keys)}
    train_idx = [ key_to_index[k] for k in train_keys if k in key_to_index ]
    val_idx = [ key_to_index[k] for k in val_keys if k in key_to_index ]

    X = X_all_cols
    y = y_all

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    logger.info("Prepared training set: %d rows, validation set: %d rows", len(X_train), len(X_val))

    # Encode labels to ensure consistent label order
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    classes = list(le.classes_)
    logger.info("Classes: %s", classes)

    # CatBoost params defaults
    if cb_params is None:
        cb_params = {
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "random_seed": random_seed,
            "od_type": "Iter",
            "od_wait": 50,
            "verbose": 100,
        }
    else:
        cb_params = dict(cb_params)
        cb_params.setdefault("random_seed", random_seed)
        cb_params.setdefault("verbose", 100)

    # Prepare CatBoost Pool objects (if Pool available)
    cat_features = []  # advanced features are numeric mostly; leave empty for CatBoost to auto-detect

    if Pool is not None:
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
    else:
        train_pool = (X_train, y_train)
        val_pool = (X_val, y_val)

    # Initialize and train
    model = CatBoostClassifier(**cb_params)
    logger.info("Starting CatBoost training with params: %s", {k: cb_params.get(k) for k in ["iterations","learning_rate","depth"]})

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, cat_features=cat_features)

    # Evaluate on validation set
    logger.info("Evaluating model on validation set")
    metrics = evaluate_model(model, X_val, y_val)

    # Save model
    model_p = Path(model_path)
    model_p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_p))
    logger.info("Saved trained model to %s", model_p)

    # Feature importance
    try:
        fi = model.get_feature_importance(train_pool if Pool is not None else X_train)
        fi_df = pd.DataFrame({"feature": feature_cols, "importance": fi})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_path = Path(feature_importance_path)
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(fi_path, index=False)
        logger.info("Wrote feature importance to %s", fi_path)
    except Exception as e:
        logger.exception("Failed to compute or save feature importance: %s", e)
        fi_path = None

    # Training summary
    report_lines = []
    report_lines.append(f"Training summary\n")
    report_lines.append(f"Model: CatBoostClassifier\n")
    report_lines.append(f"Training rows: {len(X_train)}\n")
    report_lines.append(f"Validation rows: {len(X_val)}\n")
    report_lines.append(f"Classes: {classes}\n")
    report_lines.append(f"Metrics: accuracy={metrics.get('accuracy'):.4f}, log_loss={metrics.get('log_loss'):.4f}\n")
    report_lines.append("\nClassification report:\n")
    report_lines.append(metrics.get("classification_report", ""))

    report_path = Path(training_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.writelines(report_lines)

    logger.info("Wrote training report to %s", report_path)

    return {
        "model": model,
        "model_path": str(model_p),
        "feature_importance_path": str(fi_path) if fi_path is not None else None,
        "training_report_path": str(report_path),
        "metrics": metrics,
    }


if __name__ == "__main__":
    # Quick CLI demo: requires a merged dataset at data/processed/merged_dataset.csv
    logging.basicConfig(level=logging.INFO)
    data_path = os.environ.get("MERGED_DATA_PATH", "data/processed/merged_dataset.csv")
    if not Path(data_path).exists():
        logger.error("Merged dataset not found at %s — aborting demo", data_path)
        raise SystemExit(1)
    df = pd.read_csv(data_path, parse_dates=["Date"]) if Path(data_path).exists() else None
    res = train_advanced_model(df)
    logger.info("Training complete. Model saved to %s", res["model_path"]) 
