"""
ensemble_model.py

CatBoost-only training and prediction for the football-predictor project.

Public API
----------
- train_catboost(df, *, model_path='models/football_model.cbm', ...) -> dict
- load_prediction_model(path='models/football_model.cbm') -> CatBoostClassifier
- load_artifacts(path=None) -> dict
- load_confidence_estimator(path=None) -> ConfidenceEstimator
- prepare_match_features(home_team, away_team, feature_data, model=None, artifacts=None) -> pd.DataFrame
- predict_match(model, feature_row, artifacts=None, confidence=None) -> dict
- ensure_data(merged_path, raw_dir) -> Path
- bootstrap_and_train() -> dict
"""
from __future__ import annotations

import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_deps():
    required = {"catboost": "catboost", "sklearn": "scikit-learn", "joblib": "joblib", "requests": "requests"}
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_ensure_deps()

try:
    from catboost import CatBoostClassifier, Pool
except Exception:
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

from src.features.preprocessing import build_features, FEATURE_SCHEMA
from src.models.calibration import ConfidenceEstimator

logger = logging.getLogger("football_predictor")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


DEFAULT_MODEL_PATH = "models/football_model.cbm"
DEFAULT_ARTIFACT_PATH = "models/football_model_artifacts.json"
DEFAULT_REPORT_PATH = "models/football_model_report.txt"
DEFAULT_IMPORTANCE_PATH = "models/football_model_feature_importance.csv"
DEFAULT_CONFIDENCE_PATH = "models/football_model_confidence.joblib"
MERGED_DATASET_PATH = "data/processed/merged_dataset.csv"
RAW_DIR = "data/raw"


def _require_catboost():
    if CatBoostClassifier is None:
        raise ImportError("catboost missing. Run: pip install catboost")


def _require_sklearn():
    if LabelEncoder is None:
        raise ImportError("scikit-learn missing. Run: pip install scikit-learn")


def ensure_data(merged_path: Union[str, Path] = MERGED_DATASET_PATH, raw_dir: Union[str, Path] = RAW_DIR) -> Path:
    merged_path = Path(merged_path)
    if merged_path.exists() and merged_path.stat().st_size > 0:
        logger.info("Using existing merged dataset: %s", merged_path)
        return merged_path
    logger.info("Merged dataset missing; bootstrapping from raw CSVs in %s", raw_dir)
    from src.data_pipeline.prepare_merged_dataset import build_merged_dataset
    df = build_merged_dataset(raw_dir=str(raw_dir), out_path=str(merged_path))
    if df.empty:
        raise RuntimeError("Failed to build a non-empty merged dataset.")
    return merged_path


def _time_split(df: pd.DataFrame, val_start_date: Optional[str] = None, time_gap_days: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "Date" not in df.columns:
        raise ValueError("Input must contain a 'Date' column")
    d = df.copy().sort_values("Date").reset_index(drop=True)
    if val_start_date is None:
        cutoff = d["Date"].quantile(0.80)
    else:
        cutoff = pd.to_datetime(val_start_date)
    if time_gap_days and time_gap_days > 0:
        gap_start = pd.to_datetime(cutoff) - pd.Timedelta(days=int(time_gap_days))
        train_mask = d["Date"] < gap_start
    else:
        train_mask = d["Date"] < cutoff
    val_mask = d["Date"] >= cutoff
    return d.loc[train_mask].reset_index(drop=True), d.loc[val_mask].reset_index(drop=True)


def _build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    feats = build_features(df)
    if "FTR" not in feats.columns:
        raise ValueError("Feature frame must include target column 'FTR'")
    y = feats["FTR"].astype(str)
    numeric_cols = [c for c in FEATURE_SCHEMA["numeric"] if c in feats.columns]
    cat_cols = [c for c in FEATURE_SCHEMA["categorical"] if c in feats.columns]
    feature_cols = numeric_cols + cat_cols
    X = feats[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "UNK").astype(str)
    return X, y, feature_cols, cat_cols


def train_catboost(
    df: pd.DataFrame,
    *,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    val_start_date: Optional[str] = None,
    time_gap_days: int = 0,
    random_seed: int = 42,
    cb_params: Optional[Dict] = None,
) -> Dict:
    _require_catboost()
    _require_sklearn()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    train_df, val_df = _time_split(df, val_start_date=val_start_date, time_gap_days=time_gap_days)
    logger.info("Time split: train=%d val=%d", len(train_df), len(val_df))

    X_train, y_train, feature_cols, cat_cols = _build_xy(train_df)
    X_val, y_val, _, _ = _build_xy(val_df)
    cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    logger.info("LabelEncoder classes: %s", list(le.classes_))

    params = (cb_params or {}).copy()
    defaults = {
        "iterations": 2000, "learning_rate": 0.03, "depth": 6,
        "loss_function": "MultiClass", "eval_metric": "MultiClass",
        "random_seed": random_seed, "od_type": "Iter", "od_wait": 50, "verbose": 100,
    }
    for k, v in defaults.items():
        params.setdefault(k, v)

    model = CatBoostClassifier(**params)
    logger.info("Training CatBoost on %d rows / %d features...", len(X_train), len(feature_cols))
    train_pool = Pool(X_train, y_train_enc, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, y_val_enc, cat_features=cat_feature_indices)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    metrics: Dict[str, Optional[float]] = {"accuracy": None, "log_loss": None}
    y_prob_raw = model.predict_proba(X_val)
    y_pred_enc = np.argmax(y_prob_raw, axis=1)
    if accuracy_score is not None:
        metrics["accuracy"] = float(accuracy_score(y_val_enc, y_pred_enc))
    if log_loss is not None:
        metrics["log_loss"] = float(log_loss(y_val_enc, y_prob_raw))
    logger.info("CatBoost val metrics: accuracy=%.4f log_loss=%.4f", metrics["accuracy"] or 0.0, metrics["log_loss"] or 0.0)

    # ---------------- Fit confidence calibrator on validation set ----------------
    # The calibrator maps raw probabilities to calibrated probabilities, computes
    # the reliability curve (ECE), and derives risk-band thresholds from the
    # validation distribution. All of this is data-driven and improves
    # automatically as the validation set grows.
    logger.info("Fitting confidence calibrator on %d validation predictions...", len(y_val_enc))
    confidence_est = ConfidenceEstimator.from_validation(
        raw_probs=y_prob_raw,
        y_true=y_val_enc,
        method="isotonic",
        class_names=[str(c) for c in le.classes_],
        n_bins=10,
    )
    calibrated = confidence_est.calibration.predict_proba(y_prob_raw)
    log_loss_after = float(log_loss(y_val_enc, calibrated)) if log_loss is not None else None
    logger.info(
        "Calibration fitted. ECE=%.4f, log_loss before=%.4f, after=%.4f",
        confidence_est.ece, metrics["log_loss"] or 0.0, log_loss_after or 0.0,
    )

    # ---------------- Save model + artifacts + calibrator ----------------
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info("Saved CatBoost model to %s", model_path)

    artifact_path = model_path.with_name(model_path.stem + "_artifacts.json")
    artifacts = {
        "schema_version": "v1",
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cat_feature_indices": cat_feature_indices,
        "label_classes": [str(c) for c in le.classes_],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "log_loss_raw": metrics["log_loss"],
            "log_loss_calibrated": log_loss_after,
        },
        "calibration": {
            "method": confidence_est.method,
            "ece": confidence_est.ece,
            "brier_per_class": confidence_est.brier_per_class,
            "n_validation_samples": confidence_est.n_validation_samples,
            "risk_thresholds": confidence_est.risk_thresholds,
            "reliability_curve": confidence_est.reliability_curve,
        },
        "model_path": str(model_path),
        "feature_importance": {
            name: float(imp) for name, imp in zip(feature_cols, model.get_feature_importance())
        },
    }
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(artifacts, fh, indent=2)
    logger.info("Saved artifacts to %s", artifact_path)

    # Save the calibrator next to the model
    confidence_path = model_path.with_name(model_path.stem + "_confidence.joblib")
    confidence_est.save(confidence_path)
    logger.info("Saved confidence estimator to %s", confidence_path)

    report_path = model_path.with_name(model_path.stem + "_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("Football prediction — CatBoost training report\n")
        fh.write(f"Model path: {model_path}\n")
        fh.write(f"Artifacts: {artifact_path}\n")
        fh.write(f"Confidence estimator: {confidence_path}\n")
        fh.write(f"Train rows: {len(X_train)}  Val rows: {len(X_val)}\n")
        fh.write(f"Features ({len(feature_cols)}): {feature_cols}\n")
        fh.write(f"Cat features ({len(cat_cols)}): {cat_cols}\n")
        fh.write(f"Label classes: {list(le.classes_)}\n")
        fh.write(f"Validation metrics (raw):      {metrics}\n")
        fh.write(f"Validation log_loss (calibrated): {log_loss_after}\n")
        fh.write(f"Calibration ECE: {confidence_est.ece:.4f}\n")
        fh.write(f"Risk thresholds: {confidence_est.risk_thresholds}\n")
        if classification_report is not None:
            fh.write("\nClassification report:\n")
            fh.write(classification_report(y_val_enc, y_pred_enc, target_names=list(le.classes_), zero_division=0))
        fh.write("\nTop feature importances:\n")
        sorted_imp = sorted(artifacts["feature_importance"].items(), key=lambda kv: kv[1], reverse=True)
        for name, imp in sorted_imp[:25]:
            fh.write(f"  {name:<32s} {imp:.4f}\n")
        fh.write("\nReliability curve (calibrated):\n")
        fh.write("  bin_lo  bin_hi  mean_pred  mean_actual  count  gap\n")
        for c in confidence_est.reliability_curve:
            if c["count"] > 0:
                fh.write(f"  {c['bin_lo']:.2f}    {c['bin_hi']:.2f}    "
                         f"{c['mean_predicted']:.3f}      {c['mean_actual']:.3f}        "
                         f"{c['count']:<5d} {c['gap']:+.3f}\n")

    importance_path = model_path.with_name(model_path.stem + "_feature_importance.csv")
    pd.DataFrame(
        sorted(artifacts["feature_importance"].items(), key=lambda kv: kv[1], reverse=True),
        columns=["feature", "importance"],
    ).to_csv(importance_path, index=False)

    return {
        "model_path": str(model_path),
        "artifacts_path": str(artifact_path),
        "report_path": str(report_path),
        "importance_path": str(importance_path),
        "confidence_path": str(confidence_path),
        "metrics": metrics,
        "calibration_summary": confidence_est.summary_dict(),
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "label_classes": [str(c) for c in le.classes_],
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }


def load_prediction_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> CatBoostClassifier:
    _require_catboost()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at: {p}")
    m = CatBoostClassifier()
    m.load_model(str(p))
    logger.info("Loaded CatBoost model from %s", p)
    return m


def load_artifacts(path: Optional[Union[str, Path]] = None) -> Dict:
    if path is None:
        path = Path(DEFAULT_MODEL_PATH).with_name(Path(DEFAULT_MODEL_PATH).stem + "_artifacts.json")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifacts not found at: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_confidence_estimator(path: Optional[Union[str, Path]] = None) -> ConfidenceEstimator:
    """Load the calibrated confidence estimator saved alongside the model."""
    if path is None:
        path = Path(DEFAULT_MODEL_PATH).with_name(Path(DEFAULT_MODEL_PATH).stem + "_confidence.joblib")
    return ConfidenceEstimator.load(path)


def prepare_match_features(
    home_team: str,
    away_team: str,
    feature_data: pd.DataFrame,
    model: Optional[CatBoostClassifier] = None,
    artifacts: Optional[Dict] = None,
) -> pd.DataFrame:
    if not isinstance(feature_data, pd.DataFrame):
        raise ValueError("feature_data must be a pandas DataFrame")
    if artifacts is None:
        artifacts = load_artifacts()
    feature_cols: List[str] = list(artifacts["feature_cols"])
    cat_cols: List[str] = list(artifacts["cat_cols"])

    defaults: Dict[str, object] = {}
    for c in feature_cols:
        if c in cat_cols:
            defaults[c] = "UNK"
        elif c in feature_data.columns and pd.api.types.is_numeric_dtype(feature_data[c]):
            defaults[c] = float(feature_data[c].median(skipna=True)) if feature_data[c].notna().any() else 0.0
        else:
            defaults[c] = 0.0

    home_recent = None
    away_recent = None
    if "HomeTeam" in feature_data.columns and "Date" in feature_data.columns:
        h_mask = feature_data["HomeTeam"] == home_team
        if h_mask.any():
            home_recent = feature_data.loc[h_mask].sort_values("Date").iloc[-1]
    if "AwayTeam" in feature_data.columns and "Date" in feature_data.columns:
        a_mask = feature_data["AwayTeam"] == away_team
        if a_mask.any():
            away_recent = feature_data.loc[a_mask].sort_values("Date").iloc[-1]

    row: Dict[str, object] = {}
    for c in feature_cols:
        if c in cat_cols:
            if c == "HomeTeam":
                row[c] = str(home_team)
            elif c == "AwayTeam":
                row[c] = str(away_team)
            elif c == "League" and home_recent is not None and c in home_recent.index:
                row[c] = str(home_recent[c])
            else:
                row[c] = "UNK"
        else:
            value = None
            if home_recent is not None and c in home_recent.index and pd.notna(home_recent[c]):
                value = home_recent[c]
            elif away_recent is not None and c in away_recent.index and pd.notna(away_recent[c]):
                value = away_recent[c]
            row[c] = value if value is not None else defaults[c]

    feat_row = pd.DataFrame([row], columns=feature_cols)
    for c in cat_cols:
        feat_row[c] = feat_row[c].astype(str)
    return feat_row


def predict_match(
    model: CatBoostClassifier,
    feature_row: pd.DataFrame,
    artifacts: Optional[Dict] = None,
    confidence: Optional[ConfidenceEstimator] = None,
) -> Dict:
    """Predict one match. Confidence and risk come from the data-driven calibrator.

    If `confidence` is None, the calibrator is loaded from disk automatically.
    """
    _require_catboost()
    if not isinstance(feature_row, pd.DataFrame) or feature_row.shape[0] != 1:
        raise ValueError("feature_row must be a single-row DataFrame")
    if artifacts is None:
        artifacts = load_artifacts()

    feature_cols = list(artifacts["feature_cols"])
    feature_row = feature_row[feature_cols]

    raw_probs = np.asarray(model.predict_proba(feature_row)).reshape(1, -1)[0]
    label_classes = list(artifacts["label_classes"])
    raw_mapping = {str(lbl): float(p) for lbl, p in zip(label_classes, raw_probs)}

    home_win = raw_mapping.get("H", 0.0)
    draw = raw_mapping.get("D", 0.0)
    away_win = raw_mapping.get("A", 0.0)

    # Apply calibration for confidence + risk. Calibrated probabilities replace
    # the raw mapping for the reported home/draw/away so the front-end also
    # shows calibrated numbers.
    if confidence is None:
        try:
            confidence = load_confidence_estimator()
        except FileNotFoundError:
            logger.warning("No confidence estimator found; falling back to raw probabilities.")
            confidence = None

    if confidence is not None:
        calibrated = confidence.calibration.predict_proba(raw_probs.reshape(1, -1))[0]
        cal_mapping = {str(lbl): float(p) for lbl, p in zip(label_classes, calibrated)}
        home_win = cal_mapping.get("H", home_win)
        draw = cal_mapping.get("D", draw)
        away_win = cal_mapping.get("A", away_win)

    # Renormalize defensively.
    s = home_win + draw + away_win
    if s > 0:
        home_win, draw, away_win = home_win / s, draw / s, away_win / s

    # Calibrated confidence + data-derived risk.
    if confidence is not None:
        confidence_score, risk, _ = confidence.estimate(raw_probs)
    else:
        confidence_score = float(max(home_win, draw, away_win))
        risk = "MEDIUM"

    recommended = "H" if home_win >= max(draw, away_win) else ("D" if draw >= away_win else "A")

    return {
        "match": {
            "home_team": str(feature_row["HomeTeam"].iloc[0]) if "HomeTeam" in feature_row.columns else "Unknown",
            "away_team": str(feature_row["AwayTeam"].iloc[0]) if "AwayTeam" in feature_row.columns else "Unknown",
        },
        "main_result": {
            "home_win": float(home_win),
            "draw": float(draw),
            "away_win": float(away_win),
        },
        "raw_probabilities": raw_mapping,
        "risk_level": risk,
        "confidence_score": confidence_score,
        "recommended_outcome": recommended,
    }


def bootstrap_and_train() -> Dict:
    merged_path = ensure_data(MERGED_DATASET_PATH, raw_dir=RAW_DIR)
    df = pd.read_csv(merged_path, parse_dates=["Date"])
    logger.info("Loaded merged dataset: %d rows", len(df))
    res = train_catboost(df)

    model = load_prediction_model(res["model_path"])
    artifacts = load_artifacts(Path(res["model_path"]).with_name(Path(res["model_path"]).stem + "_artifacts.json"))
    confidence = load_confidence_estimator(Path(res["model_path"]).with_name(Path(res["model_path"]).stem + "_confidence.joblib"))

    sample = df.tail(5).reset_index(drop=True)
    sample_features = build_features(sample)
    preds = []
    for _, row in sample.iterrows():
        feat_row = prepare_match_features(row["HomeTeam"], row["AwayTeam"], sample_features, model=model, artifacts=artifacts)
        preds.append(predict_match(model, feat_row, artifacts=artifacts, confidence=confidence))
    res["sample_predictions"] = preds
    logger.info("Sample predictions on last 5 rows:")
    for p in preds:
        logger.info("  %s", p)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bootstrap_and_train()
