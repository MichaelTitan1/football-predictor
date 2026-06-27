"""
calibration.py

Calibration utilities for football prediction probabilities.

Implements:
- Platt scaling (per-class logistic regression / sigmoid) for multiclass probabilities
- Isotonic regression (per-class) as a non-parametric alternative
- fit_calibration(model_outputs, true_labels, method='platt'|'isotonic') -> CalibrationModel
- calibrated_predict(cal_model, raw_probs) -> calibrated_probs
- evaluation helpers: brier_score, compare_log_loss
- ConfidenceEstimator: bundles calibration + reliability + data-derived risk thresholds
- compute_ece / compute_reliability_curve / derive_risk_thresholds

Design notes:
- Works with ensemble outputs (N x K probability arrays) or single-model outputs
- Deterministic and reproducible given same inputs
- Does NOT retrain the main ML model; learns only a mapping from raw probs to calibrated probs
- Uses sklearn for regressors and metrics; falls back with clear errors if deps missing
- Safe handling of edge cases (zero-sum rows, missing values)

API examples:
    cal = fit_calibration(raw_probs, y_true, method='platt')
    calibrated = calibrated_predict(cal, raw_probs)

    est = ConfidenceEstimator.from_validation(raw_probs, y_true, method='isotonic')
    est.save('models/football_model_confidence.joblib')

    est = ConfidenceEstimator.load('models/football_model_confidence.joblib')
    confidence, risk, top_idx = est.estimate(raw_probs_row)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
    brier_score_loss = None
    log_loss = None


# =========================================================================== #
# Existing CalibrationModel + helpers (kept verbatim)                         #
# =========================================================================== #

class CalibrationModel:
    """Container for per-class calibrators and utility methods."""

    def __init__(self, method: str, classes_: list, calibrators: Dict[Any, Any]):
        self.method = method
        self.classes_ = list(classes_)
        self.calibrators = calibrators

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_probs, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        n, k = raw.shape
        if k != len(self.classes_):
            logger.warning("raw_probs has %d classes but calibration model expects %d", k, len(self.classes_))
            if k < len(self.classes_):
                padded = np.zeros((n, len(self.classes_)), dtype=float)
                padded[:, :k] = raw
                raw = padded
                k = raw.shape[1]
            else:
                raw = raw[:, : len(self.classes_)]
                k = raw.shape[1]

        calibrated = np.zeros_like(raw)
        for j, cls in enumerate(self.classes_):
            col = raw[:, j]
            col = np.clip(col, 1e-6, 1 - 1e-6)
            calibrator = self.calibrators.get(cls)
            if calibrator is None:
                calibrated[:, j] = col
                continue
            try:
                if self.method == "platt":
                    pred = calibrator.predict_proba(col.reshape(-1, 1))[:, 1]
                else:
                    pred = calibrator.predict(col)
                pred = np.clip(pred, 1e-6, 1 - 1e-6)
                calibrated[:, j] = pred
            except Exception as e:
                logger.exception("Calibration predict failed for class %s: %s", cls, e)
                calibrated[:, j] = col

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        calibrated = calibrated / row_sums
        return calibrated


def _ensure_sklearn_available():
    if IsotonicRegression is None or LogisticRegression is None:
        raise ImportError("scikit-learn is required for calibration.")


def fit_calibration(model_outputs: np.ndarray, true_labels: np.ndarray, *, method: str = "platt", max_samples: Optional[int] = None, random_state: int = 42) -> CalibrationModel:
    _ensure_sklearn_available()
    probs = np.asarray(model_outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    n, k = probs.shape
    true = np.asarray(true_labels)
    classes_ = [str(i) for i in range(k)]
    calibrators = {}
    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    if (max_samples is not None) and (n > max_samples):
        indices = rng.choice(n, size=int(max_samples), replace=False)
    for j in range(k):
        cls = classes_[j]
        bin_y = (pd.Series(true).astype(str) == cls).astype(int).values
        X_col = probs[:, j]
        X_col_sub = X_col[indices]
        bin_y_sub = bin_y[indices]
        try:
            if method == "platt":
                lr = LogisticRegression(max_iter=1000, solver="lbfgs")
                lr.fit(X_col_sub.reshape(-1, 1), bin_y_sub)
                calibrators[cls] = lr
            else:
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                ir.fit(X_col_sub, bin_y_sub)
                calibrators[cls] = ir
        except Exception as e:
            logger.warning("Calibration failed for class %s: %s; falling back to identity", cls, e)
            calibrators[cls] = None
    return CalibrationModel(method=method, classes_=classes_, calibrators=calibrators)


def calibrated_predict(cal_model: CalibrationModel, raw_probs: np.ndarray) -> np.ndarray:
    return cal_model.predict_proba(raw_probs)


def brier_score(true_labels: np.ndarray, probs: np.ndarray, *, pos_label: Optional[str] = None) -> float:
    if brier_score_loss is None:
        raise ImportError("scikit-learn is required for brier_score.")
    true = np.asarray(true_labels)
    p = np.asarray(probs, dtype=float)
    if pos_label is not None:
        bin_y = (pd.Series(true).astype(str) == str(pos_label)).astype(int).values
        if p.shape[1] == 1:
            return float(brier_score_loss(bin_y, p[:, 0]))
        try:
            i = int(pos_label)
            return float(brier_score_loss(bin_y, p[:, i]))
        except Exception:
            return float(np.mean((p - bin_y.reshape(-1, 1)) ** 2))
    return float(np.mean((p - np.eye(p.shape[1])[true].reshape(-1, p.shape[1])) ** 2))


def compare_log_loss(true_labels: np.ndarray, probs_before: np.ndarray, probs_after: np.ndarray) -> Dict[str, Optional[float]]:
    if log_loss is None:
        raise ImportError("scikit-learn is required for compare_log_loss.")
    true = np.asarray(true_labels)
    pb = np.clip(np.asarray(probs_before, dtype=float), 1e-15, 1 - 1e-15)
    pa = np.clip(np.asarray(probs_after, dtype=float), 1e-15, 1 - 1e-15)
    n_classes = pb.shape[1]
    one_hot = np.eye(n_classes)[true.astype(int)]
    ll_b = float(-np.mean(np.sum(one_hot * np.log(pb), axis=1)))
    ll_a = float(-np.mean(np.sum(one_hot * np.log(pa), axis=1)))
    return {"log_loss_before": ll_b, "log_loss_after": ll_a}


# =========================================================================== #
# NEW: Reliability, ECE, data-derived risk thresholds, ConfidenceEstimator   #
# =========================================================================== #

def compute_reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, Any]]:
    """For each confidence bin, return mean predicted probability vs actual accuracy."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(1, -1)
    predicted = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correct = (predicted == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    out: List[Dict[str, Any]] = []
    for b in range(n_bins):
        mask = bin_indices == b
        n = int(mask.sum())
        if n == 0:
            out.append({
                "bin_index": b, "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
                "mean_predicted": None, "mean_actual": None, "count": 0, "gap": None,
            })
            continue
        mean_pred = float(confidences[mask].mean())
        mean_actual = float(correct[mask].mean())
        out.append({
            "bin_index": b, "bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]),
            "mean_predicted": mean_pred, "mean_actual": mean_actual,
            "count": n, "gap": mean_pred - mean_actual,
        })
    return out


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean |mean_predicted - mean_actual|."""
    curve = compute_reliability_curve(y_true, y_prob, n_bins=n_bins)
    total = sum(c["count"] for c in curve)
    if total == 0:
        return 0.0
    ece = 0.0
    for c in curve:
        if c["count"] > 0 and c["mean_predicted"] is not None:
            ece += (c["count"] / total) * abs(c["mean_predicted"] - c["mean_actual"])
    return float(ece)


def derive_risk_thresholds(calibrated_probs: np.ndarray, *, low_quantile: float = 0.67, medium_quantile: float = 0.33) -> Dict[str, Any]:
    """Derive risk-band thresholds from the validation distribution of calibrated confidences.

    The thresholds are quantile-based on the validation-set output. They update
    automatically with each retraining. No magic numbers.
    """
    calibrated = np.asarray(calibrated_probs, dtype=float)
    max_cal = np.max(calibrated, axis=1)
    return {
        "low_min_confidence": float(np.quantile(max_cal, low_quantile)),
        "medium_min_confidence": float(np.quantile(max_cal, medium_quantile)),
        "low_quantile": low_quantile,
        "medium_quantile": medium_quantile,
        "validation_mean_confidence": float(max_cal.mean()),
        "validation_std_confidence": float(max_cal.std(ddof=0)),
        "n_samples": int(len(max_cal)),
    }


@dataclass
class ConfidenceEstimator:
    """Bundles calibration + reliability stats + data-derived risk thresholds.

    Built from validation predictions; persisted alongside the model. The
    predict path loads it and applies it without retraining.
    """

    calibration: Any  # CalibrationModel
    ece: float
    brier_per_class: Dict[str, float]
    reliability_curve: List[Dict[str, Any]]
    risk_thresholds: Dict[str, Any]
    n_validation_samples: int
    classes: List[str]
    version: str = "v1"
    method: str = "isotonic"

    @classmethod
    def from_validation(cls, raw_probs: np.ndarray, y_true: np.ndarray, *, method: str = "isotonic", class_names: Optional[List[str]] = None, n_bins: int = 10) -> "ConfidenceEstimator":
        raw = np.asarray(raw_probs, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        y_true = np.asarray(y_true)
        cal = fit_calibration(raw, y_true, method=method)
        calibrated = cal.predict_proba(raw)
        true_str = pd.Series(y_true).astype(str).values
        classes = list(class_names) if class_names is not None else list(cal.classes_)
        brier: Dict[str, float] = {}
        for j, cname in enumerate(classes):
            bin_y = (true_str == str(cname)).astype(float)
            col = calibrated[:, j] if j < calibrated.shape[1] else np.zeros(len(true_str))
            brier[str(cname)] = float(np.mean((col - bin_y) ** 2))
        curve = compute_reliability_curve(y_true, calibrated, n_bins=n_bins)
        ece = compute_ece(y_true, calibrated, n_bins=n_bins)
        thresholds = derive_risk_thresholds(calibrated)
        return cls(
            calibration=cal, ece=ece, brier_per_class=brier,
            reliability_curve=curve, risk_thresholds=thresholds,
            n_validation_samples=int(len(y_true)), classes=classes, method=method,
        )

    def estimate(self, raw_probs_row: np.ndarray) -> Tuple[float, str, int]:
        """For a single row of raw probabilities, return (confidence, risk_level, top_class_idx)."""
        row = np.asarray(raw_probs_row, dtype=float).reshape(1, -1)
        calibrated = self.calibration.predict_proba(row)
        top_idx = int(np.argmax(calibrated[0]))
        confidence = float(calibrated[0, top_idx])
        risk = self._risk_from_confidence(confidence)
        return confidence, risk, top_idx

    def _risk_from_confidence(self, confidence: float) -> str:
        low_min = float(self.risk_thresholds.get("low_min_confidence", 1.1))
        med_min = float(self.risk_thresholds.get("medium_min_confidence", 1.1))
        if confidence >= low_min:
            return "LOW"
        if confidence >= med_min:
            return "MEDIUM"
        return "HIGH"

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "method": self.method,
            "classes": list(self.classes),
            "n_validation_samples": self.n_validation_samples,
            "ece": float(self.ece),
            "brier_per_class": dict(self.brier_per_class),
            "reliability_curve": self.reliability_curve,
            "risk_thresholds": dict(self.risk_thresholds),
        }

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        summary_path = path.with_suffix(".json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(self.summary_dict(), fh, indent=2)
        logger.info("Saved ConfidenceEstimator to %s (+ %s)", path, summary_path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConfidenceEstimator":
        path = Path(path)
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Object at {path} is not a ConfidenceEstimator")
        logger.info("Loaded ConfidenceEstimator from %s", path)
        return obj
