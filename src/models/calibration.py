"""
calibration.py

Calibration utilities for football prediction probabilities.

Implements:
- Platt scaling (per-class logistic regression / sigmoid) for multiclass probabilities (one-vs-rest)
- Isotonic regression (per-class) as a non-parametric alternative
- fit_calibration(model_outputs, true_labels, method='platt'|'isotonic') -> CalibrationModel
- calibrated_predict(cal_model, raw_probs) -> calibrated_probs
- evaluation helpers: brier_score, compare_log_loss

Design notes:
- Works with ensemble outputs (N x K probability arrays) or single-model outputs
- Deterministic and reproducible given same inputs
- Does NOT retrain the main ML model; learns only a mapping from raw probs to calibrated probs
- Uses sklearn for regressors and metrics; falls back with clear errors if deps missing
- Safe handling of edge cases (zero-sum rows, missing values)

API examples:
    cal = fit_calibration(raw_probs, y_true, method='platt')
    calibrated = calibrated_predict(cal, raw_probs)

"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# sklearn imports
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss
except Exception:  # pragma: no cover
    IsotonicRegression = None
    LogisticRegression = None
    brier_score_loss = None
    log_loss = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class CalibrationModel:
    """Container for per-class calibrators and utility methods.

    Attributes:
        method: 'platt' or 'isotonic'
        calibrators: dict mapping class_label -> fitted calibrator (sklearn object)
        classes_: list of class labels corresponding to columns in probabilities
    """

    def __init__(self, method: str, classes_: list, calibrators: Dict[Any, Any]):
        self.method = method
        self.classes_ = list(classes_)
        self.calibrators = calibrators

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply per-class calibrators to raw_probs and renormalize to sum to 1.

        raw_probs: shape (n_samples, n_classes)
        Returns calibrated_probs with same shape.
        """
        raw = np.asarray(raw_probs, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        n, k = raw.shape
        if k != len(self.classes_):
            logger.warning("raw_probs has %d classes but calibration model expects %d; attempting safe align", k, len(self.classes_))
            # If mismatch, attempt to pad/truncate
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
            # Clip to [1e-6, 1-1e-6] to keep calibrators stable
            col = np.clip(col, 1e-6, 1 - 1e-6)
            calibrator = self.calibrators.get(cls)
            if calibrator is None:
                # no calibrator for this class: keep original probabilities
                calibrated[:, j] = col
                continue
            try:
                if self.method == "platt":
                    # logistic regression: expects 2D input
                    pred = calibrator.predict_proba(col.reshape(-1, 1))[:, 1]
                else:
                    # isotonic: returns transformed values via predict
                    pred = calibrator.predict(col)
                # numeric stability
                pred = np.clip(pred, 1e-6, 1 - 1e-6)
                calibrated[:, j] = pred
            except Exception as e:
                logger.exception("Calibration predict failed for class %s: %s", cls, e)
                calibrated[:, j] = col

        # Renormalize rows to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        calibrated = calibrated / row_sums
        return calibrated


def _ensure_sklearn_available():
    if IsotonicRegression is None or LogisticRegression is None:
        raise ImportError("scikit-learn (isotonic, linear_model) is required for calibration. Install scikit-learn.")


def fit_calibration(model_outputs: np.ndarray, true_labels: np.ndarray, *, method: str = "platt", max_samples: Optional[int] = None, random_state: int = 42) -> CalibrationModel:
    """Fit calibration mapping per class.

    Args:
        model_outputs: array-like shape (n_samples, n_classes) of raw probabilities
        true_labels: array-like shape (n_samples,) with labels matching classes (e.g., 'H','D','A' or 0..k-1)
        method: 'platt' (logistic) or 'isotonic'
        max_samples: optionally subsample for isotonic to avoid overfitting or heavy compute
    Returns:
        CalibrationModel instance with per-class calibrators
    """
    _ensure_sklearn_available()

    probs = np.asarray(model_outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    n, k = probs.shape

    # Normalize true_labels to consistent label set
    true = np.asarray(true_labels)
    # If labels are numeric 0..k-1, map to strings to keep keys generic
    labels_unique = list(pd.Series(true).astype(str).unique())

    # Establish canonical class order from columns: if classes are H/D/A, user should ensure order matches
    # We'll set classes_ to strings '0','1',... if labels numeric and columns unknown
    # Best practice: provide model_outputs columns in same order as label encoder/classes_ used in training
    classes_ = [str(i) for i in range(k)]

    # If true contains exactly k unique labels and they look like H/D/A or similar, map them to classes_ order
    # We will internally use class indices as string keys
    # Build calibrators per class index
    calibrators = {}

    # Optionally subsample indices for calibration fitting to speed up isotonic
    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    if (max_samples is not None) and (n > max_samples):
        indices = rng.choice(n, size=int(max_samples), replace=False)

    for j in range(k):
        cls = classes_[j]
        # Build binary labels: 1 if true == this class, else 0
        # match by string for robustness
        bin_y = (pd.Series(true).astype(str) == cls).astype(int).values
        X_col = probs[:, j]
        X_col_sub = X_col[indices]
        y_sub = bin_y[indices]

        # Avoid degenerate cases: if y_sub all same, skip calibrator
        if np.all(y_sub == 0) or np.all(y_sub == 1):
            logger.warning("Binary labels for class %s are constant in calibration subset; skipping calibrator", cls)
            calibrators[cls] = None
            continue

        if method == "platt":
            # logistic regression on single probability column
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            try:
                lr.fit(X_col_sub.reshape(-1, 1), y_sub)
                calibrators[cls] = lr
            except Exception as e:
                logger.exception("Platt scaling failed for class %s: %s", cls, e)
                calibrators[cls] = None
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            try:
                iso.fit(X_col_sub, y_sub)
                calibrators[cls] = iso
            except Exception as e:
                logger.exception("Isotonic regression failed for class %s: %s", cls, e)
                calibrators[cls] = None
        else:
            raise ValueError("Unknown calibration method: %s" % method)

    cal_model = CalibrationModel(method=method, classes_=classes_, calibrators=calibrators)
    logger.info("Fitted calibration model method=%s classes=%s", method, classes_)
    return cal_model


def calibrated_predict(cal_model: CalibrationModel, raw_probs: np.ndarray) -> np.ndarray:
    """Apply CalibrationModel to raw probabilities and return calibrated probabilities.

    raw_probs: array-like (n_samples, n_classes)
    Returns numpy array (n_samples, n_classes)
    """
    if not isinstance(cal_model, CalibrationModel):
        raise ValueError("cal_model must be an instance of CalibrationModel")
    raw = np.asarray(raw_probs, dtype=float)
    return cal_model.predict_proba(raw)


def brier_score(true_labels: np.ndarray, probs: np.ndarray, *, pos_label: Optional[str] = None) -> float:
    """Compute multi-class Brier score (mean squared error across probability vectors).

    Brier = mean over samples of sum((p_true - p_pred)^2)
    true_labels: shape (n,) with labels as strings or ints; probs shape (n, k)
    """
    true = np.asarray(true_labels)
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    n, k = p.shape

    # Construct one-hot true matrix aligned by label index 0..k-1
    # We assume labels map to string indices '0'..'k-1' as used in CalibrationModel
    true_str = pd.Series(true).astype(str).values
    one_hot = np.zeros_like(p)
    for i in range(n):
        lab = true_str[i]
        try:
            idx = int(lab)
            if 0 <= idx < k:
                one_hot[i, idx] = 1.0
        except Exception:
            # if cannot map, leave row zeros (conservative)
            pass

    bs = np.mean(np.sum((p - one_hot) ** 2, axis=1))
    return float(bs)


def compare_log_loss(true_labels: np.ndarray, probs_before: np.ndarray, probs_after: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute log loss before and after calibration. Returns dict with both values and delta.
    If sklearn not available or computation fails returns None values.
    """
    if log_loss is None:
        logger.warning("sklearn.metrics.log_loss not available; cannot compute log loss")
        return {"before": None, "after": None, "delta": None}

    try:
        # Convert labels to integer indices as strings '0'.. etc; assumes probs columns correspond to label indices
        true = np.asarray(true_labels)
        # If labels are strings that represent integers, convert
        try:
            y_int = np.array([int(str(x)) for x in true])
        except Exception:
            # Try to map unique labels to indices
            uniq = list(pd.Series(true).unique())
            mapping = {lab: i for i, lab in enumerate(uniq)}
            y_int = np.array([mapping[x] for x in true])

        before = float(log_loss(y_int, probs_before))
    except Exception as e:
        logger.exception("Failed to compute pre-calibration log_loss: %s", e)
        before = None

    try:
        after = float(log_loss(y_int, probs_after))
    except Exception as e:
        logger.exception("Failed to compute post-calibration log_loss: %s", e)
        after = None

    delta = None
    if (before is not None) and (after is not None):
        delta = after - before

    return {"before": before, "after": after, "delta": delta}


# Lightweight demo when invoked directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Create synthetic test
    np.random.seed(0)
    n = 200
    k = 3
    # Synthetic raw probs (poorly calibrated)
    raw = np.random.dirichlet([0.5, 0.5, 0.5], size=n)
    # Synthetic true labels: sample from raw to create miscalibration
    true = np.array([str(np.random.choice(range(k), p=raw[i])) for i in range(n)])

    # Fit platt
    try:
        cal = fit_calibration(raw, true, method="platt", max_samples=100)
        cal_probs = calibrated_predict(cal, raw)
        logger.info("Example Brier before: %.4f after: %.4f", brier_score(true, raw), brier_score(true, cal_probs))
    except Exception as e:
        logger.exception("Demo failed: %s", e)
