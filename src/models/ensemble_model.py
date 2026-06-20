"""
ensemble_model.py

Ensemble training and prediction utilities combining CatBoost, LightGBM, and XGBoost.

Public API:
- train_ensemble(df, *, model_dir='models', weights=None, random_seed=42,
                 val_start_date=None, time_gap_days=0)
- load_ensemble_models(model_dir='models') -> dict
- predict_ensemble(models, feature_row_or_X, weights=None,
                   artifacts=None, model_dir=None) -> dict
- check_feature_consistency(feature_names_list) -> bool

Design notes
------------
- All three models share **one** preprocessing layer
  (``ensemble_preprocessing``). This guarantees that:
    * no raw string columns ever reach LightGBM or XGBoost;
    * categorical columns are encoded with a stable, persisted mapping;
    * the label encoder (H/D/A) is identical for all three models;
    * the feature column order used at training time is identical to the one
      used at prediction time, including after a model reload.
- Time-based train/validation split to avoid leakage.
- Logs per-model performance and saves artifacts.
- Combines probabilities via weighted averaging and renormalizes.
- Computes a model agreement score.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Delayed imports for ML libs
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    from sklearn.metrics import accuracy_score, log_loss
except Exception:  # pragma: no cover
    accuracy_score = None
    log_loss = None

from src.models.ensemble_preprocessing import (
    EnsembleArtifacts,
    CANONICAL_LABEL_ORDER,
    fit_preprocessing,
    transform,
    decode_predictions,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


DEFAULT_WEIGHTS = {"cat": 0.5, "lgb": 0.25, "xgb": 0.25}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_ml_libs():
    missing = []
    if CatBoostClassifier is None:
        missing.append("catboost")
    try:
        import lightgbm as lgb  # type: ignore
    except Exception:
        missing.append("lightgbm")
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        missing.append("xgboost")
    if missing:
        raise ImportError(
            f"Missing ML libraries: {', '.join(missing)}. Install to use ensemble training/prediction."
        )


def _time_split(
    df: pd.DataFrame,
    val_start_date: Optional[str] = None,
    time_gap_days: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "Date" not in df.columns:
        raise ValueError("train_ensemble requires a 'Date' column")
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
    train_df = d.loc[train_mask].reset_index(drop=True)
    val_df = d.loc[val_mask].reset_index(drop=True)
    logger.info(
        "Ensemble time split: train=%d val=%d cutoff=%s",
        len(train_df), len(val_df), str(cutoff),
    )
    return train_df, val_df


def check_feature_consistency(feature_sets: List[List[str]]) -> bool:
    """Ensure all feature sets are identical lists/sets of names."""
    if not feature_sets:
        return True
    base = list(feature_sets[0])
    for s in feature_sets[1:]:
        if list(s) != base:
            logger.error(
                "Feature mismatch detected. Base length %d vs other length %d",
                len(base), len(s),
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Per-model trainers
# ---------------------------------------------------------------------------


def _train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_feature_indices: List[int],
    random_seed: int = 42,
    cb_params: Optional[Dict] = None,
):
    """Train CatBoost.

    Categorical columns are passed by integer position so that CatBoost uses
    its native categorical handling, while everything else is treated as a
    numeric feature. The label is integer-encoded (0/1/2 = H/D/A).
    """
    if CatBoostClassifier is None:
        raise ImportError("catboost is required for CatBoost training")

    # CatBoost needs the original (string) labels for display, but it can also
    # train on integer labels with loss_function='MultiClass'. We use the
    # integer-encoded labels for full consistency with the other models.
    params = cb_params.copy() if cb_params else {
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

    model = CatBoostClassifier(**params)
    logger.info("Training CatBoost (%d rows, %d features, %d cat)",
                len(X_train), X_train.shape[1], len(cat_feature_indices))

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        cat_features=cat_feature_indices,  # list[int]
    )

    return model


def _train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_feature_indices: List[int],
    random_seed: int = 42,
    lgb_params: Optional[Dict] = None,
):
    """Train LightGBM with the shared encoded feature matrix."""
    import lightgbm as lgb  # type: ignore

    num_class = int(len(np.unique(np.concatenate([y_train, y_val]))))

    default = {
        "objective": "multiclass",
        "num_class": num_class,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "seed": random_seed,
        "metric": "multi_logloss",
        "verbosity": -1,
    }
    params = {**default, **(lgb_params or {})}

    # LightGBM accepts a DataFrame whose dtypes are int/float/bool.
    # Our preprocess guarantees that. We additionally declare which integer
    # columns are categorical so the booster treats them as such.
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_feature_indices or "auto",
        free_raw_data=False,
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
        categorical_feature=cat_feature_indices or "auto",
        free_raw_data=False,
    )

    logger.info("Training LightGBM (%d rows, %d features, %d cat)",
                len(X_train), X_train.shape[1], len(cat_feature_indices))

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    return booster


def _train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_feature_indices: List[int],
    random_seed: int = 42,
    xgb_params: Optional[Dict] = None,
):
    """Train XGBoost with the shared encoded feature matrix."""
    import xgboost as xgb  # type: ignore

    num_class = int(len(np.unique(np.concatenate([y_train, y_val]))))

    default = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eta": 0.05,
        "max_depth": 6,
        "seed": random_seed,
        "eval_metric": "mlogloss",
    }
    params = {**default, **(xgb_params or {})}

    # XGBoost expects only int/float/bool; our preprocess guarantees that.
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(X_val.columns))

    logger.info("Training XGBoost (%d rows, %d features)",
                len(X_train), X_train.shape[1])

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    return booster


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_models(models: Dict[str, object], model_dir: Union[str, Path]) -> Dict[str, str]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if "cat" in models and models["cat"] is not None:
        cat_path = model_dir / "ensemble_cat.cbm"
        models["cat"].save_model(str(cat_path))
        paths["cat"] = str(cat_path)
        logger.info("Saved CatBoost to %s", cat_path)

    if "lgb" in models and models["lgb"] is not None:
        import lightgbm as lgb  # type: ignore
        lgb_path = model_dir / "ensemble_lgb.txt"
        models["lgb"].save_model(str(lgb_path))
        paths["lgb"] = str(lgb_path)
        logger.info("Saved LightGBM to %s", lgb_path)

    if "xgb" in models and models["xgb"] is not None:
        import xgboost as xgb  # type: ignore
        # XGBoost 3.x infers the format from the file extension.
        # Use .ubj to silence the "Unknown file format" warning.
        xgb_path = model_dir / "ensemble_xgb.ubj"
        models["xgb"].save_model(str(xgb_path))
        paths["xgb"] = str(xgb_path)
        logger.info("Saved XGBoost to %s", xgb_path)

    return paths


def _load_models(model_dir: Union[str, Path]) -> Dict[str, Optional[object]]:
    model_dir = Path(model_dir)
    models: Dict[str, Optional[object]] = {"cat": None, "lgb": None, "xgb": None}

    cat_path = model_dir / "ensemble_cat.cbm"
    if cat_path.exists():
        if CatBoostClassifier is None:
            raise ImportError("catboost not installed to load CatBoost model")
        m = CatBoostClassifier()
        m.load_model(str(cat_path))
        models["cat"] = m
        logger.info("Loaded CatBoost from %s", cat_path)

    lgb_path = model_dir / "ensemble_lgb.txt"
    if lgb_path.exists():
        import lightgbm as lgb  # type: ignore
        booster = lgb.Booster(model_file=str(lgb_path))
        models["lgb"] = booster
        logger.info("Loaded LightGBM from %s", lgb_path)

    # XGBoost 3.x: prefer the .ubj extension; fall back to .model for
    # backwards compatibility with files saved by older versions of this code.
    xgb_path = model_dir / "ensemble_xgb.ubj"
    if not xgb_path.exists():
        xgb_path = model_dir / "ensemble_xgb.model"
    if xgb_path.exists():
        import xgboost as xgb  # type: ignore
        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        models["xgb"] = booster
        logger.info("Loaded XGBoost from %s", xgb_path)

    return models


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ensemble(
    df: pd.DataFrame,
    *,
    model_dir: str = "models",
    weights: Optional[Dict[str, float]] = None,
    random_seed: int = 42,
    val_start_date: Optional[str] = None,
    time_gap_days: int = 0,
) -> Dict:
    """Train CatBoost, LightGBM, and XGBoost on a single shared feature matrix.

    Returns a dict with trained model objects, paths, metrics per model,
    saved paths, weights, the preprocessing artifacts, and the exact
    ``feature_cols`` list used.
    """
    _require_ml_libs()

    if weights is None:
        weights = DEFAULT_WEIGHTS
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    weights = {k: float(v) / total for k, v in weights.items()}

    # ---- 1. Unified preprocessing on the full df -------------------------
    # This is the single source of truth for feature columns and types.
    X_all, y_all_int, y_all_str, artifacts = fit_preprocessing(df)

    # ---- 2. Time-based split ---------------------------------------------
    if "Date" not in df.columns:
        raise ValueError("train_ensemble requires a 'Date' column")
    df_idx = df.copy()
    df_idx["Date"] = pd.to_datetime(df_idx["Date"], errors="coerce")
    if df_idx["Date"].isna().any():
        n_bad = int(df_idx["Date"].isna().sum())
        raise ValueError(f"{n_bad} rows have unparseable Date values")

    df_idx = df_idx.reset_index(drop=True)
    train_df, val_df = _time_split(df_idx, val_start_date, time_gap_days)

    train_pos = train_df.index.to_numpy()
    val_pos = val_df.index.to_numpy()

    X_train = X_all.iloc[train_pos].reset_index(drop=True)
    X_val = X_all.iloc[val_pos].reset_index(drop=True)
    y_train = y_all_int[train_pos]
    y_val = y_all_int[val_pos]

    logger.info(
        "Train shape=%s val shape=%s, target distribution train=%s val=%s",
        X_train.shape, X_val.shape,
        dict(zip(*np.unique(y_train, return_counts=True))),
        dict(zip(*np.unique(y_val, return_counts=True))),
    )

    cat_idx = artifacts.cat_feature_indices
    metrics: Dict[str, Dict[str, float]] = {}
    models: Dict[str, Optional[object]] = {}

    # ---- 3. CatBoost ----------------------------------------------------
    try:
        cat_model = _train_catboost(
            X_train, y_train, X_val, y_val,
            cat_feature_indices=cat_idx,
            random_seed=random_seed,
        )
        models["cat"] = cat_model
        if accuracy_score is not None:
            y_pred_int = cat_model.predict(X_val).astype(int).ravel()
            y_prob = cat_model.predict_proba(X_val)
            metrics["cat"] = {
                "accuracy": float(accuracy_score(y_val, y_pred_int)),
                "log_loss": float(log_loss(y_val, y_prob)) if log_loss is not None else None,
            }
            logger.info("CatBoost val accuracy=%.4f log_loss=%s",
                        metrics["cat"]["accuracy"], metrics["cat"]["log_loss"])
    except Exception as e:
        logger.exception("CatBoost training failed: %s", e)
        models["cat"] = None

    # ---- 4. LightGBM ----------------------------------------------------
    try:
        lgb_model = _train_lightgbm(
            X_train, y_train, X_val, y_val,
            cat_feature_indices=cat_idx,
            random_seed=random_seed,
        )
        models["lgb"] = lgb_model
        if accuracy_score is not None:
            lgb_pred_probs = lgb_model.predict(X_val)
            if lgb_pred_probs.ndim == 1:
                lgb_pred_probs = lgb_pred_probs.reshape(-1, lgb_model.num_model_per_iteration() or 2)
            lgb_pred_int = np.argmax(lgb_pred_probs, axis=1)
            metrics["lgb"] = {
                "accuracy": float(accuracy_score(y_val, lgb_pred_int)),
                "log_loss": float(log_loss(y_val, lgb_pred_probs)) if log_loss is not None else None,
            }
            logger.info("LightGBM val accuracy=%.4f log_loss=%s",
                        metrics["lgb"]["accuracy"], metrics["lgb"]["log_loss"])
    except Exception as e:
        logger.exception("LightGBM training failed: %s", e)
        models["lgb"] = None

    # ---- 5. XGBoost -----------------------------------------------------
    try:
        xgb_model = _train_xgboost(
            X_train, y_train, X_val, y_val,
            cat_feature_indices=cat_idx,
            random_seed=random_seed,
        )
        models["xgb"] = xgb_model
        if accuracy_score is not None:
            import xgboost as xgb  # type: ignore
            dval = xgb.DMatrix(X_val, feature_names=list(X_val.columns))
            xgb_probs = xgb_model.predict(dval)
            if xgb_probs.ndim == 1:
                xgb_probs = xgb_probs.reshape(-1, int(getattr(xgb_model, "num_class", 3)))
            xgb_pred_int = np.argmax(xgb_probs, axis=1)
            metrics["xgb"] = {
                "accuracy": float(accuracy_score(y_val, xgb_pred_int)),
                "log_loss": float(log_loss(y_val, xgb_probs)) if log_loss is not None else None,
            }
            logger.info("XGBoost val accuracy=%.4f log_loss=%s",
                        metrics["xgb"]["accuracy"], metrics["xgb"]["log_loss"])
    except Exception as e:
        logger.exception("XGBoost training failed: %s", e)
        models["xgb"] = None

    # ---- 6. Save everything --------------------------------------------
    saved_paths = _save_models(models, model_dir)
    artifacts.save(model_dir)

    cfg = {"weights": weights}
    with open(Path(model_dir) / "ensemble_weights.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)

    return {
        "models": models,
        "metrics": metrics,
        "paths": saved_paths,
        "weights": weights,
        "feature_cols": list(artifacts.feature_names),
        "artifacts": artifacts,
    }


def load_ensemble_models(model_dir: str = "models") -> Dict[str, object]:
    """Load models and preprocessing artifacts from ``model_dir``."""
    models = _load_models(Path(model_dir))
    artifacts = EnsembleArtifacts.load(Path(model_dir))
    return {"models": models, "artifacts": artifacts}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_ensemble(
    models_or_bundle: Union[Dict[str, object], Dict],
    feature_row_or_X: Union[pd.DataFrame, pd.Series, None] = None,
    weights: Optional[Dict[str, float]] = None,
    *,
    artifacts: Optional[EnsembleArtifacts] = None,
    model_dir: Optional[Union[str, Path]] = None,
) -> Union[Dict, List[Dict]]:
    """Predict probabilities using the trained ensemble.

    Two calling styles are supported:

    1. ``predict_ensemble(bundle, X)`` — pass the dict returned by
       :func:`load_ensemble_models` (with ``models`` and ``artifacts`` keys),
       plus a DataFrame of raw features (matching the columns of the input
       dataset).
    2. ``predict_ensemble(models_dict, X, artifacts=art)`` — pass the model
       dict directly and provide the artifacts separately.

    The input is always run through the same preprocessing layer used at
    training time, which guarantees that:

    * raw string columns (HTR, Time, Referee, ...) are dropped;
    * categoricals are integer-encoded with the same mapping;
    * the column order matches what the models were trained on.
    """
    # ---- Unpack arguments ----------------------------------------------
    if isinstance(models_or_bundle, dict) and "models" in models_or_bundle and "artifacts" in models_or_bundle:
        models = models_or_bundle["models"]
        artifacts = models_or_bundle["artifacts"]
    else:
        models = models_or_bundle
        if artifacts is None and model_dir is not None:
            artifacts = EnsembleArtifacts.load(Path(model_dir))
        if artifacts is None:
            raise ValueError(
                "predict_ensemble needs preprocessing artifacts. Pass them via "
                "`artifacts=` or load them automatically by providing `model_dir`."
            )

    if weights is None:
        weights = DEFAULT_WEIGHTS
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    weights = {k: float(v) / total for k, v in weights.items()}

    # ---- Normalize input ------------------------------------------------
    single_input = False
    if feature_row_or_X is None:
        raise ValueError("feature_row_or_X must be a pandas DataFrame or Series")
    if isinstance(feature_row_or_X, pd.Series):
        X_raw = feature_row_or_X.to_frame().T
        single_input = True
    elif isinstance(feature_row_or_X, pd.DataFrame):
        X_raw = feature_row_or_X.copy()
        if X_raw.shape[0] == 1:
            single_input = True
    else:
        raise TypeError("feature_row_or_X must be a pandas DataFrame or Series")

    # ---- Apply the same preprocessing used at training time -------------
    X = transform(X_raw, artifacts)

    # ---- Collect predictions from each model ---------------------------
    prob_arrays: List[np.ndarray] = []
    model_names: List[str] = []
    canonical = list(CANONICAL_LABEL_ORDER)
    label_to_idx = {lab: i for i, lab in enumerate(artifacts.label_classes)}

    # CatBoost ------------------------------------------------------------
    if models.get("cat") is not None:
        try:
            m = models["cat"]
            p = np.asarray(m.predict_proba(X))
            prob_arrays.append(p)
            model_names.append("cat")
        except Exception as e:
            logger.exception("CatBoost predict failed: %s", e)

    # LightGBM ------------------------------------------------------------
    if models.get("lgb") is not None:
        try:
            import lightgbm as lgb  # type: ignore
            m = models["lgb"]
            p = np.asarray(m.predict(X))
            if p.ndim == 1:
                # Fallback: derive num_class from artifacts
                p = p.reshape(-1, len(artifacts.label_classes))
            prob_arrays.append(p)
            model_names.append("lgb")
        except Exception as e:
            logger.exception("LightGBM predict failed: %s", e)

    # XGBoost -------------------------------------------------------------
    if models.get("xgb") is not None:
        try:
            import xgboost as xgb  # type: ignore
            m = models["xgb"]
            dmat = xgb.DMatrix(X, feature_names=list(X.columns))
            p = np.asarray(m.predict(dmat))
            if p.ndim == 1:
                p = p.reshape(-1, len(artifacts.label_classes))
            prob_arrays.append(p)
            model_names.append("xgb")
        except Exception as e:
            logger.exception("XGBoost predict failed: %s", e)

    if not prob_arrays:
        raise RuntimeError("No model predictions available")

    # ---- Align class order to CANONICAL_LABEL_ORDER ---------------------
    aligned_probs: List[np.ndarray] = []
    for arr in prob_arrays:
        # If labels line up with the canonical order, just copy.
        # Otherwise we re-order columns.
        if arr.shape[1] == len(label_to_idx):
            aligned = np.zeros((arr.shape[0], len(canonical)), dtype=float)
            for lab, j in label_to_idx.items():
                if lab in canonical:
                    aligned[:, canonical.index(lab)] = arr[:, j]
            aligned_probs.append(aligned)
        else:
            # Defensive: pad / truncate
            n = min(arr.shape[1], len(canonical))
            aligned = np.zeros((arr.shape[0], len(canonical)), dtype=float)
            aligned[:, :n] = arr[:, :n]
            aligned_probs.append(aligned)

    # ---- Weighted average ----------------------------------------------
    n_rows = aligned_probs[0].shape[0]
    avg = np.zeros((n_rows, len(canonical)), dtype=float)
    for name, prob in zip(model_names, aligned_probs):
        avg += weights.get(name, 0.0) * prob

    row_sums = avg.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = avg / row_sums
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = normalized / row_sums

    # ---- Model agreement score -----------------------------------------
    agreement = []
    for i in range(n_rows):
        top = []
        for prob in aligned_probs:
            top.append(canonical[int(np.argmax(prob[i]))])
        if not top:
            agreement.append(0.0)
            continue
        vals, counts = np.unique(top, return_counts=True)
        agreement.append(float(counts.max()) / float(len(top)))

    # ---- Final output ---------------------------------------------------
    outputs: List[Dict[str, float]] = []
    for i in range(n_rows):
        outputs.append({
            "home_win": float(normalized[i, canonical.index("H")]),
            "draw": float(normalized[i, canonical.index("D")]),
            "away_win": float(normalized[i, canonical.index("A")]),
            "model_agreement_score": float(agreement[i]),
        })

    return outputs[0] if single_input else outputs


# Backwards-compatible alias
def load_ensemble(model_dir: str = "models") -> Dict[str, Optional[object]]:
    return load_ensemble_models(model_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    merged_path = Path("data/processed/merged_dataset.csv")
    if not merged_path.exists():
        logger.error(
            "Merged dataset not found at %s. Generate it first "
            "(e.g. `python scripts/generate_sample_data.py`).",
            merged_path,
        )
        raise SystemExit(1)
    df = pd.read_csv(merged_path, parse_dates=["Date"])
    logger.info("Loaded dataset with %d rows", len(df))

    res = train_ensemble(df, model_dir="models")
    logger.info("Ensemble training complete.")
    logger.info("Metrics: %s", res["metrics"])
    logger.info("Paths: %s", res["paths"])
    logger.info("Feature columns (%d): %s",
                len(res["feature_cols"]), res["feature_cols"])

    # Demonstrate that prediction works after model reload
    logger.info("Reloading models from disk to verify persistence...")
    bundle = load_ensemble_models("models")
    logger.info("Reloaded models and artifacts. Feature columns: %s",
                bundle["artifacts"].feature_names)

    # Use the last validation rows as a smoke test
    sample = df.tail(5).reset_index(drop=True)
    preds = predict_ensemble(bundle, sample)
    logger.info("Sample predictions on last 5 rows: %s", preds)
