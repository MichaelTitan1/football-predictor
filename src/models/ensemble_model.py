"""
ensemble_model.py

Ensemble training and prediction utilities combining CatBoost, LightGBM, and XGBoost.

Public API:
- train_ensemble(df, *, model_dir='models', weights=None, random_seed=42, val_start_date=None, time_gap_days=0)
- load_ensemble_models(model_dir='models') -> dict
- predict_ensemble(models, feature_row_or_X, weights=None) -> dict
- check_feature_consistency(feature_names_list) -> bool

Design notes:
- Uses advanced features only (build_advanced_features)
- Time-based train/validation split to avoid leakage
- Logs per-model performance and saves artifacts
- Combines probabilities via weighted averaging and renormalizes
- Computes a model agreement score

Requirements:
- catboost, lightgbm, xgboost, scikit-learn (for metrics) recommended
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# delayed imports for ML libs
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# lightgbm and xgboost imported inside functions to allow partial environments

try:
    from sklearn.metrics import accuracy_score, log_loss
except Exception:
    accuracy_score = None
    log_loss = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


DEFAULT_WEIGHTS = {"cat": 0.5, "lgb": 0.25, "xgb": 0.25}


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
        raise ImportError(f"Missing ML libraries: {', '.join(missing)}. Install to use ensemble training/prediction.")


def _time_split(df: pd.DataFrame, val_start_date: Optional[str] = None, time_gap_days: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    logger.info("Ensemble time split: train=%d val=%d cutoff=%s", len(train_df), len(val_df), str(cutoff))
    return train_df, val_df


def _prepare_advanced_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    from src.features.advanced_features import build_advanced_features

    adv = build_advanced_features(df)
    if "FTR" not in adv.columns:
        raise ValueError("Advanced features must include target column FTR")
    y = adv["FTR"].astype(str)
    exclude = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "League"}
    feature_cols = [c for c in adv.columns if c not in exclude]
    X = adv[feature_cols]
    # Fill numeric NaNs with median, categorical with 'UNK'
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna("UNK").astype(str)
    return X, y, feature_cols


def check_feature_consistency(feature_sets: List[List[str]]) -> bool:
    """Ensure all feature sets are identical lists/sets of names."""
    if not feature_sets:
        return True
    base = list(feature_sets[0])
    for s in feature_sets[1:]:
        if list(s) != base:
            logger.error("Feature mismatch detected. Base length %d vs other length %d", len(base), len(s))
            return False
    return True


def _train_catboost(X_train, y_train, X_val, y_val, random_seed=42, cb_params: Optional[Dict] = None):
    if CatBoostClassifier is None:
        raise ImportError("catboost is required for CatBoost training")

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
    logger.info("Training CatBoost (%d rows)", len(X_train))

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        cat_features="auto"   # ✅ FIX ADDED HERE
    )

    return model

def _train_lightgbm(X_train, y_train, X_val, y_val, random_seed=42, lgb_params: Optional[Dict] = None):
    import lightgbm as lgb  # type: ignore

    default = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "seed": random_seed,
        "metric": "multi_logloss",
        "verbosity": -1,
    }

    params = {**default, **(lgb_params or {})}

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    logger.info("Training LightGBM (%d rows)", len(X_train))

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )

    return booster


def _train_xgboost(X_train, y_train, X_val, y_val, random_seed=42, xgb_params: Optional[Dict] = None):
    import xgboost as xgb  # type: ignore

    num_class = len(np.unique(y_train))

    default = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eta": 0.05,
        "max_depth": 6,
        "seed": random_seed,
        "eval_metric": "mlogloss",
    }

    params = {**default, **(xgb_params or {})}

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    logger.info("Training XGBoost (%d rows)", len(X_train))

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    return booster

def _save_models(models: Dict[str, object], model_dir: Union[str, Path]):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # CatBoost
    if "cat" in models and models["cat"] is not None:
        cat_path = model_dir / "ensemble_cat.cbm"
        models["cat"].save_model(str(cat_path))
        paths["cat"] = str(cat_path)
        logger.info("Saved CatBoost to %s", cat_path)
    # LightGBM
    if "lgb" in models and models["lgb"] is not None:
        import lightgbm as lgb  # type: ignore

        lgb_path = model_dir / "ensemble_lgb.txt"
        models["lgb"].save_model(str(lgb_path))
        paths["lgb"] = str(lgb_path)
        logger.info("Saved LightGBM to %s", lgb_path)
    # XGBoost
    if "xgb" in models and models["xgb"] is not None:
        import xgboost as xgb  # type: ignore

        xgb_path = model_dir / "ensemble_xgb.model"
        models["xgb"].save_model(str(xgb_path))
        paths["xgb"] = str(xgb_path)
        logger.info("Saved XGBoost to %s", xgb_path)

    return paths


def _load_models(model_dir: Union[str, Path]) -> Dict[str, Optional[object]]:
    model_dir = Path(model_dir)
    models = {"cat": None, "lgb": None, "xgb": None}
    # CatBoost
    cat_path = model_dir / "ensemble_cat.cbm"
    if cat_path.exists():
        if CatBoostClassifier is None:
            raise ImportError("catboost not installed to load CatBoost model")
        m = CatBoostClassifier()
        m.load_model(str(cat_path))
        models["cat"] = m
        logger.info("Loaded CatBoost from %s", cat_path)
    # LightGBM
    lgb_path = model_dir / "ensemble_lgb.txt"
    if lgb_path.exists():
        import lightgbm as lgb  # type: ignore

        booster = lgb.Booster(model_file=str(lgb_path))
        models["lgb"] = booster
        logger.info("Loaded LightGBM from %s", lgb_path)
    # XGBoost
    xgb_path = model_dir / "ensemble_xgb.model"
    if xgb_path.exists():
        import xgboost as xgb  # type: ignore

        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        models["xgb"] = booster
        logger.info("Loaded XGBoost from %s", xgb_path)
    return models


def train_ensemble(
    df: pd.DataFrame,
    *,
    model_dir: str = "models",
    weights: Optional[Dict[str, float]] = None,
    random_seed: int = 42,
    val_start_date: Optional[str] = None,
    time_gap_days: int = 0,
) -> Dict:
    """Train CatBoost, LightGBM, and XGBoost on advanced features (time-based split).

    Returns dict with trained model objects, paths, metrics per model, and saved paths.
    """
    _require_ml_libs()

    if weights is None:
        weights = DEFAULT_WEIGHTS
    # Normalize weights
    total = float(sum(weights.values()))
    weights = {k: float(v) / total for k, v in weights.items()}

    # Prepare advanced X/y and features
    X_all, y_all, feature_cols = _prepare_advanced_Xy(df)
    adv = df.copy()
    adv["Date"] = pd.to_datetime(adv["Date"]) if "Date" in adv.columns else pd.to_datetime(X_all.index)

    # Time split on adv
    train_df, val_df = _time_split(adv, val_start_date, time_gap_days)

    # Map to indices in adv (Date+teams key)
    adv_keys = adv[["Date", "HomeTeam", "AwayTeam"]].astype(str).agg("__".join, axis=1)
    train_keys = train_df[["Date", "HomeTeam", "AwayTeam"]].astype(str).agg("__".join, axis=1)
    val_keys = val_df[["Date", "HomeTeam", "AwayTeam"]].astype(str).agg("__".join, axis=1)
    key_to_index = {k: i for i, k in enumerate(adv_keys)}
    train_idx = [key_to_index[k] for k in train_keys if k in key_to_index]
    val_idx = [key_to_index[k] for k in val_keys if k in key_to_index]

    X_train = X_all.iloc[train_idx].reset_index(drop=True)
    y_train = y_all.iloc[train_idx].reset_index(drop=True)
    X_val = X_all.iloc[val_idx].reset_index(drop=True)
    y_val = y_all.iloc[val_idx].reset_index(drop=True)

    # Encode labels to numeric for LightGBM/XGBoost
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    metrics = {}
    models = {}

    # Train CatBoost
    try:
        cat_model = _train_catboost(X_train, y_train, X_val, y_val, random_seed=random_seed)
        models["cat"] = cat_model
        # evaluate
        if accuracy_score is not None:
            y_pred = cat_model.predict(X_val)
            y_prob = cat_model.predict_proba(X_val)
            acc = float(accuracy_score(y_val, y_pred))
            ll = float(log_loss(y_val, y_prob, labels=list(cat_model.classes_))) if log_loss is not None else None
            metrics["cat"] = {"accuracy": acc, "log_loss": ll}
            logger.info("CatBoost val accuracy=%.4f log_loss=%s", acc, str(ll))
    except Exception as e:
        logger.exception("CatBoost training failed: %s", e)
        models["cat"] = None

    # Train LightGBM
    try:
        lgb_model = _train_lightgbm(X_train, y_train_enc, X_val, y_val_enc, random_seed=random_seed)
        models["lgb"] = lgb_model
        if accuracy_score is not None:
            lgb_pred_probs = lgb_model.predict(X_val)
            lgb_preds = np.argmax(lgb_pred_probs, axis=1)
            acc = float(accuracy_score(y_val_enc, lgb_preds))
            ll = float(log_loss(y_val_enc, lgb_pred_probs)) if log_loss is not None else None
            metrics["lgb"] = {"accuracy": acc, "log_loss": ll}
            logger.info("LightGBM val accuracy=%.4f log_loss=%s", acc, str(ll))
    except Exception as e:
        logger.exception("LightGBM training failed: %s", e)
        models["lgb"] = None

    # Train XGBoost
    try:
        xgb_model = _train_xgboost(X_train, y_train_enc, X_val, y_val_enc, random_seed=random_seed)
        models["xgb"] = xgb_model
        if accuracy_score is not None:
            import xgboost as xgb  # type: ignore

            dval = xgb.DMatrix(X_val)
            xgb_probs = xgb_model.predict(dval)
            xgb_preds = np.argmax(xgb_probs, axis=1)
            acc = float(accuracy_score(y_val_enc, xgb_preds))
            ll = float(log_loss(y_val_enc, xgb_probs)) if log_loss is not None else None
            metrics["xgb"] = {"accuracy": acc, "log_loss": ll}
            logger.info("XGBoost val accuracy=%.4f log_loss=%s", acc, str(ll))
    except Exception as e:
        logger.exception("XGBoost training failed: %s", e)
        models["xgb"] = None

    # Save models
    saved_paths = _save_models(models, model_dir)

    # Save weights config
    cfg = {"weights": weights}
    with open(Path(model_dir) / "ensemble_weights.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)

    return {"models": models, "metrics": metrics, "paths": saved_paths, "weights": weights, "feature_cols": feature_cols}


def load_ensemble_models(model_dir: str = "models") -> Dict[str, Optional[object]]:
    return _load_models(Path(model_dir))


def predict_ensemble(
    models: Dict[str, object],
    feature_row_or_X: Union[pd.DataFrame, pd.Series],
    weights: Optional[Dict[str, float]] = None
) -> Dict:
    """Predict probabilities and aggregate using weighted averaging."""

    if weights is None:
        weights = DEFAULT_WEIGHTS

    total = sum(weights.values())
    weights = {k: float(v) / total for k, v in weights.items()}

    single_input = False

    if isinstance(feature_row_or_X, pd.Series):
        X = feature_row_or_X.to_frame().T
        single_input = True

    elif isinstance(feature_row_or_X, pd.DataFrame):
        X = feature_row_or_X.copy()
        if X.shape[0] == 1:
            single_input = True

    else:
        raise ValueError("feature_row_or_X must be a pandas DataFrame or Series")

    # ----------------------------
    # COLLECT MODEL PREDICTIONS
    # ----------------------------
    prob_arrays = []
    class_orders = []
    model_names = []

    # CatBoost
    if models.get("cat") is not None:
        try:
            m = models["cat"]
            p = m.predict_proba(X)
            prob_arrays.append(np.asarray(p))
            class_orders.append(list(m.classes_))
            model_names.append("cat")
        except Exception as e:
            logger.exception("CatBoost predict failed: %s", e)

    # LightGBM
    if models.get("lgb") is not None:
        try:
            import lightgbm as lgb  # type: ignore
            m = models["lgb"]
            p = m.predict(X)
            p = np.asarray(p)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            prob_arrays.append(p)
            class_orders.append(None)
            model_names.append("lgb")
        except Exception as e:
            logger.exception("LightGBM predict failed: %s", e)

    # XGBoost
    if models.get("xgb") is not None:
        try:
            import xgboost as xgb  # type: ignore
            m = models["xgb"]
            dmat = xgb.DMatrix(X)
            p = m.predict(dmat)
            p = np.asarray(p)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            prob_arrays.append(p)
            class_orders.append(None)
            model_names.append("xgb")
        except Exception as e:
            logger.exception("XGBoost predict failed: %s", e)

    if len(prob_arrays) == 0:
        raise RuntimeError("No model predictions available")

    # ----------------------------
    # CLASS ORDER HANDLING
    # ----------------------------
    canonical = None
    if models.get("cat") is not None and hasattr(models["cat"], "classes_"):
        canonical = list(models["cat"].classes_)
    else:
        canonical = ["H", "D", "A"]

    # ----------------------------
    # ALIGN PROBABILITIES
    # ----------------------------
    probs_aligned = []

    for idx, arr in enumerate(prob_arrays):
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        k = arr.shape[1]
        co = class_orders[idx]

        aligned = np.zeros((arr.shape[0], len(canonical)), dtype=float)

        if co is None:
            min_k = min(k, len(canonical))
            aligned[:, :min_k] = arr[:, :min_k]
        else:
            for j, lab in enumerate(canonical):
                if lab in co:
                    aligned[:, j] = arr[:, co.index(lab)]

        probs_aligned.append(aligned)

    # ----------------------------
    # WEIGHTED AVERAGE
    # ----------------------------
    n_rows = probs_aligned[0].shape[0]
    avg_probs = np.zeros((n_rows, len(canonical)), dtype=float)

    for model_name, prob in zip(model_names, probs_aligned):
        w = weights.get(model_name, 0.0)
        avg_probs += w * prob

    # ----------------------------
    # NORMALIZATION (SAFE)
    # ----------------------------
    row_sums = avg_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = avg_probs / row_sums

    # FINAL SAFETY CHECK
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized = normalized / row_sums

    # ----------------------------
    # AGREEMENT SCORE
    # ----------------------------
    agreement_scores = []

    for i in range(n_rows):
        top_labels = []

        for prob in probs_aligned:
            top = int(np.argmax(prob[i]))
            top_labels.append(canonical[top])

        vals, counts = np.unique(top_labels, return_counts=True)
        agreement_scores.append(float(counts.max()) / float(len(top_labels)))

    # ----------------------------
    # OUTPUT
    # ----------------------------
    outputs = []

    for i in range(n_rows):
        outputs.append({
            "home_win": float(normalized[i, canonical.index("H")]) if "H" in canonical else 0.0,
            "draw": float(normalized[i, canonical.index("D")]) if "D" in canonical else 0.0,
            "away_win": float(normalized[i, canonical.index("A")]) if "A" in canonical else 0.0,
            "model_agreement_score": float(agreement_scores[i]),
        })

    return outputs[0] if single_input else outputs


# Expose simple wrappers
def load_ensemble(model_dir: str = "models") -> Dict[str, Optional[object]]:
    return _load_models(Path(model_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Demo: require merged dataset and models
    merged_path = Path("data/processed/merged_dataset.csv")
    if not merged_path.exists():
        logger.error("Merged dataset not found. Please prepare data before using ensemble trainer")
        raise SystemExit(1)
    df = pd.read_csv(merged_path, parse_dates=["Date"]) if merged_path.exists() else None
    if df is None:
        raise SystemExit(1)
    # Train ensemble
    res = train_ensemble(df)
    logger.info("Ensemble training complete. Paths: %s", res.get("paths"))
