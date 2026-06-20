"""ensemble_preprocessing.py

Unified, deterministic preprocessing layer shared by the CatBoost, LightGBM,
and XGBoost models in the ensemble.

Responsibilities
----------------
1. Build advanced numeric features (via ``build_advanced_features``).
2. Carry forward only the columns allowed by ``FEATURE_SCHEMA`` (numeric +
   categorical). Anything else (HTR, Time, Referee, B365*, ...) is dropped so
   that raw string columns never reach LightGBM or XGBoost.
3. Encode categorical columns consistently with a fitted ``{value: int}``
   mapping so that training and prediction use the same integer codes.
4. Encode the target label ``FTR`` (H/D/A) with a stable ``LabelEncoder`` so
   the order of classes is identical for all three models and persists
   across reload.
5. Persist the artifacts (column order, categorical maps, label encoder
   classes, schema version) to disk so that ``predict_ensemble`` can rebuild
   the exact same matrix after a model reload.

The output of ``fit_preprocessing`` and ``transform`` is a fully numeric
``pandas.DataFrame`` whose columns follow a single canonical order
(``feature_names``). This is what every model in the ensemble consumes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.preprocessing import (
    FEATURE_SCHEMA,
    TARGET_COLUMN,
    build_features,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


ARTIFACT_FILENAME = "ensemble_preprocess.json"
SCHEMA_VERSION = "ensemble-v1"
UNKNOWN_TOKEN = "UNK"
CANONICAL_LABEL_ORDER = ["H", "D", "A"]  # class index 0, 1, 2


# ---------------------------------------------------------------------------
# Artifact dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnsembleArtifacts:
    """All information required to rebuild a feature matrix at prediction time."""

    schema_version: str = SCHEMA_VERSION
    feature_names: List[str] = field(default_factory=list)
    cat_feature_names: List[str] = field(default_factory=list)
    cat_feature_indices: List[int] = field(default_factory=list)
    cat_maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    label_classes: List[str] = field(default_factory=list)

    # Helpers --------------------------------------------------------------

    @property
    def cat_indices_in_X(self) -> List[int]:
        """Integer positions of categorical columns within ``feature_names``."""
        return list(self.cat_feature_indices)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def label_to_int(self) -> Dict[str, int]:
        return {lab: i for i, lab in enumerate(self.label_classes)}

    @property
    def int_to_label(self) -> List[str]:
        return list(self.label_classes)

    # Serialization --------------------------------------------------------

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, model_dir) -> Path:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / ARTIFACT_FILENAME
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
        logger.info("Saved ensemble preprocessing artifacts to %s", path)
        return path

    @classmethod
    def load(cls, model_dir) -> "EnsembleArtifacts":
        model_dir = Path(model_dir)
        path = model_dir / ARTIFACT_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"Preprocessing artifact not found at {path}. "
                "Re-run train_ensemble() to regenerate it."
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        art = cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            feature_names=list(data.get("feature_names", [])),
            cat_feature_names=list(data.get("cat_feature_names", [])),
            cat_feature_indices=list(data.get("cat_feature_indices", [])),
            cat_maps={k: dict(v) for k, v in data.get("cat_maps", {}).items()},
            label_classes=list(data.get("label_classes", [])),
        )
        logger.info("Loaded ensemble preprocessing artifacts from %s", path)
        return art


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(s) -> str:
    """Convert a value to a clean string token; missing values -> UNK."""
    if s is None:
        return UNKNOWN_TOKEN
    try:
        if pd.isna(s):
            return UNKNOWN_TOKEN
    except (TypeError, ValueError):
        pass
    return str(s).strip() or UNKNOWN_TOKEN


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Run ``build_features`` and return the engineered frame in the canonical
    order defined by ``FEATURE_SCHEMA``.

    The result is guaranteed to contain:
      - exactly ``len(FEATURE_SCHEMA['numeric'])`` numeric columns
      - exactly ``len(FEATURE_SCHEMA['categorical'])`` categorical columns

    Any other columns (raw HTR, Time, Referee, B365*, ...) are dropped.
    """
    feats = build_features(df, debug=False)

    expected_numeric: List[str] = list(FEATURE_SCHEMA["numeric"])
    expected_categorical: List[str] = list(FEATURE_SCHEMA["categorical"])
    expected = expected_numeric + expected_categorical

    # Drop FTR if present in the engineered frame; it will be re-attached later
    feats = feats.drop(columns=[TARGET_COLUMN], errors="ignore")

    missing = [c for c in expected if c not in feats.columns]
    if missing:
        raise ValueError(
            "Feature engineering did not produce required columns: "
            f"{missing}. Available: {list(feats.columns)}"
        )

    # Coerce numeric columns to float (no string tokens allowed)
    for c in expected_numeric:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").astype(float)

    # Coerce categoricals to clean strings
    for c in expected_categorical:
        feats[c] = feats[c].map(_safe_str)

    # Return in canonical order, drop everything else
    return feats[expected].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_preprocessing(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, EnsembleArtifacts]:
    """Build the feature matrix, encode categoricals, and fit the label encoder.

    Returns
    -------
    X_encoded : pd.DataFrame
        Numeric matrix (categoricals are integer-encoded). Columns follow the
        canonical ``feature_names`` order.
    y_int : np.ndarray of int
        Label-encoded target.
    y_str : pd.Series of str
        Original string labels, useful for human-readable evaluation.
    artifacts : EnsembleArtifacts
        Everything you must persist alongside the model files in order to
        reproduce this matrix at prediction time.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # 1. Build features in canonical order, dropping raw string columns.
    feats = _build_feature_frame(df)

    # 2. Fit categorical maps on the training data only.
    cat_maps: Dict[str, Dict[str, int]] = {}
    cat_feature_names: List[str] = list(FEATURE_SCHEMA["categorical"])
    for c in cat_feature_names:
        uniques = sorted(set(feats[c].tolist()))
        # 0 reserved for unknown
        mapping = {v: i + 1 for i, v in enumerate(uniques)}
        cat_maps[c] = mapping

    # 3. Apply encoding (using fitted maps; anything unseen at predict time
    #    will fall back to 0 / UNKNOWN).
    encoded = feats.copy()
    for c in cat_feature_names:
        encoded[c] = encoded[c].map(lambda v, c=c: cat_maps[c].get(v, 0)).astype(np.int32)

    # 4. Build artifacts
    feature_names = list(encoded.columns)
    cat_feature_indices = [feature_names.index(c) for c in cat_feature_names]

    # 5. Encode target.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from input df")
    y_str = df[TARGET_COLUMN].map(_safe_str)
    valid_mask = y_str.isin(CANONICAL_LABEL_ORDER)
    if not valid_mask.all():
        n_bad = int((~valid_mask).sum())
        raise ValueError(
            f"Found {n_bad} rows with invalid FTR labels. "
            f"Expected one of {CANONICAL_LABEL_ORDER}; got examples: "
            f"{y_str[~valid_mask].unique().tolist()[:5]}"
        )

    # Force a stable label order matching CANONICAL_LABEL_ORDER
    label_classes = CANONICAL_LABEL_ORDER
    label_to_int = {lab: i for i, lab in enumerate(label_classes)}
    y_int = y_str.map(label_to_int).to_numpy(dtype=np.int32)

    # 6. Final matrix: numeric features stay float, categoricals stay int.
    #    Convert to float64 for the numeric columns (model friendly) and
    #    int32 for categoricals. CatBoost accepts int categoricals directly.
    for c in feature_names:
        if c in cat_feature_names:
            encoded[c] = encoded[c].astype(np.int32)
        else:
            encoded[c] = encoded[c].astype(np.float64)

    artifacts = EnsembleArtifacts(
        schema_version=SCHEMA_VERSION,
        feature_names=feature_names,
        cat_feature_names=cat_feature_names,
        cat_feature_indices=cat_feature_indices,
        cat_maps=cat_maps,
        label_classes=label_classes,
    )

    logger.info(
        "Preprocessing fit: %d rows, %d features (%d numeric, %d categorical), "
        "%d classes=%s",
        len(encoded),
        len(feature_names),
        len(feature_names) - len(cat_feature_names),
        len(cat_feature_names),
        len(label_classes),
        label_classes,
    )

    return encoded, y_int, y_str.reset_index(drop=True), artifacts


def transform(df: pd.DataFrame, artifacts: EnsembleArtifacts) -> pd.DataFrame:
    """Apply a fitted ``EnsembleArtifacts`` to a new dataframe.

    The result has the exact same columns and dtypes as the training matrix.
    Unknown categorical values map to 0 (UNKNOWN). Missing columns are
    filled with NaN / 0 as appropriate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    feats = _build_feature_frame(df)

    # Encode categoricals with the same maps as training
    encoded = feats.copy()
    for c in artifacts.cat_feature_names:
        mapping = artifacts.cat_maps.get(c, {})
        encoded[c] = encoded[c].map(lambda v, m=mapping: m.get(v, 0)).astype(np.int32)

    # Force numeric features to float
    for c in artifacts.feature_names:
        if c not in artifacts.cat_feature_names:
            encoded[c] = pd.to_numeric(encoded[c], errors="coerce").astype(np.float64)

    # Ensure column order matches
    encoded = encoded[artifacts.feature_names].copy()

    # Final dtype normalisation
    for c in artifacts.cat_feature_names:
        encoded[c] = encoded[c].astype(np.int32)

    return encoded


def decode_predictions(prob_arrays: List[np.ndarray], artifacts: EnsembleArtifacts) -> List[Dict[str, float]]:
    """Decode class-index probabilities back into H/D/A probabilities.

    Handles either a single 1-D vector or a 2-D matrix (rows, classes).
    """
    if not prob_arrays:
        return []

    canonical = list(CANONICAL_LABEL_ORDER)
    # The artifacts label order is canonical already; ensure align.
    label_to_idx = {lab: i for i, lab in enumerate(artifacts.label_classes)}

    out: List[Dict[str, float]] = []
    for arr in prob_arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row in arr:
            d = {lab: 0.0 for lab in canonical}
            for lab in canonical:
                if lab in label_to_idx and label_to_idx[lab] < row.shape[0]:
                    d[lab] = float(row[label_to_idx[lab]])
            out.append(d)
    return out
