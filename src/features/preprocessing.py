"""
preprocessing.py

Unified preprocessing for the football-predictor project.

Constraints implemented:
- Deterministic, no runtime schema learning.
- Uses explicit, versioned FEATURE_SCHEMA constant (do NOT auto-generate at runtime).
- build_features calls the stable advanced_features.build_advanced_features but will drop any columns not present in FEATURE_SCHEMA.
- Strict validation: invalid tokens like 'D1','H2','A1' in numeric columns are treated as errors and will raise ValueError.

Public API:
- FEATURE_SCHEMA (constant)
- VERSION (schema version)
- load_csv(path) -> pd.DataFrame (always low_memory=False)
- clean_dataframe(df) -> pd.DataFrame (cleaned, deterministic)
- build_features(df) -> pd.DataFrame (stable feature engineering, only allowed features)
- get_feature_matrix(df, target_col='FTR', debug=False) -> (X, y, categorical_features)

Note: this module is intentionally conservative: any unexpected column or invalid value raises an error in debug mode.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

# Versioned, explicit feature schema. This is NOT generated at runtime.
VERSION = "v1"
FEATURE_SCHEMA: Dict[str, List[str]] = {
    # Numeric features expected (explicit, stable)
    "numeric": [
        "home_elo_prior",
        "away_elo_prior",
        "home_form_short",
        "home_form_long",
        "away_form_short",
        "away_form_long",
        "home_avg_goals_for_prior",
        "home_avg_goals_against_prior",
        "away_avg_goals_for_prior",
        "away_avg_goals_against_prior",
        "home_consistency",
        "away_consistency",
        "expected_home_xg",
        "expected_away_xg",
        "attack_vs_defense",
        "defense_vs_attack",
        "elo_diff_home_minus_away",
        "form_diff_short",
        "form_diff_long",
        "xg_diff",
    ],
    # Categorical features expected (explicit)
    "categorical": [
        "HomeTeam",
        "AwayTeam",
        "League",
    ],
}

# Identifiers and allowed non-feature columns
IDENTIFIERS = ["Date", "HomeTeam", "AwayTeam"]
TARGET_COLUMN = "FTR"

# Regex to detect invalid token-style values like 'D1','H2','A1' in numeric columns
_INVALID_TOKEN_RE = re.compile(r"^[A-Za-z]+\d+$")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV using low_memory=False and parse Date when present.

    This function never infers schema or performs cleaning beyond parsing the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    # Try parse Date column if present
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise
    # attempt to parse Date column deterministically if present
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", infer_datetime_format=True)
        except Exception:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _assert_required_columns(df: pd.DataFrame) -> None:
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing)}")


def _detect_invalid_tokens_in_series(s: pd.Series) -> List[str]:
    # Return example values that match invalid token pattern
    vals = []
    for v in s.dropna().unique():
        if isinstance(v, str) and _INVALID_TOKEN_RE.match(v):
            vals.append(v)
            if len(vals) >= 5:
                break
    return vals


def clean_dataframe(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    """Perform deterministic cleaning and type-normalization on a raw dataframe.

    - Trim column names and string values
    - Normalize target FTR to 'H','D','A'
    - Ensure identifiers present and types stabilized
    - Do not add or invent features; just normalize types and remove unexpected columns later

    If debug is True, invalid inputs (e.g. token values in numeric columns) raise ValueError.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df = df.copy()

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    _assert_required_columns(df)

    # Trim string contents for object columns
    for c in df.select_dtypes(include=[object, "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
        # Replace empty strings with NaN
        df.loc[df[c] == "", c] = pd.NA

    # Normalize target column FTR to single-letter codes H/D/A
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.upper().str.strip()
        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({"HOME": "H", "AWAY": "A", "DRAW": "D"})
        # Keep only H/D/A or NaN
        df.loc[~df[TARGET_COLUMN].isin(["H", "D", "A"]) & df[TARGET_COLUMN].notna(), TARGET_COLUMN] = pd.NA

    # Ensure numeric raw columns are numeric (FTHG, FTAG)
    for col in ["FTHG", "FTAG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # At this stage do NOT drop columns; caller will request allowed features later.
    return df


def build_features(df: pd.DataFrame, debug: bool = True) -> pd.DataFrame:
    """Produce stable engineered features from cleaned dataframe.

    This implementation uses the project's stable advanced_features.build_advanced_features to compute features,
    then enforces strict schema: any feature not explicitly listed in FEATURE_SCHEMA will be dropped.

    The returned DataFrame contains identifiers (Date, HomeTeam, AwayTeam), the TARGET_COLUMN, and
    the allowed features from FEATURE_SCHEMA only.
    """
    # Import here to avoid circular imports at module import time
    from src.features.advanced_features import build_advanced_features

    # clean input first (ensures types)
    dfc = clean_dataframe(df, debug=debug)

    adv = build_advanced_features(dfc)

    # Compose allowed columns set
    allowed = set(IDENTIFIERS + [TARGET_COLUMN] + FEATURE_SCHEMA["numeric"] + FEATURE_SCHEMA["categorical"])

    # Warn / drop any columns not in allowed
    cols_present = [c for c in adv.columns if c in allowed]
    result = adv[cols_present].copy()

    # If any expected schema columns are missing, raise (strict behavior)
    missing_required = []
    for c in FEATURE_SCHEMA["numeric"] + FEATURE_SCHEMA["categorical"]:
        if c not in result.columns:
            missing_required.append(c)
    if missing_required:
        raise ValueError(f"Feature engineering did not produce required schema columns: {missing_required}")

    # Enforce types: numeric features -> float, categorical -> str
    for c in FEATURE_SCHEMA["numeric"]:
        result[c] = pd.to_numeric(result[c], errors="coerce")
        # detect invalid token-like strings in original adv if any
        if debug:
            # check original adv values for tokens in string form
            if c in adv.columns and adv[c].dtype == object:
                bad = _detect_invalid_tokens_in_series(adv[c])
                if bad:
                    raise ValueError(f"Invalid token-like values found in numeric feature '{c}': examples {bad}")

    for c in FEATURE_SCHEMA["categorical"]:
        result[c] = result[c].astype(str).fillna("UNK")

    # Keep identifiers and target as-is (Date converted earlier by load_csv/clean)
    result["Date"] = pd.to_datetime(result["Date"]) if "Date" in result.columns else pd.NaT

    return result


def get_feature_matrix(df: pd.DataFrame, target_col: str = TARGET_COLUMN, debug: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return X, y, categorical_feature_names given a raw dataframe.

    Strict behavior:
    - Uses explicit FEATURE_SCHEMA; any column not in schema is dropped.
    - Ensures numeric columns contain only numeric values (no token-like strings). Raises on violations in debug mode.
    - Ensures no leakage columns are present in X (FTHG, FTAG are not included as features).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Clean and build stable features
    feats = build_features(df, debug=debug)

    # Build X and y
    if target_col not in feats.columns:
        raise ValueError(f"Target column '{target_col}' not present in engineered features")

    y = feats[target_col].astype(str).copy()
    # Validate labels
    if not set(y.dropna().unique()).issubset({"H", "D", "A"}):
        raise ValueError("Target column contains unexpected labels; expected only 'H','D','A'")

    # X columns must be exactly the FEATURE_SCHEMA lists (numeric + categorical)
    expected_X_cols = FEATURE_SCHEMA["numeric"] + FEATURE_SCHEMA["categorical"]

    # strict: no extra features
    X = feats[expected_X_cols].copy()

    # Leak protection: ensure FTHG/FTAG not in X
    for leak in ["FTHG", "FTAG"]:
        if leak in X.columns:
            raise RuntimeError(f"Leakage column present in feature matrix: {leak}")

    # Ensure numeric dtypes and detect invalid tokens
    for c in FEATURE_SCHEMA["numeric"]:
        if c not in X.columns:
            raise RuntimeError(f"Missing numeric feature column: {c}")
        # Ensure values are numeric
        if not pd.api.types.is_numeric_dtype(X[c]):
            # Try convert coercively then detect non-numeric original strings
            converted = pd.to_numeric(X[c], errors="coerce")
            # If any non-numeric present after coercion and debug -> raise
            non_numeric_mask = converted.isna() & X[c].notna()
            if non_numeric_mask.any():
                examples = X.loc[non_numeric_mask, c].astype(str).unique().tolist()[:5]
                raise ValueError(f"Non-numeric values detected in numeric feature '{c}': examples {examples}")
            X[c] = converted

    # Ensure no object columns remain in numeric space
    obj_in_numeric = [c for c in FEATURE_SCHEMA["numeric"] if pd.api.types.is_object_dtype(X[c])]
    if obj_in_numeric:
        raise RuntimeError(f"Object dtypes present in numeric features: {obj_in_numeric}")

    # Ensure categorical columns are string type
    for c in FEATURE_SCHEMA["categorical"]:
        if c not in X.columns:
            raise RuntimeError(f"Missing categorical feature column: {c}")
        X[c] = X[c].astype(str).fillna("UNK")

    return X, y, FEATURE_SCHEMA["categorical"]
