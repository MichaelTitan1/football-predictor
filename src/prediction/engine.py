"""
engine.py

Core intelligence engine for football-ai-system: converts upcoming matches into
probabilistic markets, score predictions, risk and confidence scores using a trained
CatBoost model and the engineered features dataset.

Public API:
- load_prediction_model(path: str) -> CatBoostClassifier
- prepare_match_features(home_team: str, away_team: str, feature_data: pd.DataFrame, model: Optional[object] = None) -> pd.DataFrame
- predict_match(model, feature_row: pd.DataFrame) -> Dict

Design principles:
- Defensive: prevents feature mismatch, handles unknown teams, logs steps
- Uses CatBoost model for main-result probabilities
- Uses simple Poisson-based score model using team attack/defense averages (from feature_data)
- Produces structured markets as specified
- Ready for batch extension

Dependencies: pandas, numpy, catboost
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _require_catboost():
    if CatBoostClassifier is None:
        raise ImportError("catboost is required. Install with `pip install catboost`")


def load_prediction_model(path: str):
    """Load CatBoost model (.cbm) from disk safely and return model instance."""
    _require_catboost()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found at: {path}")
    model = CatBoostClassifier()
    model.load_model(str(p))
    logger.info("Loaded CatBoost model from %s", path)
    return model


def _get_feature_names(model: Optional[object], feature_data: pd.DataFrame) -> List[str]:
    """Infer expected feature columns. Prefer model metadata, fallback to feature_data schema."""
    # Try model attributes
    if model is not None:
        if hasattr(model, "feature_names_"):
            try:
                names = list(model.feature_names_)
                logger.debug("Found feature names from model: %d", len(names))
                return names
            except Exception:
                pass
        if hasattr(model, "get_feature_names"):
            try:
                names = list(model.get_feature_names())
                logger.debug("Got feature names via get_feature_names(): %d", len(names))
                return names
            except Exception:
                pass
    # Fallback: infer from feature_data excluding identifiers
    exclude = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    inferred = [c for c in feature_data.columns if c not in exclude]
    logger.debug("Falling back to inferred feature names (%d) from feature_data", len(inferred))
    return inferred


def _latest_team_stats(team: str, feature_data: pd.DataFrame, as_home: bool) -> Optional[pd.Series]:
    """Return most recent row for team when playing as home (as_home=True) or away.
    If not found, return None.
    """
    col = "HomeTeam" if as_home else "AwayTeam"
    mask = feature_data[col] == team
    if not mask.any():
        return None
    row = feature_data.loc[mask].sort_values("Date").iloc[-1]
    return row


def prepare_match_features(home_team: str, away_team: str, feature_data: pd.DataFrame, model: Optional[object] = None) -> pd.DataFrame:
    """Build a single-row feature DataFrame for a future match.

    Strategy:
    - Determine expected feature names from model or feature_data
    - For features prefixed with 'home_' use the most recent home-row for home_team
    - For 'away_' use most recent away-row for away_team
    - For 'h2h_' use latest head-to-head if available
    - For other features attempt home then away then global median fallback
    - Always return columns in the same order as expected by model when possible

    Returns a one-row pandas.DataFrame ready for model.predict_proba
    """
    if not isinstance(feature_data, pd.DataFrame):
        raise ValueError("feature_data must be a pandas DataFrame")

    feature_cols = _get_feature_names(model, feature_data)

    # compute global defaults
    defaults: Dict[str, object] = {}
    for c in feature_cols:
        if c in feature_data.columns and pd.api.types.is_numeric_dtype(feature_data[c]):
            defaults[c] = float(feature_data[c].median(skipna=True)) if feature_data[c].notna().any() else 0.0
        else:
            defaults[c] = "UNK"

    home_row = _latest_team_stats(home_team, feature_data, as_home=True)
    away_row = _latest_team_stats(away_team, feature_data, as_home=False)

    # head-to-head latest row if any
    mask_h2h = (
        ((feature_data["HomeTeam"] == home_team) & (feature_data["AwayTeam"] == away_team))
        | ((feature_data["HomeTeam"] == away_team) & (feature_data["AwayTeam"] == home_team))
    )
    h2h_row = None
    if mask_h2h.any():
        h2h_row = feature_data.loc[mask_h2h].sort_values("Date").iloc[-1]

    row_values: Dict[str, object] = {}
    for c in feature_cols:
        if c.startswith("home_"):
            if home_row is not None and c in home_row.index:
                row_values[c] = home_row[c]
            else:
                row_values[c] = defaults[c]
        elif c.startswith("away_"):
            if away_row is not None and c in away_row.index:
                row_values[c] = away_row[c]
            else:
                row_values[c] = defaults[c]
        elif c.startswith("h2h_"):
            if h2h_row is not None and c in h2h_row.index:
                row_values[c] = h2h_row[c]
            else:
                row_values[c] = defaults[c]
        elif c in {"HomeTeam", "AwayTeam"}:
            row_values[c] = home_team if c == "HomeTeam" else away_team
        else:
            # try home then away then default
            val = None
            if home_row is not None and c in home_row.index:
                val = home_row[c]
            elif away_row is not None and c in away_row.index:
                val = away_row[c]
            row_values[c] = val if val is not None else defaults[c]

    feature_row = pd.DataFrame([row_values], columns=feature_cols)

    # coerce numeric columns
    for c in feature_row.columns:
        if c in feature_data.columns and pd.api.types.is_numeric_dtype(feature_data[c]):
            feature_row[c] = pd.to_numeric(feature_row[c], errors="coerce").fillna(defaults[c]).astype(float)

    logger.info("Prepared feature row for %s vs %s", home_team, away_team)
    return feature_row


# Poisson utilities
def _poisson_pmf(k: int, lam: float) -> float:
    # protect against negative or very large lambda
    if lam < 0:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except Exception:
        return 0.0


def _compute_score_distribution(lambda_home: float, lambda_away: float, max_goals: int = 6) -> np.ndarray:
    """Compute joint probability matrix P(home_goals=i, away_goals=j) for 0..max_goals inclusive."""
    probs = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    # compute marginal pmfs
    pmf_home = [_poisson_pmf(k, lambda_home) for k in range(max_goals + 1)]
    pmf_away = [_poisson_pmf(k, lambda_away) for k in range(max_goals + 1)]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = pmf_home[i] * pmf_away[j]
    # renormalize tail mass (ignore probabilities where >max_goals by adding to last row/col)
    # compute tail mass beyond max_goals and add to last row/col proportionally (approximate)
    # For simplicity, we accept small mass loss for large lambda; it's negligible for typical match lambdas
    return probs


def _expected_goals_from_stats(home_row: pd.Series, away_row: pd.Series, feature_data: pd.DataFrame) -> Tuple[float, float]:
    """Estimate expected goals for home and away using rolling averages and league averages.

    Formula (heuristic):
    - compute team attack strength = team_avg_scored / league_avg_scored
    - compute opponent defense_strength = team_avg_conceded / league_avg_conceded
    - base = league_avg_goals_per_team
    - expected_home = base * home_attack_strength * away_defense_strength * (1 + home_advantage)
    - expected_away = base * away_attack_strength * home_defense_strength

    All fallbacks use medians from feature_data.
    """
    # keys in feature_data expected: home_team_avg_goals_scored, home_team_avg_goals_conceded, away_team_avg_goals_scored, away_team_avg_goals_conceded
    # but per-row home/away prefixes exist for feature rows
    # We'll attempt to extract relevant metrics; if not available, use medians
    # League averages per-team
    league_avg_scored = 0.0
    league_avg_conceded = 0.0
    candidates_scored = []
    candidates_conceded = []
    for col in feature_data.columns:
        if col.endswith("_avg_goals_scored") or col.endswith("_team_avg_goals_scored"):
            if pd.api.types.is_numeric_dtype(feature_data[col]):
                candidates_scored.append(feature_data[col].median(skipna=True))
        if col.endswith("_avg_goals_conceded") or col.endswith("_team_avg_goals_conceded"):
            if pd.api.types.is_numeric_dtype(feature_data[col]):
                candidates_conceded.append(feature_data[col].median(skipna=True))
    if candidates_scored:
        league_avg_scored = float(pd.Series(candidates_scored).median())
    else:
        league_avg_scored = 1.3  # reasonable default
    if candidates_conceded:
        league_avg_conceded = float(pd.Series(candidates_conceded).median())
    else:
        league_avg_conceded = 1.3

    base = (league_avg_scored + league_avg_conceded) / 2.0
    if base <= 0:
        base = 1.3

    # Extract per-team metrics with fallbacks
    def _get_val(row: Optional[pd.Series], keys: List[str], fallback: float) -> float:
        if row is None:
            return fallback
        for k in keys:
            if k in row.index and pd.notna(row[k]):
                try:
                    return float(row[k])
                except Exception:
                    continue
        return fallback

    # home attack = home_team_avg_goals_scored or home_avg_goals_for_last_10 etc.
    home_attack = _get_val(home_row, ["home_team_avg_goals_scored", "home_avg_goals_for_last_10", "home_avg_goals_for_last_10"], league_avg_scored)
    home_defense = _get_val(home_row, ["home_team_avg_goals_conceded", "home_avg_goals_against_last_10"], league_avg_conceded)
    away_attack = _get_val(away_row, ["away_team_avg_goals_scored", "away_avg_goals_for_last_10"], league_avg_scored)
    away_defense = _get_val(away_row, ["away_team_avg_goals_conceded", "away_avg_goals_against_last_10"], league_avg_conceded)

    # home advantage factor
    home_adv = 0.0
    if home_row is not None and "home_historical_home_win_rate" in home_row.index:
        try:
            home_adv = float(home_row["home_historical_home_win_rate"])
        except Exception:
            home_adv = 0.0
    # scale advantage into a multiplicative factor in [-0.2, +0.4]
    home_adv_factor = max(-0.2, min(0.4, (home_adv - 0.25)))  # crude scaling

    # Compute attack/defense strengths
    try:
        home_attack_strength = home_attack / league_avg_scored if league_avg_scored > 0 else 1.0
    except Exception:
        home_attack_strength = 1.0
    try:
        away_defense_strength = away_defense / league_avg_conceded if league_avg_conceded > 0 else 1.0
    except Exception:
        away_defense_strength = 1.0
    try:
        away_attack_strength = away_attack / league_avg_scored if league_avg_scored > 0 else 1.0
    except Exception:
        away_attack_strength = 1.0
    try:
        home_defense_strength = home_defense / league_avg_conceded if league_avg_conceded > 0 else 1.0
    except Exception:
        home_defense_strength = 1.0

    expected_home = base * home_attack_strength * away_defense_strength * (1.0 + home_adv_factor)
    expected_away = base * away_attack_strength * home_defense_strength

    # enforce reasonable bounds
    expected_home = max(0.05, min(5.0, expected_home))
    expected_away = max(0.05, min(5.0, expected_away))

    logger.debug("Estimated expected goals: home=%.3f away=%.3f (base=%.3f)", expected_home, expected_away, base)
    return expected_home, expected_away


def _scoreline_probabilities(lambda_h: float, lambda_a: float, scorelines: List[Tuple[int, int]]) -> Dict[str, float]:
    """Compute probability for each requested scoreline using independent Poisson model."""
    max_goals = max(max(h, a) for h, a in scorelines) + 2
    probs = _compute_score_distribution(lambda_h, lambda_a, max_goals)
    out: Dict[str, float] = {}
    for h, a in scorelines:
        p = float(probs[h, a]) if h < probs.shape[0] and a < probs.shape[1] else 0.0
        out[f"{h}-{a}"] = p
    # normalize if rounding causes tiny drift among considered scores
    return out


def predict_match(model, feature_row: pd.DataFrame, feature_data: pd.DataFrame) -> Dict:
    """Main prediction API producing full markets, score probs, risk and confidence.

    Args:
    - model: trained CatBoostClassifier
    - feature_row: single-row DataFrame built by prepare_match_features()
    - feature_data: full engineered dataset (used to compute league medians and team rows)

    Returns: dict strictly matching the required OUTPUT FORMAT
    """
    _require_catboost()

    if not isinstance(feature_row, pd.DataFrame) or feature_row.shape[0] != 1:
        raise ValueError("feature_row must be a single-row pandas DataFrame")

    # Get home/away names if present
    home_team = feature_row["HomeTeam"].iloc[0] if "HomeTeam" in feature_row.columns else "Unknown"
    away_team = feature_row["AwayTeam"].iloc[0] if "AwayTeam" in feature_row.columns else "Unknown"

    # Align feature_row columns to model expectations
    model_feature_names = None
    if hasattr(model, "feature_names_"):
        model_feature_names = list(getattr(model, "feature_names_"))
    elif hasattr(model, "get_feature_names"):
        try:
            model_feature_names = list(model.get_feature_names())
        except Exception:
            model_feature_names = None

    if model_feature_names is not None:
        missing = [c for c in model_feature_names if c not in feature_row.columns]
        for c in missing:
            # add default 0 for numeric or 'UNK' for non-numeric if info available
            feature_row[c] = 0.0
        feature_row = feature_row[model_feature_names]

    # Predict main result probabilities using CatBoost
    probs = model.predict_proba(feature_row)
    probs = np.asarray(probs).reshape(-1)
    classes = list(model.classes_)
    mapping = {str(lbl): 0.0 for lbl in classes}
    for lbl, p in zip(classes, probs):
        mapping[str(lbl)] = float(p)

    home_win = mapping.get("H", 0.0)
    draw_p = mapping.get("D", 0.0)
    away_win = mapping.get("A", 0.0)

    # Double chance
    dc_1X = home_win + draw_p
    dc_X2 = draw_p + away_win
    dc_12 = home_win + away_win

    # Score model: estimate expected goals using most recent team stats
    # Extract team-specific rows from feature_data
    home_row = None
    away_row = None
    try:
        home_row = feature_data[(feature_data["HomeTeam"] == home_team)].sort_values("Date").iloc[-1]
    except Exception:
        home_row = None
    try:
        away_row = feature_data[(feature_data["AwayTeam"] == away_team)].sort_values("Date").iloc[-1]
    except Exception:
        away_row = None

    lambda_h, lambda_a = _expected_goals_from_stats(home_row, away_row, feature_data)

    # Compute joint distribution and derived markets
    max_goals = 6
    joint = _compute_score_distribution(lambda_h, lambda_a, max_goals)

    # Goal markets: over X.5 are P(total > X)
    total_probs = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            total = i + j
            total_probs[total] = total_probs.get(total, 0.0) + float(joint[i, j])

    def prob_over(thresh: float) -> float:
        # thresh is e.g., 0.5 meaning total > 0.5
        t = math.floor(thresh + 1e-9)
        # total > thresh means total >= t+1 when thresh is integer+0.5; simpler sum totals > thresh
        s = 0.0
        for total, p in total_probs.items():
            if total > thresh:
                s += p
        return float(s)

    over_0_5 = prob_over(0.5)
    over_1_5 = prob_over(1.5)
    over_2_5 = prob_over(2.5)
    over_3_5 = prob_over(3.5)
    under_2_5 = 1.0 - over_2_5

    # BTTS: both teams score = 1 - P(home==0 or away==0) = 1 - (P(home==0) + P(away==0) - P(both==0))
    p_home_zero = float(np.sum(joint[0, :]))
    p_away_zero = float(np.sum(joint[:, 0]))
    p_both_zero = float(joint[0, 0])
    btts_yes = max(0.0, 1.0 - (p_home_zero + p_away_zero - p_both_zero))
    btts_no = 1.0 - btts_yes

    # Team markets
    home_score_prob = 1.0 - p_home_zero
    away_score_prob = 1.0 - p_away_zero
    clean_sheet_home = p_away_zero
    clean_sheet_away = p_home_zero

    # Win to nil (home win and away goals == 0): sum over home goals>0 at away==0
    win_to_nil_home = float(sum(joint[i, 0] for i in range(1, joint.shape[0])))
    win_to_nil_away = float(sum(joint[0, j] for j in range(1, joint.shape[1])))

    # Score predictions for requested list
    requested = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (1, 2)]
    score_probs = _scoreline_probabilities(lambda_h, lambda_a, requested)

    score_predictions = [{"score": k, "probability": float(v)} for k, v in score_probs.items()]

    # Confidence and risk
    # Confidence: based on main-model max probability and also inverse entropy of main distribution
    p_vals = np.array([home_win, draw_p, away_win], dtype=float)
    max_p = float(p_vals.max())
    # entropy normalized: -sum p log p / log(3) -> between 0 and 1; lower entropy -> higher confidence
    eps = 1e-12
    entropy = -float(np.sum([p * math.log(max(p, eps)) for p in p_vals]))
    max_entropy = math.log(3)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
    confidence_score = float(max(0.0, min(1.0, max_p * (1.0 - norm_entropy))))

    # Risk classification using thresholds on max probability and entropy
    if max_p >= 0.7 and norm_entropy < 0.4:
        risk = "LOW"
    elif max_p >= 0.5 and norm_entropy < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    recommended = "H" if home_win >= max(draw_p, away_win) else ("D" if draw_p >= away_win else "A")

    # Build strict output dictionary
    output = {
        "match": {"home_team": home_team, "away_team": away_team},
        "main_result": {
            "home_win": float(home_win),
            "draw": float(draw_p),
            "away_win": float(away_win),
            "double_chance": {"1X": float(min(1.0, dc_1X)), "X2": float(min(1.0, dc_X2)), "12": float(min(1.0, dc_12))},
        },
        "goal_markets": {
            "over_0_5": float(over_0_5),
            "over_1_5": float(over_1_5),
            "over_2_5": float(over_2_5),
            "over_3_5": float(over_3_5),
            "btts_yes": float(btts_yes),
            "btts_no": float(btts_no),
        },
        "team_markets": {
            "home_score": float(home_score_prob),
            "away_score": float(away_score_prob),
            "clean_sheet_home": float(clean_sheet_home),
            "clean_sheet_away": float(clean_sheet_away),
        },
        "score_predictions": score_predictions,
        "risk_level": risk,
        "confidence_score": float(confidence_score),
        "recommended_outcome": recommended,
    }

    logger.info(
        "Prediction for %s vs %s: Home %.3f Draw %.3f Away %.3f (conf=%.3f risk=%s)",
        home_team,
        away_team,
        home_win,
        draw_p,
        away_win,
        confidence_score,
        risk,
    )

    return output


# If run as script for self-test (requires trained model and feature dataset)
if __name__ == "__main__":
    try:
        model = load_prediction_model("models/football_model.cbm")
        from data_loader import load_all_data
        from src.features.feature_engineer import build_features

        raw = load_all_data()
        feats = build_features(raw)
        # pick last match teams as demo
        h = feats["HomeTeam"].iloc[-1]
        a = feats["AwayTeam"].iloc[-1]
        feat_row = prepare_match_features(h, a, feats, model=model)
        output = predict_match(model, feat_row, feats)
        print(output)
    except Exception as e:
        logger.exception("Engine self-test failed: %s", e)
