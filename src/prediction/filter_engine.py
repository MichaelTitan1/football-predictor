"""
filter_engine.py

Filters full prediction outputs and returns only high-confidence betting opportunities.

Public API
- filter_predictions(prediction_dict: dict, *, high_threshold: float = 0.70,
                     medium_threshold: float = 0.55, include_medium: bool = False,
                     max_picks: int = 10) -> dict

Behavior
- Does not mutate the input prediction_dict
- Extracts candidate markets from the prediction dict (main_result, double_chance,
  goal_markets, team_markets, score_predictions, and any numeric probabilities found)
- Applies thresholds:
    HIGH: probability >= high_threshold
    MEDIUM: medium_threshold <= probability < high_threshold
    LOW: probability < medium_threshold (discard)
- By default only returns HIGH picks (include_medium=True will include MEDIUM as well)
- Sorts picks by probability descending and returns at most max_picks
- Returns a dict: {"best_picks": [ {"market": str, "probability": float, "confidence": "HIGH"}, ... ]}

This module is lightweight, production-ready, and suitable for reuse in API or dashboard code.
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


__all__ = ["filter_predictions"]


def _safe_float(val) -> float:
    """Safely coerce a value to float in range [0,1]. Returns -1 for invalid values."""
    try:
        f = float(val)
    except Exception:
        return -1.0
    # clamp
    if f != f:  # NaN
        return -1.0
    if f < 0:
        return -1.0
    # assume probabilities are in 0..1; if >1 and <=100 maybe percent -> convert
    if f > 1.0 and f <= 100.0:
        f = f / 100.0
    if f > 1.0:
        # unrealistic probability
        return -1.0
    return f


def _extract_markets(prediction: Dict) -> List[Tuple[str, float]]:
    """Extract candidate (market_name, probability) pairs from the prediction dict.

    This function is resilient: it handles the common structured keys used by the engine
    (main_result, double_chance, goal_markets, team_markets, score_predictions) and
    will also scan other top-level items for numeric probabilities.
    """
    markets: List[Tuple[str, float]] = []

    if not isinstance(prediction, dict):
        logger.warning("Expected prediction to be a dict, got %s", type(prediction))
        return markets

    p = prediction

    # 1. main_result
    main = p.get("main_result")
    if isinstance(main, dict):
        mapping = {
            "home_win": "Main: Home Win",
            "draw": "Main: Draw",
            "away_win": "Main: Away Win",
        }
        for key, label in mapping.items():
            if key in main:
                prob = _safe_float(main.get(key))
                if prob >= 0:
                    markets.append((label, prob))
        # double chance
        dc = main.get("double_chance")
        if isinstance(dc, dict):
            for k, v in dc.items():
                label = f"Double Chance: {k}"
                prob = _safe_float(v)
                if prob >= 0:
                    markets.append((label, prob))

    # 2. goal_markets
    gm = p.get("goal_markets")
    if isinstance(gm, dict):
        # friendly labels
        label_map = {
            "over_0_5": "Goals: Over 0.5",
            "over_1_5": "Goals: Over 1.5",
            "over_2_5": "Goals: Over 2.5",
            "over_3_5": "Goals: Over 3.5",
            "btts_yes": "BTTS: Yes",
            "btts_no": "BTTS: No",
            "under_2_5": "Goals: Under 2.5",
        }
        for k, v in gm.items():
            lbl = label_map.get(k, f"Goal: {k}")
            prob = _safe_float(v)
            if prob >= 0:
                markets.append((lbl, prob))

    # 3. team_markets
    tm = p.get("team_markets")
    if isinstance(tm, dict):
        mapping = {
            "home_score": "Team: Home to Score",
            "away_score": "Team: Away to Score",
            "clean_sheet_home": "Team: Clean Sheet Home",
            "clean_sheet_away": "Team: Clean Sheet Away",
            "win_to_nil_home": "Team: Win to Nil Home",
            "win_to_nil_away": "Team: Win to Nil Away",
        }
        for k, v in tm.items():
            lbl = mapping.get(k, f"Team: {k}")
            prob = _safe_float(v)
            if prob >= 0:
                markets.append((lbl, prob))

    # 4. score_predictions
    sp = p.get("score_predictions")
    if isinstance(sp, list):
        for item in sp:
            if isinstance(item, dict):
                sc = item.get("score") or item.get("label")
                prob = _safe_float(item.get("probability") or item.get("prob"))
                if sc is not None and prob >= 0:
                    markets.append((f"Score: {sc}", prob))

    # 5. scan other top-level numeric entries that look like markets
    known_keys = {"main_result", "goal_markets", "team_markets", "score_predictions"}
    for k, v in p.items():
        if k in known_keys:
            continue
        # if value is numeric probability, include
        prob = _safe_float(v)
        if prob >= 0:
            markets.append((f"Other: {k}", prob))
        # if dict with numeric leaves, include them (shallow scan)
        if isinstance(v, dict):
            for subk, subv in v.items():
                prob2 = _safe_float(subv)
                if prob2 >= 0:
                    markets.append((f"{k}: {subk}", prob2))

    return markets


def filter_predictions(
    prediction_dict: Dict,
    *,
    high_threshold: float = 0.70,
    medium_threshold: float = 0.55,
    include_medium: bool = False,
    max_picks: int = 10,
) -> Dict:
    """Filter the full model output and return only the best picks.

    Rules:
    - HIGH: probability >= high_threshold
    - MEDIUM: medium_threshold <= probability < high_threshold (not shown by default)
    - LOW: probability < medium_threshold (discard)

    Args:
      prediction_dict: the full prediction output produced by the intelligence engine
      high_threshold: float threshold for HIGH confidence
      medium_threshold: float threshold lower bound for MEDIUM
      include_medium: if True, include MEDIUM picks in the output
      max_picks: maximum number of picks to return (sorted by probability desc)

    Returns:
      dict: {"best_picks": [ {"market": str, "probability": float, "confidence": "HIGH"}, ... ]}
    """
    orig = prediction_dict
    if not isinstance(orig, dict):
        raise ValueError("prediction_dict must be a dict")

    # Do not modify input
    # Extract all candidate markets
    candidates = _extract_markets(copy.deepcopy(orig))

    logger.debug("Extracted %d candidate markets", len(candidates))

    picks: List[Dict] = []
    for market, prob in candidates:
        if prob >= high_threshold:
            picks.append({"market": market, "probability": float(prob), "confidence": "HIGH"})
        elif include_medium and prob >= medium_threshold:
            picks.append({"market": market, "probability": float(prob), "confidence": "MEDIUM"})
        else:
            # discard low-confidence picks
            continue

    # Sort picks by probability descending
    picks = sorted(picks, key=lambda x: x["probability"], reverse=True)

    # Limit to max_picks
    if max_picks is not None and isinstance(max_picks, int) and max_picks > 0:
        picks = picks[:max_picks]

    logger.info("Filtered picks count: %d (high_threshold=%.2f include_medium=%s)", len(picks), high_threshold, include_medium)

    return {"best_picks": picks}
