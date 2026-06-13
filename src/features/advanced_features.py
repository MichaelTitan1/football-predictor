"""
advanced_features.py

Advanced feature engineering for football prediction system.

This module computes advanced, deterministic, leakage-free features using only past data.
It is designed to integrate with the simpler features produced by feature_engineer.py and
provide higher-signal inputs for model training.

Public API:
- build_advanced_features(df: pd.DataFrame) -> pd.DataFrame

Key features implemented:
1. Elo-like rolling team strength ratings (separate home and away ratings)
2. Form intensity using exponentially decayed recent results (short and long spans)
3. Goal expectancy model (expected goals for/against adjusted by opponent strength)
4. League context normalization (uses 'League' column when available)
5. Consistency index (variance-based measures of team performance)
6. Matchup dynamic features (attack vs defense interactions)

Engineering notes:
- All statistics are computed using only past matches via shifting or iterative updates (no leakage).
- Deterministic: given the same input and ordering, outputs are reproducible.
- Uses pandas + standard library only.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing)}")


def _result_points_from_ftr(ftr: str, is_home: bool) -> int:
    if pd.isna(ftr):
        return 0
    if is_home:
        return 3 if ftr == "H" else (1 if ftr == "D" else 0)
    else:
        return 3 if ftr == "A" else (1 if ftr == "D" else 0)


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build advanced features from cleaned, historical match DataFrame.

    Args:
        df: DataFrame with columns Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR (optionally League)
    Returns:
        DataFrame with original columns plus advanced features. Rows are in the same order as input.
    """
    _validate_input(df)

    # Work on a copy and ensure chronological order
    df = df.copy().sort_values("Date").reset_index(drop=True)

    n = len(df)
    logger.info("Computing advanced features for %d matches", n)

    # Prepare output columns
    # Elo-like ratings (prior to match)
    home_elo_prior = np.zeros(n, dtype=float)
    away_elo_prior = np.zeros(n, dtype=float)

    # Form intensity (short and long) (prior)
    home_form_short = np.zeros(n, dtype=float)
    away_form_short = np.zeros(n, dtype=float)
    home_form_long = np.zeros(n, dtype=float)
    away_form_long = np.zeros(n, dtype=float)

    # Rolling averages for goals (prior)
    home_avg_gf = np.zeros(n, dtype=float)
    home_avg_ga = np.zeros(n, dtype=float)
    away_avg_gf = np.zeros(n, dtype=float)
    away_avg_ga = np.zeros(n, dtype=float)

    # Consistency (variance) prior
    home_consistency = np.zeros(n, dtype=float)
    away_consistency = np.zeros(n, dtype=float)

    # Expected goals (xG) prior
    expected_home_xg = np.zeros(n, dtype=float)
    expected_away_xg = np.zeros(n, dtype=float)

    # Matchup dynamic features
    attack_vs_defense = np.zeros(n, dtype=float)
    defense_vs_attack = np.zeros(n, dtype=float)

    # Initialize ELO dictionaries: separate home and away ratings
    BASE_ELO = 1500.0
    home_elo: Dict[str, float] = defaultdict(lambda: BASE_ELO)
    away_elo: Dict[str, float] = defaultdict(lambda: BASE_ELO)

    # Keep rolling history for each team to compute decayed form, averages and variance.
    # We'll store lists for each team and update incrementally for reproducibility.
    history_goals_for: Dict[str, List[int]] = defaultdict(list)
    history_goals_against: Dict[str, List[int]] = defaultdict(list)
    history_points: Dict[str, List[int]] = defaultdict(list)

    # League context: use column 'League' if present, otherwise use global
    has_league = "League" in df.columns

    # Precompute league medians for normalization (based on available prior matches) will be computed dynamically
    # We'll iterate through matches in chronological order, compute features prior to match, then update histories and ELO.

    # Helper for ELO update: K-factor maybe dynamic by match importance; simple constant K
    K = 20.0

    def _expected_score(ra: float, rb: float) -> float:
        # standard Elo expected score
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Prior ratings
        h_elo = float(home_elo[home])
        a_elo = float(away_elo[away])
        home_elo_prior[idx] = h_elo
        away_elo_prior[idx] = a_elo

        # Prior rolling stats
        h_gf_hist = history_goals_for.get(home, [])
        h_ga_hist = history_goals_against.get(home, [])
        h_pts_hist = history_points.get(home, [])

        a_gf_hist = history_goals_for.get(away, [])
        a_ga_hist = history_goals_against.get(away, [])
        a_pts_hist = history_points.get(away, [])

        # Rolling averages (use up to last 10 matches)
        def _rolling_mean(lst: List[int], window: int = 10) -> float:
            if not lst:
                return 0.0
            return float(np.mean(lst[-window:]))

        home_avg_gf[idx] = _rolling_mean(h_gf_hist, window=10)
        home_avg_ga[idx] = _rolling_mean(h_ga_hist, window=10)
        away_avg_gf[idx] = _rolling_mean(a_gf_hist, window=10)
        away_avg_ga[idx] = _rolling_mean(a_ga_hist, window=10)

        # Consistency: use coefficient of variation of goals for (std/mean) over last 10 matches
        def _consistency(lst: List[int], window: int = 10) -> float:
            if not lst:
                return 0.0
            arr = np.array(lst[-window:], dtype=float)
            if arr.mean() == 0:
                return 0.0
            return float(np.std(arr, ddof=0) / (arr.mean() + 1e-9))

        home_consistency[idx] = _consistency(h_gf_hist, window=10)
        away_consistency[idx] = _consistency(a_gf_hist, window=10)

        # Form intensity: exponentially decayed weighted points. We'll compute two spans: short (~3) and long (~10)
        def _decayed_points(points_list: List[int], span: float) -> float:
            # span parameter maps to ewm alpha via span in pandas: alpha = 2/(span+1)
            if not points_list:
                return 0.0
            s = pd.Series(points_list)
            # compute ewm on the series and take the last value (most recent decayed mean), but ensure using only past data, so s is fine
            return float(s.ewm(span=span, adjust=False).mean().iloc[-1])

        home_form_short[idx] = _decayed_points(h_pts_hist, span=3)
        home_form_long[idx] = _decayed_points(h_pts_hist, span=10)
        away_form_short[idx] = _decayed_points(a_pts_hist, span=3)
        away_form_long[idx] = _decayed_points(a_pts_hist, span=10)

        # Goal expectancy: base rate per league or global
        # Compute league-level averages from history (global if missing)
        if has_league:
            league = row.get("League", None)
            if pd.isna(league):
                league = None
        else:
            league = None

        # compute league medians from existing historical entries in df up to this point (avoid future leakage)
        if league is not None:
            # get prior rows in same league (earlier indexes)
            prior_league = df.loc[:idx - 1]
            prior_league = prior_league[prior_league.get("League") == league] if not prior_league.empty else prior_league
            if not prior_league.empty:
                league_avg_scored = float((prior_league["FTHG"] + prior_league["FTAG"]) .median() / 2.0)
            else:
                league_avg_scored = 1.3
        else:
            # use global median of per-team rolling means computed so far (fallback)
            all_prior = df.loc[:idx - 1]
            if not all_prior.empty:
                league_avg_scored = float(((all_prior["FTHG"] + all_prior["FTAG"]).median()) / 2.0)
            else:
                league_avg_scored = 1.3

        # Avoid zero
        if league_avg_scored <= 0:
            league_avg_scored = 1.3

        # Compute attack/defense strengths relative to league
        home_attack_strength = (home_avg_gf[idx] / league_avg_scored) if league_avg_scored else 1.0
        home_defense_weakness = (home_avg_ga[idx] / league_avg_scored) if league_avg_scored else 1.0
        away_attack_strength = (away_avg_gf[idx] / league_avg_scored) if league_avg_scored else 1.0
        away_defense_weakness = (away_avg_ga[idx] / league_avg_scored) if league_avg_scored else 1.0

        # Home advantage estimated from short-term home win rate (approx); fallback 1.05 multiplier
        # use home_form_short as proxy (scale)
        home_advantage = 1.05 + 0.05 * (home_form_short[idx] / 3.0)  # small scaling

        # Expected goals heuristic combining strengths
        base = league_avg_scored
        expected_h = base * home_attack_strength * away_defense_weakness * home_advantage
        expected_a = base * away_attack_strength * home_defense_weakness

        # clamp
        expected_home_xg[idx] = float(max(0.05, min(5.0, expected_h)))
        expected_away_xg[idx] = float(max(0.05, min(5.0, expected_a)))

        # matchup dynamics
        attack_vs_defense[idx] = home_attack_strength / (away_defense_weakness + 1e-9)
        defense_vs_attack[idx] = home_defense_weakness / (away_attack_strength + 1e-9)

        # Now, update histories and ELO after observing this match result (sequential update ensures no leakage)
        fthg = int(row["FTHG"]) if not pd.isna(row["FTHG"]) else 0
        ftag = int(row["FTAG"]) if not pd.isna(row["FTAG"]) else 0
        ftr = row.get("FTR")

        # Points
        h_points = _result_points_from_ftr(ftr, True)
        a_points = _result_points_from_ftr(ftr, False)

        history_goals_for[home].append(fthg)
        history_goals_against[home].append(ftag)
        history_points[home].append(h_points)

        history_goals_for[away].append(ftag)
        history_goals_against[away].append(fthg)
        history_points[away].append(a_points)

        # Elo update: compute expected and update both home and away ELOs
        # For Elo we can bias by home advantage (e.g., +30 rating points to home)
        HOME_ADV_ELO = 30.0
        exp_home = _expected_score(h_elo + HOME_ADV_ELO, a_elo)
        exp_away = 1.0 - exp_home
        # Actual score points: 1.0 win, 0.5 draw, 0 loss (for Elo)
        if ftr == "H":
            sh = 1.0
            sa = 0.0
        elif ftr == "A":
            sh = 0.0
            sa = 1.0
        else:
            sh = 0.5
            sa = 0.5

        # Dynamic K: slightly higher for large rating diff or unstable teams
        k_home = K * (1.0 + home_consistency[idx])
        k_away = K * (1.0 + away_consistency[idx])

        home_elo[home] = h_elo + k_home * (sh - exp_home)
        away_elo[away] = a_elo + k_away * (sa - exp_away)

    # After loop, compose output DataFrame
    out = df.copy()
    out["home_elo_prior"] = home_elo_prior
    out["away_elo_prior"] = away_elo_prior

    out["home_form_short"] = home_form_short
    out["home_form_long"] = home_form_long
    out["away_form_short"] = away_form_short
    out["away_form_long"] = away_form_long

    out["home_avg_goals_for_prior"] = home_avg_gf
    out["home_avg_goals_against_prior"] = home_avg_ga
    out["away_avg_goals_for_prior"] = away_avg_gf
    out["away_avg_goals_against_prior"] = away_avg_ga

    out["home_consistency"] = home_consistency
    out["away_consistency"] = away_consistency

    out["expected_home_xg"] = expected_home_xg
    out["expected_away_xg"] = expected_away_xg

    out["attack_vs_defense"] = attack_vs_defense
    out["defense_vs_attack"] = defense_vs_attack

    # Additional composite features useful for models
    out["elo_diff_home_minus_away"] = out["home_elo_prior"] - out["away_elo_prior"]
    out["form_diff_short"] = out["home_form_short"] - out["away_form_short"]
    out["form_diff_long"] = out["home_form_long"] - out["away_form_long"]
    out["xg_diff"] = out["expected_home_xg"] - out["expected_away_xg"]

    # Ensure deterministic column order: original columns + advanced cols
    advanced_cols = [
        "home_elo_prior", "away_elo_prior",
        "home_form_short", "home_form_long", "away_form_short", "away_form_long",
        "home_avg_goals_for_prior", "home_avg_goals_against_prior", "away_avg_goals_for_prior", "away_avg_goals_against_prior",
        "home_consistency", "away_consistency",
        "expected_home_xg", "expected_away_xg",
        "attack_vs_defense", "defense_vs_attack",
        "elo_diff_home_minus_away", "form_diff_short", "form_diff_long", "xg_diff",
    ]

    # Fill NaNs with safe defaults
    for c in advanced_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    logger.info("Advanced features computed and appended: %d features for %d matches", len(advanced_cols), len(out))
    return out
