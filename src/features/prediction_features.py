"""
prediction_features.py

Single-match feature builder for live prediction.

Public API:
- build_features_for_match(home_team: str, away_team: str, historical_data: Union[str, pd.DataFrame], last_n: int = 10) -> Dict

Rules implemented:
- No model training or fitting is performed.
- Uses only historical_data provided (path to clean_dataset.csv or a DataFrame).
- Computes features for a single match using only past matches (sorted by Date) — deterministic.
- Handles missing teams safely by falling back to global medians/priors.

Required features provided:
- home_team_avg_goals_scored
- home_team_avg_goals_conceded
- away_team_avg_goals_scored
- away_team_avg_goals_conceded
- home_win_rate_last_n
- away_win_rate_last_n
- head_to_head_win_rate

Return: dict as specified by the integration tests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _load_historical(historical_data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load historical data from a CSV path or accept a DataFrame as-is. Ensures Date is parsed and rows are sorted chronologically."""
    if isinstance(historical_data, pd.DataFrame):
        df = historical_data.copy()
    else:
        p = Path(historical_data)
        if not p.exists():
            raise FileNotFoundError(f"historical_data path not found: {historical_data}")
        df = pd.read_csv(p, parse_dates=["Date"], dayfirst=True)
    # Ensure canonical column names exist
    # Trim whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns:
        raise ValueError("historical_data must contain a 'Date' column")
    # Parse dates if necessary
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Sort chronologically for deterministic "past matches" selection
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _team_history(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return rows where the team played (home or away)."""
    mask = (df.get("HomeTeam") == team) | (df.get("AwayTeam") == team)
    return df.loc[mask].copy()


def _compute_goals_stats(team_df: pd.DataFrame, team: str, last_n: int) -> (float, float):
    """Compute average goals scored and conceded for team using most recent last_n matches.

    Returns (avg_scored, avg_conceded)
    If not enough data, compute using available matches. If zero matches, return global fallbacks later.
    """
    if team_df.empty:
        return float("nan"), float("nan")
    # Consider most recent matches
    recent = team_df.tail(last_n)
    # Determine goals scored/conceded per row depending on home/away
    scored = []
    conceded = []
    for _, r in recent.iterrows():
        if r.get("HomeTeam") == team:
            # team was home
            gf = r.get("FTHG")
            ga = r.get("FTAG")
        else:
            gf = r.get("FTAG")
            ga = r.get("FTHG")
        try:
            gf = float(gf) if not pd.isna(gf) else np.nan
        except Exception:
            gf = np.nan
        try:
            ga = float(ga) if not pd.isna(ga) else np.nan
        except Exception:
            ga = np.nan
        scored.append(gf)
        conceded.append(ga)
    # compute means ignoring NaNs
    avg_scored = float(np.nanmean(scored)) if len(scored) > 0 else float("nan")
    avg_conceded = float(np.nanmean(conceded)) if len(conceded) > 0 else float("nan")
    return avg_scored, avg_conceded


def _compute_win_rate(team_df: pd.DataFrame, team: str, last_n: int) -> float:
    """Compute win rate (fraction wins) for the team's most recent last_n matches.

    For a match, win is determined from FTR: 'H' -> home win, 'A' -> away win, 'D' draw (not a win).
    """
    if team_df.empty:
        return float("nan")
    recent = team_df.tail(last_n)
    wins = 0
    total = 0
    for _, r in recent.iterrows():
        ftr = r.get("FTR")
        if pd.isna(ftr):
            continue
        total += 1
        if r.get("HomeTeam") == team:
            if str(ftr) == "H":
                wins += 1
        else:
            if str(ftr) == "A":
                wins += 1
    if total == 0:
        return float("nan")
    return float(wins) / float(total)


def _compute_head_to_head(df: pd.DataFrame, home: str, away: str) -> float:
    """Compute head-to-head win rate for home team vs away team using past matches only.

    Returns fraction of past head-to-head matches won by home team (counting wins by either side irrespective of venue).
    If no head-to-head data, returns NaN for caller to fill neutral prior.
    """
    mask = ((df.get("HomeTeam") == home) & (df.get("AwayTeam") == away)) | ((df.get("HomeTeam") == away) & (df.get("AwayTeam") == home))
    h2h = df.loc[mask].copy()
    if h2h.empty:
        return float("nan")
    wins = 0
    total = 0
    for _, r in h2h.iterrows():
        ftr = r.get("FTR")
        if pd.isna(ftr):
            continue
        total += 1
        # if result is home-team-win and home side equals requested home team
        if str(ftr) == "H":
            winner = r.get("HomeTeam")
        elif str(ftr) == "A":
            winner = r.get("AwayTeam")
        else:
            winner = None
        if winner is not None and winner == home:
            wins += 1
    if total == 0:
        return float("nan")
    return float(wins) / float(total)


def build_features_for_match(home_team: str, away_team: str, historical_data: Union[str, Path, pd.DataFrame], last_n: int = 10) -> Dict:
    """Build a single-row feature dict for a forthcoming match between home_team and away_team.

    Parameters
    - home_team, away_team: team names (strings)
    - historical_data: path to clean_dataset.csv or a pandas DataFrame containing historical matches
    - last_n: number of recent matches to consider for rate/avg calculations (default 10)

    Returns a dict with the required feature keys. Values are floats and deterministic. No leakage is introduced because
    only matches present in historical_data (assumed past) are used, and selections are based on chronologically earlier rows.
    """
    # Load and prepare
    df = _load_historical(historical_data)

    # Defensive normalization of team names (trim)
    home = str(home_team).strip()
    away = str(away_team).strip()

    # Global fallbacks (computed from entire dataset) to handle missing teams
    # Use per-team averages aggregated across dataset when available
    global_avg_scored = None
    global_avg_conceded = None
    try:
        # For global averages, compute per-team means and then median to be robust
        # Build per-team totals
        teams = set(df.get("HomeTeam").dropna().unique().tolist() + df.get("AwayTeam").dropna().unique().tolist())
        per_team_scored = []
        per_team_conceded = []
        for t in teams:
            tdf = _team_history(df, t)
            if tdf.empty:
                continue
            s, c = _compute_goals_stats(tdf, t, last_n=99999)  # compute across all matches
            if not np.isnan(s):
                per_team_scored.append(s)
            if not np.isnan(c):
                per_team_conceded.append(c)
        if per_team_scored:
            global_avg_scored = float(np.nanmedian(per_team_scored))
        else:
            global_avg_scored = 1.3
        if per_team_conceded:
            global_avg_conceded = float(np.nanmedian(per_team_conceded))
        else:
            global_avg_conceded = 1.3
    except Exception:
        global_avg_scored = 1.3
        global_avg_conceded = 1.3

    # Home team stats
    h_hist = _team_history(df, home)
    h_avg_scored, h_avg_conceded = _compute_goals_stats(h_hist, home, last_n)
    if np.isnan(h_avg_scored):
        h_avg_scored = global_avg_scored
    if np.isnan(h_avg_conceded):
        h_avg_conceded = global_avg_conceded

    # Away team stats
    a_hist = _team_history(df, away)
    a_avg_scored, a_avg_conceded = _compute_goals_stats(a_hist, away, last_n)
    if np.isnan(a_avg_scored):
        a_avg_scored = global_avg_scored
    if np.isnan(a_avg_conceded):
        a_avg_conceded = global_avg_conceded

    # Win rates
    h_win_rate = _compute_win_rate(h_hist, home, last_n)
    a_win_rate = _compute_win_rate(a_hist, away, last_n)
    # Fallback to neutral prior (approximate) if NaN
    if np.isnan(h_win_rate):
        h_win_rate = 0.33
    if np.isnan(a_win_rate):
        a_win_rate = 0.33

    # Head-to-head
    h2h_rate = _compute_head_to_head(df, home, away)
    if np.isnan(h2h_rate):
        # If no direct history, fallback to weighted average of team win rates (as a weak prior)
        h2h_rate = float((h_win_rate + (1.0 - a_win_rate)) / 2.0)

    # Clamp / sanitize outputs
    def _safe_float(x: float) -> float:
        try:
            if x is None:
                return 0.0
            if np.isnan(x):
                return 0.0
            return float(x)
        except Exception:
            return 0.0

    out = {
        "HomeTeam": home,
        "AwayTeam": away,
        "home_team_avg_goals_scored": _safe_float(h_avg_scored),
        "home_team_avg_goals_conceded": _safe_float(h_avg_conceded),
        "away_team_avg_goals_scored": _safe_float(a_avg_scored),
        "away_team_avg_goals_conceded": _safe_float(a_avg_conceded),
        "home_win_rate_last_n": _safe_float(h_win_rate),
        "away_win_rate_last_n": _safe_float(a_win_rate),
        "head_to_head_win_rate": _safe_float(h2h_rate),
    }

    return out
