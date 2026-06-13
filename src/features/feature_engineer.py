"""
feature_engineer.py

Transforms cleaned match-level data (from data_loader.load_all_data) into ML-ready features
for a football match prediction pipeline.

Core API:
- build_features(df: pd.DataFrame, *, last_n_form=5, rolling_window_goals=10, rolling_window_points=5) -> pd.DataFrame

Engineering notes:
- Only uses pandas + standard library
- All rolling features use past matches only (no leakage) via .shift(1)
- Works on large datasets by grouping by team and operating per-group in vectorized or linear-time loops
- Handles missing data gracefully (fills with sensible defaults)

Expected input columns (after data_loader):
- Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR

Output:
- Original match rows with appended features for home/away teams, head-to-head, and home-advantage metrics.

"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional
from pathlib import Path

import pandas as pd

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _points_from_result(result: pd.Series, is_home: bool) -> pd.Series:
    """Convert FTR result to points for team where is_home indicates perspective."""
    # result: 'H', 'A', 'D'
    # For home team: H -> 3, D -> 1, A -> 0
    # For away team: A -> 3, D -> 1, H -> 0
    r = result.fillna("")
    if is_home:
        return r.map({"H": 3, "D": 1, "A": 0}).astype("Int64")
    else:
        return r.map({"A": 3, "D": 1, "H": 0}).astype("Int64")


def _wl_draw_from_result(result: pd.Series, is_home: bool) -> pd.DataFrame:
    """Return DataFrame with columns win, draw, loss (0/1) from perspective of team."""
    r = result.fillna("")
    if is_home:
        win = (r == "H").astype(int)
        draw = (r == "D").astype(int)
        loss = (r == "A").astype(int)
    else:
        win = (r == "A").astype(int)
        draw = (r == "D").astype(int)
        loss = (r == "H").astype(int)
    return pd.DataFrame({"win": win, "draw": draw, "loss": loss})


def _compute_streaks(flag_series: pd.Series) -> List[int]:
    """Compute previous-match streak lengths for a boolean/int flag series (1 means event occurred).

    For position i the returned value is the consecutive count of 1s immediately preceding position i (does not include i itself).
    Example: input [1,1,0,1] -> output [0,1,0,0]
    """
    streaks: List[int] = []
    current = 0
    for val in flag_series:
        streaks.append(current)
        if int(val):
            current += 1
        else:
            current = 0
    return streaks


def build_features(
    df: pd.DataFrame,
    *,
    last_n_form: int = 5,
    rolling_window_goals: int = 10,
    rolling_window_points: int = 5,
) -> pd.DataFrame:
    """
    Build ML-ready features from cleaned match-level data.

    Args:
        df: cleaned DataFrame from data_loader (must include Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR)
        last_n_form: window size to compute last-N match form
        rolling_window_goals: window size to compute avg goals scored/conceded
        rolling_window_points: window size for rolling points per team

    Returns:
        pandas.DataFrame: original matches with appended features. Rows remain in the same chronological order.
    """
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {sorted(missing)}")

    # Work on a copy and sort by date to ensure chronological order
    df_work = df.copy().reset_index(drop=True)
    df_work = df_work.sort_values("Date").reset_index(drop=True)
    logger.info("Starting feature engineering on %d matches", len(df_work))

    # Assign match id for stable joins
    df_work["_match_id"] = df_work.index.astype(int)

    # Build a long-format table with one row per team-match (two rows per match)
    # Columns: _match_id, Date, Team, Opponent, IsHome, GoalsFor, GoalsAgainst, FTR, Points
    home = pd.DataFrame(
        {
            "_match_id": df_work["_match_id"],
            "Date": df_work["Date"],
            "Team": df_work["HomeTeam"],
            "Opponent": df_work["AwayTeam"],
            "IsHome": True,
            "GoalsFor": df_work["FTHG"].astype(int),
            "GoalsAgainst": df_work["FTAG"].astype(int),
            "FTR": df_work["FTR"],
        }
    )

    away = pd.DataFrame(
        {
            "_match_id": df_work["_match_id"],
            "Date": df_work["Date"],
            "Team": df_work["AwayTeam"],
            "Opponent": df_work["HomeTeam"],
            "IsHome": False,
            "GoalsFor": df_work["FTAG"].astype(int),
            "GoalsAgainst": df_work["FTHG"].astype(int),
            "FTR": df_work["FTR"],
        }
    )

    long = pd.concat([home, away], ignore_index=True)
    # Keep original chronological order per team by sorting by Date then _match_id
    long = long.sort_values(["Team", "Date", "_match_id"]).reset_index(drop=True)

    # Points and W/D/L flags from perspective of the team
    long["Points"] = _points_from_result(long["FTR"], long["IsHome"])  # Int64
    wdl = _wl_draw_from_result(long["FTR"], long["IsHome"])  # DataFrame with win/draw/loss
    long = pd.concat([long, wdl], axis=1)

    # Prepare containers for computed features per long-row
    # We will compute groupwise per team using groupby.apply-like iteration
    feature_cols = {
        # form & rolling
        f"rolling_points_last_{rolling_window_points}": [],
        f"last_{last_n_form}_matches_points": [],
        # attack/defense
        f"avg_goals_for_last_{rolling_window_goals}": [],
        f"avg_goals_against_last_{rolling_window_goals}": [],
        # momentum
        "win_streak": [],
        "loss_streak": [],
        f"draw_rate_last_{last_n_form}": [],
        # home advantage (for home rows only we'll compute home_win_rate)
        "historical_home_win_rate": [],
    }

    # Initialize with default values
    for k in feature_cols.keys():
        long[k] = pd.NA

    # Group by team and compute features
    grouped = long.groupby("Team", sort=False, group_keys=False)

    def _process_team(group: pd.DataFrame) -> pd.DataFrame:
        # Ensure group is sorted chronologically (should already be)
        group = group.sort_values(["Date", "_match_id"]).reset_index(drop=True)

        # Points as int
        points = group["Points"].astype("Int64").fillna(0).astype(int)
        gf = group["GoalsFor"].astype(int)
        ga = group["GoalsAgainst"].astype(int)
        is_home = group["IsHome"].astype(bool)
        win = group["win"].astype(int)
        draw = group["draw"].astype(int)
        loss = group["loss"].astype(int)

        n = len(group)

        # Rolling points (exclude current match using shift)
        if n > 0:
            pts_shift = pd.Series(points).shift(1).fillna(0)
            rolling_pts = pts_shift.rolling(window=rolling_window_points, min_periods=1).sum().astype(float)
        else:
            rolling_pts = pd.Series([], dtype=float)

        # Last-N matches points (exact last_n_form sum)
        last_n_pts = pts_shift.rolling(window=last_n_form, min_periods=1).sum().astype(float)

        # Rolling averages for goals for/against
        gf_shift = gf.shift(1).fillna(0)
        ga_shift = ga.shift(1).fillna(0)
        avg_gf = gf_shift.rolling(window=rolling_window_goals, min_periods=1).mean().astype(float)
        avg_ga = ga_shift.rolling(window=rolling_window_goals, min_periods=1).mean().astype(float)

        # win/loss streaks (previous-match streaks)
        win_streaks = _compute_streaks(win)
        loss_streaks = _compute_streaks(loss)

        # draw rate over last_n_form
        draw_rate = draw.shift(1).rolling(window=last_n_form, min_periods=1).mean().astype(float)

        # historical home win rate: compute based on past home matches only
        home_mask = is_home
        home_wins = (is_home & (win == 1)).astype(int)
        # cumulative sums for home matches
        home_matches_count = home_mask.cumsum()
        home_wins_cum = home_wins.cumsum()
        # historical home win rate prior to current match: compute (home_wins_cum - curr_home_win)/ (home_matches_count - curr_is_home)
        # Using shift to get prior counts
        home_matches_prior = home_mask.cumsum().shift(1).fillna(0).astype(int)
        home_wins_prior = home_wins.cumsum().shift(1).fillna(0).astype(int)
        # avoid division by zero
        with pd.option_context("mode.use_inf_as_na", True):
            historical_home_win_rate = (
                home_wins_prior / home_matches_prior.replace({0: pd.NA})
            ).astype(float)
        historical_home_win_rate = historical_home_win_rate.fillna(0.0)

        # Assign results back into group
        group[f"rolling_points_last_{rolling_window_points}"] = rolling_pts.values
        group[f"last_{last_n_form}_matches_points"] = last_n_pts.values
        group[f"avg_goals_for_last_{rolling_window_goals}"] = avg_gf.values
        group[f"avg_goals_against_last_{rolling_window_goals}"] = avg_ga.values
        group["win_streak"] = win_streaks
        group["loss_streak"] = loss_streaks
        group[f"draw_rate_last_{last_n_form}"] = draw_rate.values
        group["historical_home_win_rate"] = historical_home_win_rate.values

        return group

    long = grouped.apply(_process_team).reset_index(drop=True)

    # Now extract features for home and away teams for each match and merge back to match-level
    # Split long back into home and away perspectives by IsHome
    home_feats = long[long["IsHome"]].set_index("_match_id")[
        [
            f"rolling_points_last_{rolling_window_points}",
            f"last_{last_n_form}_matches_points",
            f"avg_goals_for_last_{rolling_window_goals}",
            f"avg_goals_against_last_{rolling_window_goals}",
            "win_streak",
            "loss_streak",
            f"draw_rate_last_{last_n_form}",
            "historical_home_win_rate",
        ]
    ].rename(columns=lambda c: f"home_{c}")

    away_feats = long[~long["IsHome"]].set_index("_match_id")[
        [
            f"rolling_points_last_{rolling_window_points}",
            f"last_{last_n_form}_matches_points",
            f"avg_goals_for_last_{rolling_window_goals}",
            f"avg_goals_against_last_{rolling_window_goals}",
            "win_streak",
            "loss_streak",
            f"draw_rate_last_{last_n_form}",
        ]
    ].rename(columns=lambda c: f"away_{c}")

    # Merge features into df_work by match id
    features = df_work[["_match_id"]].set_index("_match_id").join(home_feats).join(away_feats)

    # Head-to-head features: compute previous head-to-head counts and averages
    # We'll define pair_key as alphabetical tuple so that past matches in either venue are included
    def _pair_key(row):
        a = row["HomeTeam"]
        b = row["AwayTeam"]
        return "__".join(sorted([a, b]))

    df_work["_pair_key"] = df_work.apply(_pair_key, axis=1)

    # Group by pair and compute cumulative metrics shifted by 1
    pair_group = df_work.groupby("_pair_key", sort=False)

    # Metrics to compute: h2h_matches_prior, h2h_home_wins_prior (home team wins in previous encounters),
    # h2h_away_wins_prior, h2h_draws_prior, h2h_avg_total_goals_prior
    h2h_cols = [
        "h2h_matches_prior",
        "h2h_home_wins_prior",
        "h2h_away_wins_prior",
        "h2h_draws_prior",
        "h2h_avg_total_goals_prior",
    ]
    for c in h2h_cols:
        df_work[c] = 0

    def _process_pair(group: pd.DataFrame) -> pd.DataFrame:
        # group in chronological order
        group = group.sort_values(["Date", "_match_id"]).reset_index(drop=True)
        # cumulative counts
        total = []
        home_wins = []
        away_wins = []
        draws = []
        avg_goals = []

        matches_so_far = 0
        home_wins_so_far = 0
        away_wins_so_far = 0
        draws_so_far = 0
        goals_sum_so_far = 0

        for _, r in group.iterrows():
            # prior values
            if matches_so_far == 0:
                total.append(0)
                home_wins.append(0)
                away_wins.append(0)
                draws.append(0)
                avg_goals.append(0.0)
            else:
                total.append(matches_so_far)
                home_wins.append(home_wins_so_far)
                away_wins.append(away_wins_so_far)
                draws.append(draws_so_far)
                avg_goals.append(goals_sum_so_far / matches_so_far)

            # update counts including current match
            matches_so_far += 1
            # Determine if home or away team of this row won (based on FTR)
            res = r.get("FTR")
            if res == "H":
                home_wins_so_far += 1
            elif res == "A":
                away_wins_so_far += 1
            else:
                draws_so_far += 1

            goals_sum_so_far += (int(r.get("FTHG", 0)) + int(r.get("FTAG", 0)))

        group = group.copy()
        group["h2h_matches_prior"] = total
        group["h2h_home_wins_prior"] = home_wins
        group["h2h_away_wins_prior"] = away_wins
        group["h2h_draws_prior"] = draws
        group["h2h_avg_total_goals_prior"] = avg_goals
        return group

    df_work = pair_group.apply(_process_pair).reset_index(drop=True)

    # Merge head-to-head features into features DataFrame (which is indexed by _match_id)
    hh = df_work.set_index("_match_id")[
        [
            "h2h_matches_prior",
            "h2h_home_wins_prior",
            "h2h_away_wins_prior",
            "h2h_draws_prior",
            "h2h_avg_total_goals_prior",
        ]
    ]
    features = features.join(hh)

    # HOME ADVANTAGE FEATURE: encode home advantage impact based on historical home win rate
    # We already computed historical_home_win_rate for home teams in features.home_historical_home_win_rate
    # Create a single numeric feature: home_advantage_score = home_historical_home_win_rate - away_recent_win_rate
    # Compute away recent win rate from away team perspective using last_n_form window
    # But we have away_draw_rate; compute away recent win rate (last_n_form) from long and pivot
    # For simplicity compute away_recent_win_rate from long like we did for home but extract as away_win_rate_last_N

    # Extract away recent win rate: compute from long grouped data a draw_rate was computed; need win rate
    # We'll recompute a short per-team last_n_form win rate pivot for away side
    # For performance reuse columns in long
    long_win_rate = long.groupby(["Team"]) ["win"].apply(
        lambda s: s.shift(1).rolling(window=last_n_form, min_periods=1).mean()
    )
    # long_win_rate is aligned to long index
    long = long.copy()
    long["win_rate_last_n"] = long_win_rate.reset_index(level=0, drop=True)

    away_win_rate = long[~long["IsHome"]].set_index("_match_id")["win_rate_last_n"].rename(
        "away_win_rate_last_n"
    )
    features = features.join(away_win_rate)

    # compute home_advantage_score (difference between home historical home win rate and away recent win rate)
    features["home_advantage_score"] = (
        features["home_historical_home_win_rate"].fillna(0.0) - features["away_win_rate_last_n"].fillna(0.0)
    )

    # Final touch: fill NaNs with sensible defaults (0 for numeric features)
    numeric_cols = features.select_dtypes(include=["number"]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    # Join features back to the chronological match DataFrame
    result = df_work.set_index("_match_id").join(features, how="left", rsuffix="_feat")

    # Drop helper columns and keep only ML-ready columns + original match info
    # Keep original core columns and add engineered features
    keep_cols = [
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]
    engineered_cols = list(features.columns)

    out_cols = keep_cols + engineered_cols
    result = result[out_cols].reset_index(drop=True)

    logger.info("Feature engineering complete: produced %d features for %d matches", len(engineered_cols), len(result))
    return result


# If run as script for quick smoke test (requires data_loader and data files)
if __name__ == "__main__":
    try:
        import sys
        from data_loader import load_all_data  # type: ignore

        raw = load_all_data()
        feats = build_features(raw)
        print(feats.head())
    except Exception as e:
        logger.error("feature_engineer.py run failed: %s", e)
