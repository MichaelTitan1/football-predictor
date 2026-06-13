"""
clean_training_pipeline.py

Builds a clean training dataset and orchestrates training using only validated data.

Main entrypoints:
- build_clean_dataset(min_league_score=70) -> pd.DataFrame
- run_training_pipeline(min_league_score=70, min_rows=1000)

Behavior summary:
- Reads PROJECT_STATUS.md produced by src/data_pipeline/project_status.py
- Selects only leagues with completeness score >= min_league_score
- Loads only valid CSV files (as reported by PROJECT_STATUS.md) from data/raw
- Excludes missing seasons and broken files
- Merges files, deduplicates by Date+HomeTeam+AwayTeam, sorts by Date
- Attaches league metadata (Tier, LeagueStrength) from downloader's LEAGUE_CONFIG if available
- Saves final clean dataset to data/processed/clean_dataset.csv
- If dataset size meets min_rows, calls feature engineering and (if available) training routines
- If dataset too small or no valid leagues: stops and prints "DATA NOT READY FOR TRAINING"

Engineering notes:
- Defensive and logging-heavy to make decisions auditable in CI
- Compatible with GitHub Actions: deterministic file locations and no interactive prompts
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_PATH = PROCESSED_DIR / "clean_dataset.csv"
PROJECT_MD = Path("PROJECT_STATUS.md")
REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}

# Try to import downloader config to attach metadata, optional
try:
    from src.data_pipeline.data_downloader import LEAGUE_CONFIG
except Exception:
    LEAGUE_CONFIG = {}

# Try to import feature builder
try:
    from src.features.advanced_features import build_advanced_features
except Exception:
    build_advanced_features = None

# Try to import training entrypoints (optional)
try:
    from src.models import ensemble_model
except Exception:
    ensemble_model = None

try:
    from src.models import train_advanced as train_module
except Exception:
    train_module = None


def parse_project_status(md_path: Path = PROJECT_MD) -> Dict:
    """Parse PROJECT_STATUS.md and extract per-league completeness, missing seasons, broken files.

    Returns dict:
    {
      'leagues': { league: {'score': int, 'missing_seasons': [int,...], 'present_seasons': [int,...]} },
      'broken_files': [ {'file': name, 'reason': text}, ... ]
    }

    The parser is defensive and will return empty structures if the file doesn't exist or is unparseable.
    """
    result = {"leagues": {}, "broken_files": []}
    if not md_path.exists():
        logger.warning("%s not found — no project status available", md_path)
        return result

    text = md_path.read_text(encoding="utf-8")
    lines = [l.rstrip() for l in text.splitlines()]

    current_league = None
    for i, line in enumerate(lines):
        if line.startswith("### "):
            # league header
            current_league = line[4:].strip()
            result["leagues"].setdefault(current_league, {"score": 0, "missing_seasons": [], "present_seasons": []})
            continue
        if current_league:
            # look for completeness score
            if "League completeness score" in line:
                # line like: - League completeness score: 62 / 100
                try:
                    parts = line.split(":", 1)[1]
                    score_text = parts.strip().split()[0]
                    score = int(score_text)
                    result["leagues"][current_league]["score"] = score
                except Exception:
                    pass
                continue
            if line.startswith("- Missing seasons"):
                # line like: - Missing seasons (7): 2010,2011,2022
                try:
                    if ":" in line:
                        tail = line.split(":", 1)[1].strip()
                        if tail and tail.lower() != 'none':
                            years = [int(x.strip()) for x in tail.split(",") if x.strip().isdigit()]
                            result["leagues"][current_league]["missing_seasons"] = years
                except Exception:
                    pass
                continue
            if line.startswith("- Present seasons"):
                try:
                    if ":" in line:
                        tail = line.split(":", 1)[1].strip()
                        if tail and tail.lower() != 'none':
                            years = [int(x.strip()) for x in tail.split(",") if x.strip().isdigit()]
                            result["leagues"][current_league]["present_seasons"] = years
                except Exception:
                    pass
                continue
        # Broken files section
        if line.startswith("## Broken or invalid files"):
            # collect following lines until next header
            j = i + 1
            while j < len(lines) and not lines[j].startswith("##") and not lines[j].startswith("###"):
                l = lines[j].strip()
                if l.startswith("-"):
                    # format: - filename: reason
                    try:
                        rest = l[1:].strip()
                        if ":" in rest:
                            fname, reason = rest.split(":", 1)
                            result["broken_files"].append({"file": fname.strip(), "reason": reason.strip()})
                        else:
                            result["broken_files"].append({"file": rest.strip(), "reason": "unknown"})
                    except Exception:
                        pass
                j += 1
            # no further parsing of broken files here
            break

    return result


def _is_file_broken(fname: str, broken_list: List[Dict]) -> Optional[str]:
    for b in broken_list:
        if b.get("file") == fname:
            return b.get("reason")
    return None


def build_clean_dataset(min_league_score: int = 70) -> pd.DataFrame:
    """Build and save the clean_dataset.csv using only leagues that meet min_league_score.

    Returns the merged DataFrame (empty if none accepted).
    """
    status = parse_project_status()
    leagues_info = status.get("leagues", {})
    broken_files = status.get("broken_files", [])

    # Determine accepted leagues
    accepted_leagues = [l for l, v in leagues_info.items() if v.get("score", 0) >= min_league_score]
    logger.info("Accepted leagues (score>=%d): %s", min_league_score, accepted_leagues)

    if not accepted_leagues:
        logger.error("No leagues meet the minimum completeness score (%d). Aborting dataset build.", min_league_score)
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    accepted_files = []

    for league in accepted_leagues:
        present = leagues_info.get(league, {}).get("present_seasons", [])
        for season in present:
            fname = f"{league}_{season}.csv"
            fpath = RAW_DIR / fname
            if not fpath.exists():
                logger.warning("Expected file missing for accepted league: %s", fpath)
                continue
            # if file listed as broken, skip
            broken_reason = _is_file_broken(fname, broken_files)
            if broken_reason:
                logger.info("Skipping %s due to broken report: %s", fname, broken_reason)
                continue
            # quick validation of required columns
            try:
                df = pd.read_csv(fpath, parse_dates=["Date"], dayfirst=True)
            except Exception as e:
                logger.warning("Failed to read %s: %s", fpath, e)
                continue
            cols = {c.strip() for c in df.columns}
            missing = REQUIRED_COLS - cols
            if missing:
                logger.warning("Skipping %s because missing columns: %s", fpath, missing)
                continue
            # attach league metadata
            meta = LEAGUE_CONFIG.get(league, {})
            if "League" not in df.columns:
                df["League"] = league
            if "Tier" not in df.columns:
                df["Tier"] = meta.get("tier")
            if "LeagueStrength" not in df.columns:
                df["LeagueStrength"] = meta.get("strength")

            frames.append(df)
            accepted_files.append(str(fpath))
            logger.info("Included %s rows from %s", len(df), fpath)

    if not frames:
        logger.error("No valid files were included in the clean dataset. DATA NOT READY FOR TRAINING")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Deduplicate by Date+HomeTeam+AwayTeam
    try:
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    except Exception:
        pass
    merged = merged.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    merged["_dup_key"] = merged["Date"].dt.strftime("%Y-%m-%d") + "__" + merged["HomeTeam"].astype(str) + "__" + merged["AwayTeam"].astype(str)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["_dup_key"], keep="last").drop(columns=["_dup_key"]).reset_index(drop=True)
    after = len(merged)
    dup_removed = before - after
    if dup_removed > 0:
        logger.info("Removed %d duplicate match rows during clean merge", dup_removed)

    # Sort by Date
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Save clean dataset
    try:
        merged.to_csv(CLEAN_PATH, index=False)
        logger.info("Saved clean dataset to %s (%d rows, %d files)", CLEAN_PATH, len(merged), len(accepted_files))
    except Exception as e:
        logger.exception("Failed to save clean dataset: %s", e)

    return merged


def run_training_pipeline(min_league_score: int = 70, min_rows: int = 1000) -> None:
    """Orchestrate the full clean dataset build and training steps.

    - Build clean dataset
    - If dataset rows >= min_rows: run feature engineering and training (if training functions available)
    - Else: print DATA NOT READY FOR TRAINING and stop
    """
    logger.info("Starting training pipeline (min_league_score=%d min_rows=%d)", min_league_score, min_rows)
    clean_df = build_clean_dataset(min_league_score=min_league_score)
    if clean_df.empty or len(clean_df) < min_rows:
        logger.error("DATA NOT READY FOR TRAINING")
        print("DATA NOT READY FOR TRAINING")
        return

    logger.info("Clean dataset ready with %d rows — proceeding to feature engineering", len(clean_df))

    # Run feature engineering if available
    features_df = None
    if build_advanced_features is not None:
        try:
            # build_advanced_features expects a historical dataset; we pass merged clean_df
            features_df = build_advanced_features(clean_df)
            logger.info("Feature engineering produced %d rows and %d cols", len(features_df), features_df.shape[1])
        except Exception as e:
            logger.exception("Feature engineering failed: %s", e)
            features_df = None
    else:
        logger.warning("Feature engineering module not available; skipping feature build")

    # Call training routines (best-effort)
    trained = False
    # Prefer explicit train functions in train_module
    if train_module is not None:
        try:
            if hasattr(train_module, "train_catboost"):
                logger.info("Running train_module.train_catboost()")
                train_module.train_catboost(features_df if features_df is not None else clean_df)
                trained = True
            if hasattr(train_module, "train_ensemble"):
                logger.info("Running train_module.train_ensemble()")
                train_module.train_ensemble(features_df if features_df is not None else clean_df)
                trained = True
        except Exception as e:
            logger.exception("Training via train_module failed: %s", e)

    # Fallback to ensemble_model.train_ensemble
    if not trained and ensemble_model is not None:
        try:
            if hasattr(ensemble_model, "train_ensemble"):
                logger.info("Running ensemble_model.train_ensemble()")
                ensemble_model.train_ensemble(clean_df)
                trained = True
        except Exception as e:
            logger.exception("ensemble_model.train_ensemble failed: %s", e)

    if not trained:
        logger.warning("No training function executed — training step skipped (no available trainer in repo)")
    else:
        logger.info("Training pipeline completed")


if __name__ == "__main__":
    # Convenience: run the pipeline with defaults
    run_training_pipeline()
