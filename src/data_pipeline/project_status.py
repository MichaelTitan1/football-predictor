"""
project_status.py

Scan data/raw and produce PROJECT_STATUS.md summarizing real dataset state.

Usage:
    python src/data_pipeline/project_status.py

This script writes PROJECT_STATUS.md at the repository root. It is safe to run in CI or locally.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Try to import downloader config to get allowed leagues and years
try:
    from src.data_pipeline.data_downloader import ALLOWED_LEAGUES, START_YEAR, CURRENT_YEAR, LEAGUE_CONFIG
except Exception:
    ALLOWED_LEAGUES = []
    START_YEAR = 2010
    CURRENT_YEAR = datetime.datetime.now().year
    LEAGUE_CONFIG = {}

RAW_DIR = Path("data/raw")
PROJECT_MD = Path("PROJECT_STATUS.md")
REQUIRED_COLS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


def _parse_filename(fname: str) -> Tuple[str, int]:
    """Parse filenames of form LEAGUE_YYYY.csv -> (league, year) or (None, None)"""
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        league = "_".join(parts[:-1])
        try:
            year = int(parts[-1])
            return league, year
        except Exception:
            return league, None
    return None, None


def scan_datasets() -> Dict:
    report = {
        "leagues": {},
        "broken_files": [],
        "duplicates": {},
    }

    if not RAW_DIR.exists():
        logger.info("No data/raw directory found.")
        return report

    files = sorted(RAW_DIR.glob("*.csv"))
    for f in files:
        league, year = _parse_filename(f.name)
        if league is None or year is None:
            report["broken_files"].append({"file": f.name, "reason": "unrecognized_name"})
            continue
        # initialize league entry
        if league not in report["leagues"]:
            report["leagues"][league] = {"files": {}, "total_rows": 0, "valid_seasons": []}
        try:
            df = pd.read_csv(f, parse_dates=["Date"], dayfirst=True)
        except Exception as e:
            report["broken_files"].append({"file": f.name, "reason": f"parse_error: {e}"})
            continue
        cols = {c.strip() for c in df.columns}
        missing = REQUIRED_COLS - cols
        if missing:
            report["broken_files"].append({"file": f.name, "reason": f"missing_columns: {sorted(list(missing))}"})
            continue
        # valid file
        nrows = len(df)
        report["leagues"][league]["files"][year] = {"rows": int(nrows), "path": str(f)}
        report["leagues"][league]["total_rows"] += int(nrows)
        report["leagues"][league]["valid_seasons"].append(year)
        # duplicates detection per file
        try:
            df_keys = df["Date"].dt.strftime("%Y-%m-%d") + "__" + df["HomeTeam"].astype(str) + "__" + df["AwayTeam"].astype(str)
            dup_count = df_keys.duplicated().sum()
            if dup_count > 0:
                report["duplicates"][f.name] = int(dup_count)
        except Exception:
            # ignore duplicate detection failures
            pass

    return report


def compute_completeness(report: Dict) -> Dict:
    leagues = report.get("leagues", {})
    completeness = {}
    scores = []
    expected_years = list(range(START_YEAR, CURRENT_YEAR + 1))
    expected_count = len(expected_years)

    # If ALLOWED_LEAGUES provided, restrict to them; otherwise use discovered leagues
    target_leagues = ALLOWED_LEAGUES if ALLOWED_LEAGUES else sorted(leagues.keys())

    for league in target_leagues:
        info = leagues.get(league, {"valid_seasons": [], "total_rows": 0})
        valid = sorted(info.get("valid_seasons", []))
        present = len(valid)
        missing = [y for y in expected_years if y not in valid]
        score = int((present / expected_count) * 100) if expected_count > 0 else 0
        # penalize if there are broken files for this league
        broken_penalty = 0
        # accumulate
        completeness[league] = {
            "present_seasons": valid,
            "present_count": present,
            "missing_seasons": missing,
            "total_rows": int(info.get("total_rows", 0)),
            "score": score,
        }
        scores.append(score)

    overall = int(sum(scores) / len(scores)) if scores else 0
    return {"per_league": completeness, "overall_score": overall}


def render_markdown(report: Dict, completeness: Dict) -> str:
    lines = []
    lines.append("# Project Dataset Status")
    lines.append("")
    lines.append(f"Generated: {datetime.datetime.utcnow().isoformat()} UTC")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Allowed leagues considered: {', '.join(ALLOWED_LEAGUES) if ALLOWED_LEAGUES else 'auto-detected'}")
    lines.append(f"- Expected seasons: {START_YEAR} → {CURRENT_YEAR} ({CURRENT_YEAR - START_YEAR + 1} seasons)")
    lines.append("")
    lines.append(f"### Overall completeness score: **{completeness['overall_score']} / 100**")
    lines.append("")
    lines.append("## Per-league details")
    lines.append("")

    for league, stats in completeness["per_league"].items():
        lines.append(f"### {league}")
        lines.append("")
        lines.append(f"- Present seasons ({stats['present_count']}): {', '.join(str(y) for y in sorted(stats['present_seasons'])) if stats['present_seasons'] else 'None'}")
        lines.append(f"- Missing seasons ({len(stats['missing_seasons'])}): {', '.join(str(y) for y in stats['missing_seasons']) if stats['missing_seasons'] else 'None'}")
        lines.append(f"- Total rows: {stats['total_rows']}")
        lines.append(f"- League completeness score: {stats['score']} / 100")
        lines.append("")

    lines.append("## Broken or invalid files")
    lines.append("")
    if report.get("broken_files"):
        lines.append("Files rejected during scan:")
        for b in report.get("broken_files"):
            lines.append(f"- {b['file']}: {b['reason']}")
    else:
        lines.append("None")
    lines.append("")

    lines.append("## Duplicate detections (per-file)")
    lines.append("")
    if report.get("duplicates"):
        for fname, cnt in report.get("duplicates").items():
            lines.append(f"- {fname}: {cnt} duplicated match rows")
    else:
        lines.append("None detected")
    lines.append("")

    lines.append("## Notes & Next steps")
    lines.append("")
    lines.append("- Run `python src/data_pipeline/project_status.py` locally or in CI to regenerate this report.")
    lines.append("- If some Tier-2 leagues are missing, update src/data_pipeline/data_downloader.py LEAGUE_CONFIG with vetted sources.")
    lines.append("")
    return "\n".join(lines)


def main():
    report = scan_datasets()
    completeness = compute_completeness(report)
    md = render_markdown(report, completeness)
    PROJECT_MD.write_text(md, encoding="utf-8")
    logger.info("Wrote %s", PROJECT_MD)


if __name__ == "__main__":
    main()
