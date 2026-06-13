#!/usr/bin/env python3
"""
run_real_test.py

Run a real end-to-end test using the existing clean dataset and the system pipeline.

Behavior:
- Verifies that data/processed/clean_dataset.csv exists (the script will abort otherwise)
- Calls run_system(retrain=True)
- Runs predict_game("Arsenal","Chelsea") and prints the JSON result to stdout

Rules: does not mock data. Executes actual pipeline code and prints real results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when run from repository root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Import the unified entrypoint
    from src.main_pipeline import run_system, predict_game
except Exception as e:
    print(f"ERROR: Failed to import pipeline entrypoint: {e}", file=sys.stderr)
    raise

CLEAN_PATH = Path("data/processed/clean_dataset.csv")

if not CLEAN_PATH.exists():
    print(f"ERROR: required dataset not found at {CLEAN_PATH}. Please generate or place your clean_dataset.csv at this path.", file=sys.stderr)
    sys.exit(2)

# Run the full system and force retraining
try:
    print("Running full system (this will retrain the model). This may take a while...", file=sys.stderr)
    res = run_system(retrain=True)
    print("run_system result:", file=sys.stderr)
    print(json.dumps(res, indent=2), file=sys.stderr)
except Exception as e:
    print(f"ERROR: run_system failed: {e}", file=sys.stderr)
    raise

# Perform prediction for Arsenal vs Chelsea
try:
    pred = predict_game("Arsenal", "Chelsea")
    # Print only the prediction dict to stdout as required (no markdown)
    print(json.dumps(pred))
except Exception as e:
    print(f"ERROR: predict_game failed: {e}", file=sys.stderr)
    raise
