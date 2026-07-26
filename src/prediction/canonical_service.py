"""Canonical V1 prediction path.

Uses the existing leakage-safe feature engineering and CatBoost model. Optional
calibration is applied when a persisted ConfidenceEstimator is available. V1
archives only the primary 1X2 outcome probabilities; secondary markets are not
part of the canonical product contract.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from data_loader import load_all_data
from src.features.feature_engineer import build_features
from src.models.calibration import ConfidenceEstimator
from src.prediction.engine import load_prediction_model, prepare_match_features
from src.data_pipeline.supabase_store import SupabaseStore


class CanonicalPredictionService:
    def __init__(self, model_path: str | None = None, calibration_path: str | None = None):
        self.model_path = model_path or os.getenv("FOOTBALL_MODEL_PATH", "models/football_model.cbm")
        self.calibration_path = calibration_path or os.getenv("FOOTBALL_CALIBRATION_PATH", "models/football_model_confidence.joblib")
        self.model_version = os.getenv("FOOTBALL_MODEL_VERSION", "catboost-v1-existing")
        self.feature_schema_version = os.getenv("FOOTBALL_FEATURE_SCHEMA_VERSION", "leakage-safe-v1")
        self.model = load_prediction_model(self.model_path)
        self.feature_data = build_features(load_all_data())
        self.calibrator: ConfidenceEstimator | None = None
        if Path(self.calibration_path).exists():
            self.calibrator = ConfidenceEstimator.load(self.calibration_path)
        self.artifact_hash = hashlib.sha256(Path(self.model_path).read_bytes()).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.model_version,
            "model_family": "catboost",
            "artifact_path": self.model_path,
            "artifact_sha256": self.artifact_hash,
            "feature_schema_version": self.feature_schema_version,
            "calibration_method": getattr(self.calibrator, "method", None),
        }

    def predict(self, home_team: str, away_team: str) -> dict[str, Any]:
        row = prepare_match_features(home_team, away_team, self.feature_data, model=self.model)
        raw = np.asarray(self.model.predict_proba(row), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise RuntimeError("CatBoost model must return three 1X2 probabilities")
        probs = raw[0, :3]
        if self.calibrator is not None:
            probs = self.calibrator.calibration.predict_proba(raw)[0, :3]
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        selected_index = int(np.argmax(probs))
        selected = ["H", "D", "A"][selected_index]
        confidence = float(probs[selected_index])
        freshness = None
        if "Date" in self.feature_data.columns:
            freshness = str(self.feature_data["Date"].max())
        return {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": {"H": float(probs[0]), "D": float(probs[1]), "A": float(probs[2])},
            "selected_prediction": selected,
            "confidence": confidence,
            "model_version": self.model_version,
            "model_artifact_hash": self.artifact_hash,
            "feature_schema_version": self.feature_schema_version,
            "calibration_method": getattr(self.calibrator, "method", None),
            "data_freshness_at": freshness,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
        }

    def archive(self, match_canonical_key: str, result: dict[str, Any], store: SupabaseStore) -> None:
        probs = result["probabilities"]
        predicted_at = result["predicted_at"]
        prediction_key = hashlib.sha256(f"{match_canonical_key}|{predicted_at}|{result['model_version']}".encode()).hexdigest()
        archive_row = {
            "prediction_key": prediction_key,
            "match_canonical_key": match_canonical_key,
            "predicted_at": predicted_at,
            "model_version": result["model_version"],
            "model_artifact_hash": result["model_artifact_hash"],
            "home_probability": probs["H"],
            "draw_probability": probs["D"],
            "away_probability": probs["A"],
            "selected_prediction": result["selected_prediction"],
            "confidence": result["confidence"],
        }
        store.upsert("prediction_archive", [archive_row], "prediction_key")
        store.upsert(
            "predictions",
            [{
                **archive_row,
                "data_freshness_at": result.get("data_freshness_at"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }],
            "match_canonical_key",
        )
