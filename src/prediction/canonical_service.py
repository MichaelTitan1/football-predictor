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
import pandas as pd

from data_loader import load_all_data
from src.features.feature_engineer import build_features
from src.models.calibration import ConfidenceEstimator
from src.prediction.engine import load_prediction_model, prepare_match_features
from src.data_pipeline.neon_store import NeonStore


class CanonicalPredictionService:
    def __init__(self, model_path: str | None = None, calibration_path: str | None = None, store: NeonStore | None = None):
        self.model_path = model_path or os.getenv("FOOTBALL_MODEL_PATH", "models/football_model.cbm")
        self.calibration_path = calibration_path or os.getenv("FOOTBALL_CALIBRATION_PATH", "models/football_model_confidence.joblib")
        self.model_version = os.getenv("FOOTBALL_MODEL_VERSION", "catboost-v1-existing")
        self.feature_schema_version = os.getenv("FOOTBALL_FEATURE_SCHEMA_VERSION", "leakage-safe-v1")
        self.model = load_prediction_model(self.model_path)
        self.feature_data = build_features(load_all_data())
        self.store = store
        self.calibrator: ConfidenceEstimator | None = None
        if Path(self.calibration_path).exists():
            self.calibrator = ConfidenceEstimator.load(self.calibration_path)
        self.artifact_hash = hashlib.sha256(Path(self.model_path).read_bytes()).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {"version": self.model_version, "model_family": "catboost", "artifact_path": self.model_path, "artifact_sha256": self.artifact_hash, "feature_schema_version": self.feature_schema_version, "calibration_method": getattr(self.calibrator, "method", None)}

    @staticmethod
    def _team_slug(team_key: str) -> str:
        return str(team_key).split(":", 1)[-1]

    def _select_one(self, table: str, where: str, values: tuple[Any, ...], order: str = "") -> dict[str, Any] | None:
        if self.store is None:
            return None
        rows = self.store.select(table, where, values, order=order, limit=1)
        return rows[0] if rows else None

    def context_for_match(self, match: dict[str, Any]) -> dict[str, Any]:
        """Fetch live enrichment rows used for a prediction, without adding new providers."""
        league = str(match.get("league_canonical_key") or "")
        home_slug = self._team_slug(str(match.get("home_team_canonical_key") or ""))
        away_slug = self._team_slug(str(match.get("away_team_canonical_key") or ""))
        return {
            "match_id": match.get("source_match_id") or match.get("canonical_key"),
            "weather": self._select_one("match_weather", "match_canonical_key = %s", (match.get("canonical_key"),)),
            "home_team_statistics": self._select_one("team_statistics", "league_canonical_key = %s and team_slug = %s", (league, home_slug), "season desc"),
            "away_team_statistics": self._select_one("team_statistics", "league_canonical_key = %s and team_slug = %s", (league, away_slug), "season desc"),
            "home_team_strength": self._select_one("team_strength", "team_slug = %s", (home_slug,)),
            "away_team_strength": self._select_one("team_strength", "team_slug = %s", (away_slug,)),
        }

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _apply_enrichment_to_row(self, row: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
        """Consume team_statistics/team_strength values when model schema has matching features."""
        enriched = row.copy()
        home_stats = context.get("home_team_statistics") or {}
        away_stats = context.get("away_team_statistics") or {}
        home_strength = context.get("home_team_strength") or {}
        away_strength = context.get("away_team_strength") or {}
        replacements = {
            "home_elo_prior": self._float(home_strength.get("elo")),
            "away_elo_prior": self._float(away_strength.get("elo")),
            "expected_home_xg": self._float(home_stats.get("xg")),
            "expected_away_xg": self._float(away_stats.get("xg")),
            "home_avg_goals_for_prior": self._float(home_stats.get("goals")),
            "away_avg_goals_for_prior": self._float(away_stats.get("goals")),
            "home_avg_goals_against_prior": self._float(home_stats.get("xga")),
            "away_avg_goals_against_prior": self._float(away_stats.get("xga")),
        }
        for column, value in replacements.items():
            if value is not None and column in enriched.columns:
                enriched.loc[enriched.index[0], column] = value
        if {"expected_home_xg", "expected_away_xg"}.issubset(enriched.columns):
            enriched.loc[enriched.index[0], "xg_diff"] = float(enriched.iloc[0]["expected_home_xg"]) - float(enriched.iloc[0]["expected_away_xg"])
        if {"home_elo_prior", "away_elo_prior"}.issubset(enriched.columns):
            enriched.loc[enriched.index[0], "elo_diff_home_minus_away"] = float(enriched.iloc[0]["home_elo_prior"]) - float(enriched.iloc[0]["away_elo_prior"])
        return enriched

    def predict(self, home_team: str, away_team: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        row = prepare_match_features(home_team, away_team, self.feature_data, model=self.model)
        row = self._apply_enrichment_to_row(row, context)
        raw = np.asarray(self.model.predict_proba(row), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 3: raise RuntimeError("CatBoost model must return three 1X2 probabilities")
        probs = raw[0, :3]
        if self.calibrator is not None: probs = self.calibrator.calibration.predict_proba(raw)[0, :3]
        probs = np.clip(probs, 1e-8, 1.0); probs = probs / probs.sum()
        selected_index = int(np.argmax(probs)); selected = ["H", "D", "A"][selected_index]; confidence = float(probs[selected_index])
        feature_values = row.iloc[0].replace({np.nan: None}).to_dict()
        home_form = {k: feature_values.get(k) for k in ("home_form_short", "home_form_long") if k in feature_values}
        away_form = {k: feature_values.get(k) for k in ("away_form_short", "away_form_long") if k in feature_values}
        return {"home_team":home_team,"away_team":away_team,"probabilities":{"H":float(probs[0]),"D":float(probs[1]),"A":float(probs[2])},"selected_prediction":selected,"confidence":confidence,"model_version":self.model_version,"model_artifact_hash":self.artifact_hash,"feature_schema_version":self.feature_schema_version,"calibration_method":getattr(self.calibrator,"method",None),"data_freshness_at":str(self.feature_data["Date"].max()) if "Date" in self.feature_data.columns else None,"predicted_at":datetime.now(timezone.utc).isoformat(),"feature_values":feature_values,"home_elo":feature_values.get("home_elo_prior"),"away_elo":feature_values.get("away_elo_prior"),"home_xg":feature_values.get("expected_home_xg"),"away_xg":feature_values.get("expected_away_xg"),"home_form":home_form,"away_form":away_form,"weather":context.get("weather"),"match_id":context.get("match_id")}

    def archive(self, match_canonical_key: str, result: dict[str, Any], store: NeonStore) -> None:
        probs = result["probabilities"]; predicted_at=result["predicted_at"]
        prediction_key=hashlib.sha256(f"{match_canonical_key}|{predicted_at}|{result['model_version']}".encode()).hexdigest()
        archive_row={"prediction_key":prediction_key,"match_id":result.get("match_id") or match_canonical_key,"match_canonical_key":match_canonical_key,"predicted_at":predicted_at,"model_version":result["model_version"],"model_artifact_hash":result["model_artifact_hash"],"feature_schema_version":result["feature_schema_version"],"home_probability":probs["H"],"draw_probability":probs["D"],"away_probability":probs["A"],"selected_prediction":result["selected_prediction"],"confidence":result["confidence"],"home_elo":result.get("home_elo"),"away_elo":result.get("away_elo"),"home_xg":result.get("home_xg"),"away_xg":result.get("away_xg"),"home_form":result.get("home_form"),"away_form":result.get("away_form"),"weather":result.get("weather"),"feature_values":result.get("feature_values")}
        store.upsert("prediction_archive", [archive_row], "prediction_key")
        store.upsert("predictions", [{"match_canonical_key":match_canonical_key,"predicted_at":predicted_at,"model_version":result["model_version"],"model_artifact_hash":result["model_artifact_hash"],"home_probability":probs["H"],"draw_probability":probs["D"],"away_probability":probs["A"],"selected_prediction":result["selected_prediction"],"confidence":result["confidence"],"data_freshness_at":result.get("data_freshness_at"),"updated_at":datetime.now(timezone.utc).isoformat()}], "match_canonical_key")
