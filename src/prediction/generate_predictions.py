"""Generate and archive V1 predictions for upcoming canonical fixtures.

This job is intended for a scheduled backend worker, not frontend requests.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from api.repository import FovraRepository
from src.data_pipeline.supabase_store import SupabaseStore
from src.prediction.canonical_service import CanonicalPredictionService


def run(limit: int = 100) -> dict:
    repo = FovraRepository()
    store = SupabaseStore()
    service = CanonicalPredictionService()
    metadata = service.metadata()
    store.upsert("model_versions", [{**metadata, "is_active": True}], "version")
    matches = repo.upcoming(limit)
    generated = 0
    errors = []
    for match in matches:
        try:
            home = str(match["home_team_canonical_key"]).split(":", 1)[-1]
            away = str(match["away_team_canonical_key"]).split(":", 1)[-1]
            result = service.predict(home, away)
            service.archive(str(match["canonical_key"]), result, store)
            generated += 1
        except Exception as exc:
            errors.append({"match": match.get("canonical_key"), "error": str(exc)})
    return {"generated": generated, "requested": len(matches), "errors": errors, "model": metadata, "finished_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
