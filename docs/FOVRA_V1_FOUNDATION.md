# Fovra V1 Production Foundation

## Architecture

```text
Free football sources
        |
        v
Canonical provider layer
        |
        v
Supabase PostgreSQL  <-- production source of truth
        |
        v
Fovra FastAPI
        |
        v
Fovra frontend
```

SQLite from the earlier draft is **not** a production dependency. The old local
store remains available only through `--sqlite-local --offline` for isolated
unit/development testing. It is never selected by the production update command.

## Supabase reconciliation

Existing tables are reused:

- `leagues` — canonical competition identity + provider freshness fields
- `teams` — canonical team identity + league/provider fields
- `matches` — canonical fixtures/results + kickoff/status/scores/source fields
- `predictions` — latest prediction per canonical match
- `prediction_archive` — immutable prediction snapshots with eventual result resolution
- `match_statistics` — existing table retained for future statistics ingestion
- `users` — existing application/auth data retained
- `news` — existing table retained but outside focused V1 scope

Only genuinely missing control-plane entities are added:

- `data_sources` — provider registry and latest freshness/error status
- `ingestion_runs` — auditable update attempts and row counts
- `model_versions` — reproducible model/calibration metadata

No second database service is introduced.

## Canonical identity and idempotency

Every newly ingested league, team and match receives a stable `canonical_key`.
The match key is a SHA-256 digest of provider + league + season + UTC kickoff +
home team + away team. Supabase unique partial indexes reject duplicate
canonical records.

A fixture can be ingested first as `scheduled` and later re-ingested as
`finished`; the same canonical key is updated with the result instead of
creating another match.

## Prediction path

The canonical prediction path is `src/prediction/canonical_service.py`.
It keeps the existing CatBoost model and existing leakage-safe feature
engineering. The existing calibration implementation is used when its persisted
calibration artifact is available; otherwise raw CatBoost probabilities are
reported rather than pretending calibration exists.

V1 only archives the primary 1X2 probabilities. The current prediction table is
updated by canonical match key, while every prediction event is retained in
`prediction_archive` with:

- prediction timestamp
- match canonical key
- model version
- model artifact SHA-256
- home/draw/away probabilities
- selected prediction
- confidence
- eventual actual result and scores
- resolution timestamp
- correctness

The database trigger prevents changing or deleting the original prediction
snapshot. Only result-resolution fields may be updated.

## Result resolution

Every successful data ingestion runs prediction resolution. When a match becomes
finished, any unresolved archive rows for that canonical match receive the
actual result, final score, resolution timestamp and correctness flag.

## Data freshness

Football-Data.co.uk is the initial free provider. It is not treated as a live
feed. Fovra stores source attempt time, successful update time, newest data time,
row counts and errors. The API exposes `/api/v1/data-freshness`, and the frontend
shows the freshness state.

The provider is isolated behind `FootballDataProvider`; another free provider
can later implement the same snapshot contract without changing Supabase or the
API model.

## Scheduled updates

`.github/workflows/fovra-data-update.yml` runs every six hours and can also be
started manually. It performs:

1. canonical Football-Data ingestion into Supabase
2. automatic prediction-result resolution
3. persisted CatBoost predictions for upcoming canonical fixtures

GitHub Actions secrets required:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

The service key is backend/job-only and must never be sent to the browser.

## API

- `GET /health`
- `GET /api/v1/matches/today`
- `GET /api/v1/matches/upcoming`
- `GET /api/v1/matches/{canonical_key}`
- `GET /api/v1/matches/{canonical_key}/prediction`
- `GET /api/v1/predictions`
- `GET /api/v1/best-picks`
- `GET /api/v1/prediction-archive`
- `GET /api/v1/teams`
- `GET /api/v1/teams/{team_key}/form`
- `GET /api/v1/teams/{team_key}/h2h/{opponent_key}`
- `GET /api/v1/leagues`
- `GET /api/v1/leagues/{league_key}/standings`
- `GET /api/v1/leagues/{league_key}/fixtures`
- `GET /api/v1/data-freshness`

The frontend consumes persisted API data; it does not call football-data sources.

## Deployment

The backend/frontend can run together in the supplied Docker image. Configure
`.env` values from `.env.example`. The service listens on `PORT` (default 8000)
and serves the frontend at `/app/`.

## Known limitations

1. The repository-visible project does not include the actual `data/raw` and
   `data/processed` historical CSV datasets, so the ingestion path cannot be
   validated end-to-end against those repository-local rows in this environment.
2. The connected GitHub tooling cannot inspect the live Supabase project's
   catalog. The migration therefore reconciles the explicitly supplied existing
   table names additively, rather than claiming to have verified every existing
   column/constraint. Before applying the migration to production, run it in the
   Supabase SQL editor and review any conflicts from legacy NOT NULL columns.
3. Football-Data's free fixtures/results feeds are periodic rather than
   guaranteed real-time. Fovra intentionally exposes freshness instead of
   hiding that limitation.
4. Match statistics/news/player intelligence are outside focused V1 ingestion;
   existing tables remain untouched for future providers/features.
