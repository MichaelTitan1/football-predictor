# Fovra V1 Supabase verification

PR #2 cannot claim production-ready Supabase compatibility until the migration is checked against the existing Fovra project.

## 1. In Supabase SQL Editor

Use the existing Fovra Supabase project. Do not create a new project or database.

First run:

`supabase/migrations/202607260001_fovra_v1_foundation.sql`

Then run:

`supabase/verification/202607260001_fovra_v1_verify.sql`

The verification script is read-only.

## 2. Required checks

The verification output should show:

- all existing Fovra tables plus `data_sources`, `ingestion_runs`, and `model_versions`
- the canonical columns with the expected PostgreSQL types
- existing and new foreign keys
- canonical unique/index definitions
- RLS enabled on the canonical tables
- expected public read policies and no public write policies on the control/archive tables
- **zero duplicate canonical identities**
- **zero legacy NOT NULL/no-default columns that the canonical writer does not populate**

If the last query returns a row, do not treat the migration as compatible yet. That legacy column must either already be populated by a default/trigger or the canonical writer/migration must be adjusted deliberately.

## 3. Send back the results

Paste the SQL Editor results for sections 1-8 into the PR review/chat. The most important outputs are:

1. table/column types
2. foreign keys
3. indexes
4. RLS/policies
5. duplicate rows (must be empty)
6. legacy required columns (should be empty)

No secrets are needed for this review.

## 4. Real data ingestion after Supabase verification

The repository intentionally does not commit downloaded/generated football CSVs. They are ignored under `data/raw/*.csv` and `data/processed/*.csv`.

From a machine with internet access, after installing dependencies and setting the existing Supabase project credentials:

```bash
python -m pip install -r requirements.txt
export SUPABASE_URL='https://YOUR_PROJECT.supabase.co'
export SUPABASE_SERVICE_ROLE_KEY='YOUR_SERVICE_ROLE_KEY'
python -m src.data_pipeline.update_canonical
python -m src.prediction.generate_predictions
```

This uses the existing canonical path:

Football-Data.co.uk -> canonical provider -> Supabase -> prediction generation/archive/resolution.

For a local-only provider/SQLite validation without Supabase:

```bash
python -m src.data_pipeline.update_canonical --sqlite-local --offline
```

Do not use that local mode as production storage.

## 5. Historical data bootstrap

If historical raw CSVs are not present locally, download them with:

```bash
python -m src.data_pipeline.data_downloader
```

For an incremental/current-season refresh:

```bash
python -c "from src.data_pipeline.data_downloader import update_latest_season; print(update_latest_season())"
```

The current-season refresh deliberately re-downloads the current season even when a local copy already exists, because Football-Data files can receive later result updates.

Downloaded datasets remain local and are never committed.

## 6. Current free-provider limitation

Football-Data.co.uk is a periodic downloadable source, not a guaranteed real-time feed. The system records provider fetch time and source data time and must expose stale data rather than label it live.
