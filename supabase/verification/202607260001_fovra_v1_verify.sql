-- Fovra V1 Supabase verification (read-only)
-- Run this AFTER applying 202607260001_fovra_v1_foundation.sql in Supabase SQL Editor.
-- It does not modify data.

-- 1) Required tables must exist.
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in ('leagues','teams','matches','predictions','prediction_archive','users','news','match_statistics','data_sources','ingestion_runs','model_versions')
order by table_name;

-- 2) Required canonical columns and their actual database types.
select table_name, column_name, data_type, udt_name, is_nullable
from information_schema.columns
where table_schema = 'public'
  and (
    (table_name = 'leagues' and column_name in ('canonical_key','source_provider','source_updated_at','created_at','updated_at')) or
    (table_name = 'teams' and column_name in ('canonical_key','league_canonical_key','source_provider','source_updated_at','created_at','updated_at')) or
    (table_name = 'matches' and column_name in ('canonical_key','league_canonical_key','season','kickoff_at','status','home_team_canonical_key','away_team_canonical_key','home_score','away_score','source_provider','source_match_id','source_updated_at','first_seen_at','created_at','updated_at')) or
    (table_name = 'predictions' and column_name in ('match_canonical_key','predicted_at','model_version','model_artifact_hash','home_probability','draw_probability','away_probability','selected_prediction','confidence','data_freshness_at','updated_at')) or
    (table_name = 'prediction_archive' and column_name in ('prediction_key','match_canonical_key','predicted_at','model_version','model_artifact_hash','home_probability','draw_probability','away_probability','selected_prediction','confidence','actual_result','actual_home_score','actual_away_score','resolved_at','is_correct','created_at'))
  )
order by table_name, ordinal_position;

-- 3) Added control-plane table definitions.
select table_name, column_name, data_type, is_nullable
from information_schema.columns
where table_schema = 'public'
  and table_name in ('data_sources','ingestion_runs','model_versions')
order by table_name, ordinal_position;

-- 4) Foreign keys relevant to the canonical layer.
select tc.table_name, kcu.column_name, ccu.table_name as referenced_table, ccu.column_name as referenced_column, tc.constraint_name
from information_schema.table_constraints tc
join information_schema.key_column_usage kcu on tc.constraint_name = kcu.constraint_name and tc.table_schema = kcu.table_schema
join information_schema.constraint_column_usage ccu on ccu.constraint_name = tc.constraint_name and ccu.table_schema = tc.table_schema
where tc.constraint_type = 'FOREIGN KEY'
  and tc.table_schema = 'public'
  and tc.table_name in ('ingestion_runs','leagues','teams','matches','predictions','prediction_archive')
order by tc.table_name, tc.constraint_name;

-- 5) Canonical indexes/unique indexes.
select schemaname, tablename, indexname, indexdef
from pg_indexes
where schemaname = 'public'
  and (
    indexname like '%canonical_key%' or
    indexname in ('predictions_match_uq','prediction_archive_key_uq','matches_kickoff_status_idx','matches_league_kickoff_idx','matches_home_team_idx','matches_away_team_idx','prediction_archive_match_idx','prediction_archive_resolved_idx','ingestion_runs_provider_idx')
  )
order by tablename, indexname;

-- 6) RLS state and policies.
select n.nspname as schema_name, c.relname as table_name, c.relrowsecurity as rls_enabled, c.relforcerowsecurity as forced_rls
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in ('leagues','teams','matches','predictions','prediction_archive','data_sources','ingestion_runs','model_versions')
order by c.relname;

select schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
from pg_policies
where schemaname = 'public'
  and tablename in ('leagues','teams','matches','predictions','prediction_archive','data_sources','ingestion_runs','model_versions')
order by tablename, policyname;

-- 7) Duplicate canonical identities must be zero before production ingestion.
select 'leagues' as entity, canonical_key, count(*)
from public.leagues where canonical_key is not null group by canonical_key having count(*) > 1
union all
select 'teams', canonical_key, count(*)
from public.teams where canonical_key is not null group by canonical_key having count(*) > 1
union all
select 'matches', canonical_key, count(*)
from public.matches where canonical_key is not null group by canonical_key having count(*) > 1
union all
select 'predictions', match_canonical_key, count(*)
from public.predictions where match_canonical_key is not null group by match_canonical_key having count(*) > 1
union all
select 'prediction_archive', prediction_key, count(*)
from public.prediction_archive where prediction_key is not null group by prediction_key having count(*) > 1;

-- 8) Existing legacy columns that could make the additive migration incompatible
-- with the canonical writer because of NOT NULL requirements. This should return
-- zero rows for columns not explicitly populated by the canonical ingestion path.
select table_name, column_name, data_type, is_nullable, column_default
from information_schema.columns
where table_schema = 'public'
  and table_name in ('leagues','teams','matches','predictions','prediction_archive')
  and is_nullable = 'NO'
  and column_default is null
  and column_name not in (
    'canonical_key','source_provider','source_updated_at','created_at','updated_at',
    'league_canonical_key','season','kickoff_at','status','home_team_canonical_key','away_team_canonical_key',
    'home_score','away_score','source_match_id','first_seen_at',
    'match_canonical_key','predicted_at','model_version','model_artifact_hash','home_probability','draw_probability','away_probability','selected_prediction','confidence','data_freshness_at',
    'prediction_key','actual_result','actual_home_score','actual_away_score','resolved_at','is_correct'
  )
order by table_name, ordinal_position;

-- Expected interpretation:
-- * Section 1: all 11 required tables present.
-- * Section 2/3: required columns/types match the migration.
-- * Section 4: no unexpected type conflicts; ingestion_runs.provider_key -> data_sources.provider_key is present.
-- * Section 5: canonical unique/index set is present.
-- * Section 6: RLS is enabled; public read policies exist only where intended.
-- * Section 7: ZERO rows.
-- * Section 8: ZERO rows is the safest result. Any returned legacy NOT NULL column
--   must be reviewed before claiming the canonical upsert is compatible.
