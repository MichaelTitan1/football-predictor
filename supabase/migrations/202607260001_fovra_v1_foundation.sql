-- Fovra V1 production data foundation.
-- Reuses existing leagues, teams, matches, predictions and prediction_archive tables.
-- The migration is additive: it does not create a second production database.

create extension if not exists pgcrypto;

-- Canonical identity and freshness columns on existing domain tables.
alter table public.leagues add column if not exists canonical_key text;
alter table public.leagues add column if not exists source_provider text;
alter table public.leagues add column if not exists source_updated_at timestamptz;
alter table public.leagues add column if not exists created_at timestamptz default now();
alter table public.leagues add column if not exists updated_at timestamptz default now();

alter table public.teams add column if not exists canonical_key text;
alter table public.teams add column if not exists league_canonical_key text;
alter table public.teams add column if not exists source_provider text;
alter table public.teams add column if not exists source_updated_at timestamptz;
alter table public.teams add column if not exists created_at timestamptz default now();
alter table public.teams add column if not exists updated_at timestamptz default now();

alter table public.matches add column if not exists canonical_key text;
alter table public.matches add column if not exists league_canonical_key text;
alter table public.matches add column if not exists season text;
alter table public.matches add column if not exists kickoff_at timestamptz;
alter table public.matches add column if not exists status text;
alter table public.matches add column if not exists home_team_canonical_key text;
alter table public.matches add column if not exists away_team_canonical_key text;
alter table public.matches add column if not exists home_score integer;
alter table public.matches add column if not exists away_score integer;
alter table public.matches add column if not exists source_provider text;
alter table public.matches add column if not exists source_match_id text;
alter table public.matches add column if not exists source_updated_at timestamptz;
alter table public.matches add column if not exists first_seen_at timestamptz default now();
alter table public.matches add column if not exists created_at timestamptz default now();
alter table public.matches add column if not exists updated_at timestamptz default now();

-- Current predictions table: latest/current state only. The archive below is the
-- accountability ledger and should never be overwritten as a prediction snapshot.
alter table public.predictions add column if not exists match_canonical_key text;
alter table public.predictions add column if not exists predicted_at timestamptz;
alter table public.predictions add column if not exists model_version text;
alter table public.predictions add column if not exists model_artifact_hash text;
alter table public.predictions add column if not exists home_probability numeric(8,7);
alter table public.predictions add column if not exists draw_probability numeric(8,7);
alter table public.predictions add column if not exists away_probability numeric(8,7);
alter table public.predictions add column if not exists selected_prediction text;
alter table public.predictions add column if not exists confidence numeric(8,7);
alter table public.predictions add column if not exists data_freshness_at timestamptz;
alter table public.predictions add column if not exists updated_at timestamptz default now();

-- Prediction archive is the accountability record. Existing columns are preserved;
-- these additions make the canonical V1 snapshot explicit.
alter table public.prediction_archive add column if not exists prediction_key text;
alter table public.prediction_archive add column if not exists match_canonical_key text;
alter table public.prediction_archive add column if not exists predicted_at timestamptz;
alter table public.prediction_archive add column if not exists model_version text;
alter table public.prediction_archive add column if not exists model_artifact_hash text;
alter table public.prediction_archive add column if not exists home_probability numeric(8,7);
alter table public.prediction_archive add column if not exists draw_probability numeric(8,7);
alter table public.prediction_archive add column if not exists away_probability numeric(8,7);
alter table public.prediction_archive add column if not exists selected_prediction text;
alter table public.prediction_archive add column if not exists confidence numeric(8,7);
alter table public.prediction_archive add column if not exists actual_result text;
alter table public.prediction_archive add column if not exists actual_home_score integer;
alter table public.prediction_archive add column if not exists actual_away_score integer;
alter table public.prediction_archive add column if not exists resolved_at timestamptz;
alter table public.prediction_archive add column if not exists is_correct boolean;
alter table public.prediction_archive add column if not exists created_at timestamptz default now();

-- New tables are only metadata/control-plane tables genuinely missing from the
-- existing schema.
create table if not exists public.data_sources (
    id uuid primary key default gen_random_uuid(),
    provider_key text not null unique,
    display_name text not null,
    base_url text,
    is_enabled boolean not null default true,
    freshness_policy text not null default 'periodic',
    last_attempt_at timestamptz,
    last_success_at timestamptz,
    last_data_at timestamptz,
    last_success_rows integer not null default 0,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.ingestion_runs (
    id uuid primary key default gen_random_uuid(),
    provider_key text not null references public.data_sources(provider_key) on update cascade,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    status text not null check (status in ('running','succeeded','failed','partial')),
    records_seen integer not null default 0,
    records_upserted integer not null default 0,
    newest_match_at timestamptz,
    error_message text,
    created_at timestamptz not null default now()
);

create table if not exists public.model_versions (
    id uuid primary key default gen_random_uuid(),
    version text not null unique,
    model_family text not null default 'catboost',
    artifact_path text,
    artifact_sha256 text,
    feature_schema_version text not null,
    calibration_method text,
    training_data_cutoff timestamptz,
    is_active boolean not null default false,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

-- Canonical uniqueness. Nullable legacy rows remain valid until they are
-- re-ingested; all new canonical records must have these keys.
create unique index if not exists leagues_canonical_key_uq on public.leagues(canonical_key) where canonical_key is not null;
create unique index if not exists teams_canonical_key_uq on public.teams(canonical_key) where canonical_key is not null;
create unique index if not exists matches_canonical_key_uq on public.matches(canonical_key) where canonical_key is not null;
create unique index if not exists predictions_match_uq on public.predictions(match_canonical_key) where match_canonical_key is not null;
create unique index if not exists prediction_archive_key_uq on public.prediction_archive(prediction_key) where prediction_key is not null;
create index if not exists matches_kickoff_status_idx on public.matches(kickoff_at, status);
create index if not exists matches_league_kickoff_idx on public.matches(league_canonical_key, kickoff_at);
create index if not exists matches_home_team_idx on public.matches(home_team_canonical_key, kickoff_at);
create index if not exists matches_away_team_idx on public.matches(away_team_canonical_key, kickoff_at);
create index if not exists prediction_archive_match_idx on public.prediction_archive(match_canonical_key, predicted_at desc);
create index if not exists prediction_archive_resolved_idx on public.prediction_archive(resolved_at);
create index if not exists ingestion_runs_provider_idx on public.ingestion_runs(provider_key, started_at desc);

insert into public.data_sources(provider_key, display_name, base_url, freshness_policy)
values ('football-data.co.uk', 'Football-Data.co.uk', 'https://www.football-data.co.uk', 'periodic')
on conflict (provider_key) do update set display_name = excluded.display_name, base_url = excluded.base_url, updated_at = now();

create or replace function public.fovra_set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists leagues_set_updated_at on public.leagues;
create trigger leagues_set_updated_at before update on public.leagues for each row execute function public.fovra_set_updated_at();
drop trigger if exists teams_set_updated_at on public.teams;
create trigger teams_set_updated_at before update on public.teams for each row execute function public.fovra_set_updated_at();
drop trigger if exists matches_set_updated_at on public.matches;
create trigger matches_set_updated_at before update on public.matches for each row execute function public.fovra_set_updated_at();
drop trigger if exists predictions_set_updated_at on public.predictions;
create trigger predictions_set_updated_at before update on public.predictions for each row execute function public.fovra_set_updated_at();

create or replace function public.fovra_protect_prediction_archive()
returns trigger language plpgsql as $$
begin
  if tg_op = 'DELETE' then
    raise exception 'prediction_archive is append-only; rows cannot be deleted';
  end if;
  if tg_op = 'UPDATE' then
    if new.prediction_key is distinct from old.prediction_key
       or new.match_canonical_key is distinct from old.match_canonical_key
       or new.predicted_at is distinct from old.predicted_at
       or new.model_version is distinct from old.model_version
       or new.model_artifact_hash is distinct from old.model_artifact_hash
       or new.home_probability is distinct from old.home_probability
       or new.draw_probability is distinct from old.draw_probability
       or new.away_probability is distinct from old.away_probability
       or new.selected_prediction is distinct from old.selected_prediction
       or new.confidence is distinct from old.confidence
    then
      raise exception 'prediction_archive prediction snapshot is immutable';
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists prediction_archive_immutable on public.prediction_archive;
create trigger prediction_archive_immutable before update or delete on public.prediction_archive for each row execute function public.fovra_protect_prediction_archive();

alter table public.leagues enable row level security;
alter table public.teams enable row level security;
alter table public.matches enable row level security;
alter table public.predictions enable row level security;
alter table public.prediction_archive enable row level security;
alter table public.data_sources enable row level security;
alter table public.ingestion_runs enable row level security;
alter table public.model_versions enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='leagues' and policyname='fovra_public_read_leagues') then
    create policy fovra_public_read_leagues on public.leagues for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='teams' and policyname='fovra_public_read_teams') then
    create policy fovra_public_read_teams on public.teams for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='matches' and policyname='fovra_public_read_matches') then
    create policy fovra_public_read_matches on public.matches for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='predictions' and policyname='fovra_public_read_predictions') then
    create policy fovra_public_read_predictions on public.predictions for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='prediction_archive' and policyname='fovra_public_read_archive') then
    create policy fovra_public_read_archive on public.prediction_archive for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='data_sources' and policyname='fovra_public_read_sources') then
    create policy fovra_public_read_sources on public.data_sources for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='model_versions' and policyname='fovra_public_read_model_versions') then
    create policy fovra_public_read_model_versions on public.model_versions for select to anon, authenticated using (is_active = true);
  end if;
end $$;

revoke insert, update, delete on public.data_sources from anon, authenticated;
revoke insert, update, delete on public.ingestion_runs from anon, authenticated;
revoke insert, update, delete on public.model_versions from anon, authenticated;
revoke insert, update, delete on public.predictions from anon, authenticated;
revoke insert, update, delete on public.prediction_archive from anon, authenticated;
