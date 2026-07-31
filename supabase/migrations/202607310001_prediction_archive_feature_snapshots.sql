-- Store immutable prediction-time feature snapshots for model auditing and retraining.
-- Additive only: extends the existing append-only prediction_archive ledger.

alter table public.prediction_archive add column if not exists match_id text;
alter table public.prediction_archive add column if not exists feature_schema_version text;
alter table public.prediction_archive add column if not exists home_elo numeric(10,3);
alter table public.prediction_archive add column if not exists away_elo numeric(10,3);
alter table public.prediction_archive add column if not exists home_xg numeric(8,4);
alter table public.prediction_archive add column if not exists away_xg numeric(8,4);
alter table public.prediction_archive add column if not exists home_form jsonb;
alter table public.prediction_archive add column if not exists away_form jsonb;
alter table public.prediction_archive add column if not exists weather jsonb;
alter table public.prediction_archive add column if not exists odds jsonb;
alter table public.prediction_archive add column if not exists feature_values jsonb;

comment on column public.prediction_archive.match_id is 'Optional legacy/source match identifier captured at prediction time; match_canonical_key remains the canonical immutable match identity.';
comment on column public.prediction_archive.feature_schema_version is 'Feature schema version used to build feature_values for this prediction event.';
comment on column public.prediction_archive.home_elo is 'Home team Elo value actually used at prediction time, copied from feature_values for queryability.';
comment on column public.prediction_archive.away_elo is 'Away team Elo value actually used at prediction time, copied from feature_values for queryability.';
comment on column public.prediction_archive.home_xg is 'Home expected-goals feature value actually used at prediction time, copied from feature_values for queryability.';
comment on column public.prediction_archive.away_xg is 'Away expected-goals feature value actually used at prediction time, copied from feature_values for queryability.';
comment on column public.prediction_archive.home_form is 'Home form feature snapshot actually used at prediction time, for example short and long form values.';
comment on column public.prediction_archive.away_form is 'Away form feature snapshot actually used at prediction time, for example short and long form values.';
comment on column public.prediction_archive.weather is 'Weather snapshot available to the prediction at prediction time; null when unavailable or unused.';
comment on column public.prediction_archive.odds is 'Odds snapshot available to the prediction at prediction time; null when unavailable or unused.';
comment on column public.prediction_archive.feature_values is 'Complete model input feature vector actually passed to the model for this prediction event.';

create index if not exists prediction_archive_model_version_idx on public.prediction_archive(model_version, predicted_at desc);
create index if not exists prediction_archive_feature_values_gin_idx on public.prediction_archive using gin (feature_values);

create or replace function public.fovra_protect_prediction_archive()
returns trigger language plpgsql as $$
begin
  if tg_op = 'DELETE' then
    raise exception 'prediction_archive is append-only; rows cannot be deleted';
  end if;
  if tg_op = 'UPDATE' then
    if new.prediction_key is distinct from old.prediction_key
       or new.match_id is distinct from old.match_id
       or new.match_canonical_key is distinct from old.match_canonical_key
       or new.predicted_at is distinct from old.predicted_at
       or new.model_version is distinct from old.model_version
       or new.model_artifact_hash is distinct from old.model_artifact_hash
       or new.feature_schema_version is distinct from old.feature_schema_version
       or new.home_probability is distinct from old.home_probability
       or new.draw_probability is distinct from old.draw_probability
       or new.away_probability is distinct from old.away_probability
       or new.selected_prediction is distinct from old.selected_prediction
       or new.confidence is distinct from old.confidence
       or new.home_elo is distinct from old.home_elo
       or new.away_elo is distinct from old.away_elo
       or new.home_xg is distinct from old.home_xg
       or new.away_xg is distinct from old.away_xg
       or new.home_form is distinct from old.home_form
       or new.away_form is distinct from old.away_form
       or new.weather is distinct from old.weather
       or new.odds is distinct from old.odds
       or new.feature_values is distinct from old.feature_values
    then
      raise exception 'prediction_archive prediction snapshot is immutable';
    end if;
  end if;
  return new;
end;
$$;
