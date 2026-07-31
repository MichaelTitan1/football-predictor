-- Add Soccerdata FBref team statistics snapshots used by feature refreshes.
-- Additive only: no existing prediction or match data is replaced.

create table if not exists public.team_statistics (
    id uuid primary key default gen_random_uuid(),
    league_canonical_key text not null,
    team_slug text not null,
    team_name text not null,
    season text not null,
    xg numeric(10,4),
    xga numeric(10,4),
    goals numeric(10,4),
    shots numeric(10,4),
    shots_on_target numeric(10,4),
    possession numeric(10,4),
    passes numeric(10,4),
    pass_accuracy numeric(10,4),
    defensive_actions numeric(10,4),
    goalkeeper_stats numeric(10,4),
    raw_stats jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now(),
    unique (league_canonical_key, team_slug, season)
);

create index if not exists team_statistics_league_season_idx on public.team_statistics(league_canonical_key, season);
create index if not exists team_statistics_raw_stats_gin_idx on public.team_statistics using gin (raw_stats);

insert into public.data_sources(provider_key, display_name, base_url, freshness_policy)
values ('soccerdata-fbref', 'Soccerdata FBref', 'https://fbref.com', 'every six hours, best-effort by configured league')
on conflict (provider_key) do update set display_name = excluded.display_name, base_url = excluded.base_url, freshness_policy = excluded.freshness_policy, updated_at = now();

alter table public.team_statistics enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='team_statistics' and policyname='fovra_public_read_team_statistics') then
    create policy fovra_public_read_team_statistics on public.team_statistics for select to anon, authenticated using (true);
  end if;
end $$;

revoke insert, update, delete on public.team_statistics from anon, authenticated;
