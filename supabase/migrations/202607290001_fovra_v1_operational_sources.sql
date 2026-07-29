-- Fovra V1 operational API-Football, ClubElo and MET Norway extensions.
-- Additive only: reuses existing football entity tables and adds missing enrichment tables.

alter table public.matches add column if not exists venue_name text;
alter table public.matches add column if not exists venue_latitude numeric(9,6);
alter table public.matches add column if not exists venue_longitude numeric(9,6);

create table if not exists public.league_standings (
    id uuid primary key default gen_random_uuid(),
    league_canonical_key text not null,
    team_canonical_key text not null,
    season text not null,
    rank integer,
    points integer,
    played integer,
    wins integer,
    draws integer,
    losses integer,
    goals_for integer,
    goals_against integer,
    updated_at timestamptz not null default now(),
    unique (league_canonical_key, team_canonical_key, season)
);

create table if not exists public.team_strength (
    id uuid primary key default gen_random_uuid(),
    team_slug text not null unique,
    team_name text not null,
    elo numeric(8,3),
    rank integer,
    country text,
    level text,
    from_date date,
    to_date date,
    updated_at timestamptz not null default now()
);

create table if not exists public.match_weather (
    id uuid primary key default gen_random_uuid(),
    match_canonical_key text not null unique,
    forecast_at timestamptz,
    temperature_c numeric(6,2),
    rain_mm numeric(7,2),
    wind_mps numeric(7,2),
    condition text,
    updated_at timestamptz not null default now()
);

create index if not exists league_standings_league_rank_idx on public.league_standings(league_canonical_key, season, rank);
create index if not exists team_strength_elo_idx on public.team_strength(elo desc);
create index if not exists match_weather_forecast_idx on public.match_weather(forecast_at);

insert into public.data_sources(provider_key, display_name, base_url, freshness_policy)
values
  ('api-football', 'API-Football', 'https://v3.football.api-sports.io', 'fixtures daily, results every two hours'),
  ('clubelo', 'ClubElo', 'http://api.clubelo.com', 'daily'),
  ('met-norway', 'MET Norway Locationforecast', 'https://api.met.no/weatherapi/locationforecast/2.0', 'fixtures inside next 24 hours every four hours')
on conflict (provider_key) do update set display_name = excluded.display_name, base_url = excluded.base_url, freshness_policy = excluded.freshness_policy, updated_at = now();

alter table public.league_standings enable row level security;
alter table public.team_strength enable row level security;
alter table public.match_weather enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='league_standings' and policyname='fovra_public_read_standings') then
    create policy fovra_public_read_standings on public.league_standings for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='team_strength' and policyname='fovra_public_read_team_strength') then
    create policy fovra_public_read_team_strength on public.team_strength for select to anon, authenticated using (true);
  end if;
  if not exists (select 1 from pg_policies where schemaname='public' and tablename='match_weather' and policyname='fovra_public_read_match_weather') then
    create policy fovra_public_read_match_weather on public.match_weather for select to anon, authenticated using (true);
  end if;
end $$;

revoke insert, update, delete on public.league_standings from anon, authenticated;
revoke insert, update, delete on public.team_strength from anon, authenticated;
revoke insert, update, delete on public.match_weather from anon, authenticated;
