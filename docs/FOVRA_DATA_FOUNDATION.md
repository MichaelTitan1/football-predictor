# Fovra V1 Data Foundation

## Canonical path

```text
Football-Data.co.uk / existing data/raw CSVs
                |
                v
       FootballDataProvider
                |
                v
     canonical validation/schema
                |
                v
       SQLite canonical store
                |
        +-------+-------+
        |               |
      matches       leagues/teams
```

The provider boundary is intentionally replaceable. A future provider must emit
`LeagueRecord`, `TeamRecord`, and `MatchRecord` objects and does not need to know
anything about the storage layer.

## Current provider

The first V1 provider is Football-Data.co.uk because it is free, publishes
computer-readable historical results, and publishes an upcoming-fixtures CSV.
The existing repository downloader already uses its season CSV convention, so
this implementation preserves that useful historical foundation rather than
replacing it.

The provider refreshes the active season and the public fixture feed when the
network is available. Existing `data/raw/*.csv` files are also ingested. No
simulated matches are generated.

## Canonical storage

`data/processed/fovra_data.sqlite3` is created locally and ignored by git.
Tables:

- `leagues`: canonical competition records
- `teams`: canonical team records scoped to league
- `matches`: scheduled and completed matches with scores where available
- `source_updates`: provider freshness/update metadata and errors

Match upserts are deterministic and idempotent. A repeated update does not
create duplicate matches. A later completed result can update an earlier
scheduled record.

## Commands

Read-only audit:

```bash
python -m src.data_pipeline.data_audit
```

Offline ingestion of existing repository data only:

```bash
python -m src.data_pipeline.update_canonical --offline
```

Normal update (existing raw data + current remote source):

```bash
python -m src.data_pipeline.update_canonical
```

## Data contract

Canonical matches contain:

- league
- season
- UTC kickoff timestamp
- home team
- away team
- status (`scheduled` or `finished`, with reserved postponed/cancelled states)
- home score / away score when completed
- source/provider metadata

This is sufficient groundwork for the existing prediction feature pipeline and
for future fixtures, results, archive, and match APIs.

## Provider limitations

Football-Data.co.uk is a good free foundation for V1 but is not a guaranteed
real-time/live feed. Its fixtures page states that fixtures are collected on a
weekly cadence, and its historical result files are updated periodically.
Fovra therefore records source update timestamps and must expose freshness
rather than pretending the feed is real-time.

TheSportsDB is a second free candidate for future provider work, with free V1
schedule endpoints and a documented 30 requests/minute free limit. It should
only be added as a second provider after its coverage and terms are validated
for Fovra's exact leagues; it is not wired into the canonical path yet.
