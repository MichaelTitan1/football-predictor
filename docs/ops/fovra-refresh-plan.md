# Fovra refresh plan

- Football-Data historical/current sources are bootstrapped once when the GitHub Actions cache is empty.
- Subsequent scheduled refreshes use the cached raw baseline and run the incremental downloader.
- Football-Data and ClubElo refresh twice weekly on Monday and Wednesday.
- Weather is not part of the scheduled refresh.
- API-Football upcoming-match prediction/result ingestion remains a separate workflow to be added later.
