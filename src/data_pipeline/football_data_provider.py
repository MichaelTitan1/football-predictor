from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .canonical_data import LeagueRecord, MatchRecord, TeamRecord
from .data_downloader import LEAGUE_CONFIG, is_football_data_unavailable, mark_football_data_unavailable
from .providers import ProviderSnapshot

logger = logging.getLogger(__name__)
BASE_URL = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
REQUEST_TIMEOUT = 30

class FootballDataProvider:
    name = "football-data.co.uk"
    def __init__(self, raw_dir: str | Path = "data/raw", include_remote: bool = True): self.raw_dir=Path(raw_dir); self.include_remote=include_remote
    @staticmethod
    def _season_code(year:int)->str: return f"{year%100:02d}{(year+1)%100:02d}"
    @staticmethod
    def _season_from_date(kickoff:str)->str:
        dt=datetime.fromisoformat(kickoff.replace("Z","+00:00")); start=dt.year if dt.month>=7 else dt.year-1; return f"{start}-{start+1}"
    @staticmethod
    def _team_key(name:str)->str:
        value=re.sub(r"[^a-z0-9]+","-",name.strip().lower()).strip("-")
        if not value: raise ValueError("empty team name")
        return value
    @staticmethod
    def _parse_datetime(row:pd.Series)->str:
        raw_date=row.get("Date"); raw_time=row.get("Time")
        if pd.isna(raw_date): raise ValueError("match has no date")
        text=str(raw_date).strip(); time_text="" if pd.isna(raw_time) else str(raw_time).strip()
        value = f"{text} {time_text}".strip()
        formats = ("%d/%m/%Y %H:%M", "%d/%m/%y %H:%M", "%d/%m/%Y", "%d/%m/%y")
        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.replace(tzinfo=ZoneInfo("Europe/London")).astimezone(timezone.utc).isoformat(timespec="seconds")
            except ValueError:
                continue
        raise ValueError(f"unsupported Football-Data date format: {value}")
    def _get(self,url:str):
        try:
            r=requests.get(url,timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                return None, 404
            r.raise_for_status(); return r.content, r.status_code
        except Exception as exc: logger.warning("Football-Data fetch failed: %s (%s)",url,exc); return None, None
    def _local_frames(self):
        frames=[]; code_to_key={v["code"]:k for k,v in LEAGUE_CONFIG.items()}
        if not self.raw_dir.exists(): return frames
        for path in sorted(self.raw_dir.glob("*.csv")):
            try:
                df=pd.read_csv(path); stem=path.stem.split("_")[0]
                if "League" not in df.columns:
                    if stem in LEAGUE_CONFIG: df["League"]=stem
                    elif stem in code_to_key: df["League"]=code_to_key[stem]
                frames.append(df)
            except Exception as exc: logger.warning("Skipping %s: %s",path,exc)
        return frames
    def _remote_frames(self):
        frames=[]; now=datetime.now(timezone.utc); season_year=now.year if now.month>=7 else now.year-1
        for league_key,info in LEAGUE_CONFIG.items():
            if is_football_data_unavailable(league_key):
                logger.info("Football-Data unavailable for %s; skipping remote current-season request", league_key)
                continue
            url=f"{BASE_URL}/{self._season_code(season_year)}/{info['code']}.csv"
            data,status=self._get(url)
            if status == 404:
                mark_football_data_unavailable(league_key, url, status)
                continue
            if data:
                try:
                    df=pd.read_csv(io.BytesIO(data)); df["League"]=league_key; frames.append(df)
                except Exception as exc: logger.warning("Could not parse current %s: %s",league_key,exc)
        data,_status=self._get(FIXTURES_URL)
        if data:
            try: frames.append(pd.read_csv(io.BytesIO(data)))
            except Exception as exc: logger.warning("Could not parse fixture feed: %s",exc)
        return frames
    def _normalize(self,frames):
        leagues={}; teams={}; matches={}; code_to_key={v["code"]:k for k,v in LEAGUE_CONFIG.items()}
        for raw in frames:
            if raw is None or raw.empty: continue
            df=raw.copy(); df.columns=[str(c).strip() for c in df.columns]; league_value=df.get("League",df.get("Div"))
            if league_value is None: continue
            for idx,row in df.iterrows():
                code=str(league_value.loc[idx]).strip() if idx in league_value.index else ""; lk=code if code in LEAGUE_CONFIG else code_to_key.get(code)
                home=str(row.get("HomeTeam","")).strip(); away=str(row.get("AwayTeam","")).strip()
                if not lk or not home or not away or home=="nan" or away=="nan": continue
                try: kickoff=self._parse_datetime(row)
                except Exception: continue
                info=LEAGUE_CONFIG[lk]; leagues[lk]=LeagueRecord(lk,info["name"])
                hk,ak=self._team_key(home),self._team_key(away); teams[(lk,hk)]=TeamRecord(hk,home,lk); teams[(lk,ak)]=TeamRecord(ak,away,lk)
                fthg=pd.to_numeric(pd.Series([row.get("FTHG")]),errors="coerce").iloc[0]; ftag=pd.to_numeric(pd.Series([row.get("FTAG")]),errors="coerce").iloc[0]; ftr=str(row.get("FTR",""))
                finished=pd.notna(fthg) and pd.notna(ftag) and ftr in {"H","D","A"}
                m=MatchRecord(self.name,lk,self._season_from_date(kickoff),kickoff,hk,ak,"finished" if finished else "scheduled",int(fthg) if pd.notna(fthg) else None,int(ftag) if pd.notna(ftag) else None,str(row.get("MatchID")) if pd.notna(row.get("MatchID")) else None)
                matches[m.match_key]=m
        return list(leagues.values()),list(teams.values()),list(matches.values())
    def fetch(self)->ProviderSnapshot:
        frames=self._local_frames(); frames.extend(self._remote_frames() if self.include_remote else [])
        if not frames: raise RuntimeError("no Football-Data data is available locally or remotely")
        leagues,teams,matches=self._normalize(frames)
        if not matches: raise RuntimeError("Football-Data returned no valid canonical matches")
        return ProviderSnapshot(tuple(leagues),tuple(teams),tuple(matches),datetime.now(timezone.utc).isoformat(timespec="seconds"),self.name)
