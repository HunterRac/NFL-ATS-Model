"""Download nflverse data and cache it locally as parquet."""
import time
from datetime import date
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FIRST_SEASON = 2006  # nflverse play-by-play EPA is reliable from here on
PBP_COLS = ["game_id", "posteam", "defteam", "play_type", "epa", "success"]


def current_season():
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def _stale(path, max_age_hours):
    return not path.exists() or time.time() - path.stat().st_mtime > max_age_hours * 3600


def load_schedules(refresh=False):
    """Every game since 1999 with scores, closing spread, odds, rest days and starting QBs."""
    path = DATA_DIR / "schedules.parquet"
    if refresh or _stale(path, max_age_hours=12):
        DATA_DIR.mkdir(exist_ok=True)
        print("Downloading schedules...")
        nfl.load_schedules().to_pandas().to_parquet(path)
    return pd.read_parquet(path)


def _summarize_pbp(season):
    """Collapse one season of play-by-play into one row per team per game."""
    pbp = nfl.load_pbp([season]).select(PBP_COLS).to_pandas()
    plays = pbp[pbp.play_type.isin(["pass", "run"]) & pbp.epa.notna() & pbp.posteam.notna()]

    def summarize(team_col, prefix):
        keys = ["game_id", team_col]
        out = pd.DataFrame({
            f"{prefix}_epa": plays.groupby(keys).epa.mean(),
            f"{prefix}_success": plays.groupby(keys).success.mean(),
            f"{prefix}_pass_epa": plays[plays.play_type == "pass"].groupby(keys).epa.mean(),
            f"{prefix}_rush_epa": plays[plays.play_type == "run"].groupby(keys).epa.mean(),
        })
        return out.rename_axis(["game_id", "team"])

    return summarize("posteam", "off").join(summarize("defteam", "def")).reset_index()


def load_team_games(seasons, refresh=False):
    """Per-team, per-game offensive and defensive efficiency. The current season is always refreshed."""
    frames = []
    for season in seasons:
        path = DATA_DIR / f"team_games_{season}.parquet"
        if refresh or not path.exists() or (season == current_season() and _stale(path, 12)):
            DATA_DIR.mkdir(exist_ok=True)
            print(f"Downloading {season} play-by-play...")
            _summarize_pbp(season).to_parquet(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)
