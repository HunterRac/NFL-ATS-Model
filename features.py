"""Turn raw schedules + team efficiency into one row per game of pre-kickoff features.

Every feature must be knowable before kickoff: rolling stats are shifted by one
game so a game's own result never leaks into its features.
"""
import numpy as np
import pandas as pd

HALFLIFE_GAMES = 6
ATS_FORM_WINDOW = 8
RATING_COLS = [
    "off_epa", "off_success", "off_pass_epa", "off_rush_epa",
    "def_epa", "def_success", "def_pass_epa", "def_rush_epa",
]
# Keep relocated franchises continuous so their rolling ratings don't reset.
RELOCATIONS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

FEATURES = (
    ["spread_line", "total_line", "rest_diff", "div_game", "dome", "wind",
     "playoff", "neutral", "week", "net_epa_diff", "ats_form_diff",
     "home_qb_change", "away_qb_change"]
    + [f"{c}_diff" for c in RATING_COLS]
)


def _team_long(sched):
    """One row per team per game, from that team's point of view."""
    base = ["game_id", "season", "gameday", "result", "spread_line"]
    sides = []
    for side, sign in (("home", 1), ("away", -1)):
        df = sched[base + [f"{side}_team", f"{side}_qb_id"]].rename(
            columns={f"{side}_team": "team", f"{side}_qb_id": "qb_id"})
        df["is_home"] = int(side == "home")
        # Points by which this team beat the spread (positive = covered).
        df["cover_margin"] = sign * (df.result - df.spread_line)
        sides.append(df)
    long = pd.concat(sides, ignore_index=True)
    long["team"] = long.team.replace(RELOCATIONS)
    return long.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def build_dataset(sched, team_games):
    sched = sched[sched.spread_line.notna()].copy()
    sched["gameday"] = pd.to_datetime(sched.gameday)

    team_games = team_games.assign(team=team_games.team.replace(RELOCATIONS))
    long = _team_long(sched).merge(team_games, on=["game_id", "team"], how="left")
    g = long.groupby("team")

    # Exponentially weighted efficiency from *previous* games only.
    for c in RATING_COLS:
        long[f"{c}_pre"] = g[c].transform(
            lambda s: s.shift(1).ewm(halflife=HALFLIFE_GAMES, ignore_na=True).mean())
    long["ats_form"] = g.cover_margin.transform(
        lambda s: s.shift(1).rolling(ATS_FORM_WINDOW, min_periods=3).mean())
    prev_qb = g.qb_id.shift(1)
    long["qb_change"] = (long.qb_id.notna() & prev_qb.notna() & (long.qb_id != prev_qb)).astype(int)

    team_feats = [f"{c}_pre" for c in RATING_COLS] + ["ats_form", "qb_change"]
    home = long[long.is_home == 1].set_index("game_id")[team_feats].add_prefix("home_")
    away = long[long.is_home == 0].set_index("game_id")[team_feats].add_prefix("away_")
    games = sched.set_index("game_id").join(home).join(away).reset_index()

    for c in RATING_COLS:
        games[f"{c}_diff"] = games[f"home_{c}_pre"] - games[f"away_{c}_pre"]
    games["net_epa_diff"] = games.off_epa_diff - games.def_epa_diff
    games["ats_form_diff"] = games.home_ats_form - games.away_ats_form
    games["rest_diff"] = games.home_rest - games.away_rest
    games["dome"] = games.roof.isin(["dome", "closed"]).astype(int)
    games["wind"] = games.wind.fillna(0)
    games["playoff"] = (games.game_type != "REG").astype(int)
    games["neutral"] = (games.location == "Neutral").astype(int)

    # Target: did the home team cover? Pushes are NaN (bets refunded), as are unplayed games.
    games["home_cover"] = np.select(
        [games.result > games.spread_line, games.result < games.spread_line], [1.0, 0.0], np.nan)
    games.loc[games.result.isna(), "home_cover"] = np.nan
    for side in ("home", "away"):
        col = f"{side}_spread_odds"
        games[col] = games[col].fillna(-110) if col in games else -110
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
