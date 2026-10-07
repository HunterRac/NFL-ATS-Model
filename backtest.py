"""Walk-forward backtest of against-the-spread models.

For each test season S, every model is trained only on seasons before S, then
used to pick every game in S. That mimics what you could actually have done in
real time. Run:  python backtest.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import FIRST_SEASON, current_season, load_schedules, load_team_games
from features import FEATURES, build_dataset

OUT_DIR = Path(__file__).parent / "outputs"
BREAKEVEN = 110 / 210  # win rate needed at -110 odds (52.38%)
EDGES = [0.0, 0.02, 0.04, 0.06]  # bet only when |p - 0.5| exceeds this


def logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=1000))


def gboost():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                          min_samples_leaf=50, l2_regularization=1.0)


# "spread_only" is the sanity check: the spread alone should not beat the spread.
MODELS = {
    "spread_only": (["spread_line"], logistic),
    "logistic": (FEATURES, logistic),
    "gboost": (FEATURES, gboost),
}


def walk_forward(games, first_test, models=MODELS):
    played = games[games.home_cover.notna() & (games.season > FIRST_SEASON)]
    out = []
    for season in range(first_test, current_season() + 1):
        train, test = played[played.season < season], games[games.season == season].copy()
        for name, (cols, make) in models.items():
            test[f"p_{name}"] = make().fit(train[cols], train.home_cover).predict_proba(test[cols])[:, 1]
        out.append(test)
    return pd.concat(out, ignore_index=True)


def american_payout(odds):
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, 100 / -odds, odds / 100)


def place_bets(df, p, edge):
    """Units won/lost per game (1 unit risked per bet); NaN where no bet or push."""
    bet_home, bet_away = p > 0.5 + edge, p < 0.5 - edge
    won = np.where(bet_home, df.home_cover == 1, df.home_cover == 0)
    odds = np.where(bet_home, df.home_spread_odds, df.away_spread_odds)
    profit = np.where(won, american_payout(odds), -1.0)
    return pd.Series(np.where((bet_home | bet_away) & df.home_cover.notna(), profit, np.nan), index=df.index)


def report(preds, models=MODELS):
    settled = preds[preds.home_cover.notna()]
    print(f"\nTest games: {len(settled)} ({settled.season.min()}-{settled.season.max()}), "
          f"break-even win rate {BREAKEVEN:.2%}")
    print(f"Coin-flip log loss: {np.log(2):.4f}  (lower is better; beating this at all is hard)\n")

    rows = []
    for name in models:
        p = settled[f"p_{name}"]
        for edge in EDGES:
            profit = place_bets(settled, p, edge).dropna()
            wins = int((profit > 0).sum())
            rows.append({
                "model": name, "edge": edge, "bets": len(profit),
                "win_rate": wins / max(len(profit), 1),
                "units": profit.sum(), "roi": profit.sum() / max(len(profit), 1),
                # Probability of doing this well by luck if the true win rate were break-even.
                "p_value": binomtest(wins, len(profit), BREAKEVEN, alternative="greater").pvalue
                if len(profit) else np.nan,
                "log_loss": log_loss(settled.home_cover, p),
                "brier": brier_score_loss(settled.home_cover, p),
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nPer season at edge 0.02 (units):")
    by_season = pd.DataFrame({
        name: place_bets(settled, settled[f"p_{name}"], 0.02).groupby(settled.season).sum()
        for name in models})
    print(by_season.round(1).to_string())
    return summary


def plot(preds, models=MODELS):
    settled = preds[preds.home_cover.notna()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in models:
        p = settled[f"p_{name}"]
        ax1.plot(settled.gameday, place_bets(settled, p, 0.02).fillna(0).cumsum(), label=name)
        frac, mean_p = calibration_curve(settled.home_cover, p, n_bins=10, strategy="quantile")
        ax2.plot(mean_p, frac, marker="o", label=name)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set(title="Cumulative units (edge 0.02, 1 unit per bet)", ylabel="units")
    ax2.plot([0.3, 0.7], [0.3, 0.7], "--", color="gray")
    ax2.set(title="Calibration", xlabel="predicted P(home covers)", ylabel="actual cover rate")
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "backtest.png", dpi=120)
    return fig


def upcoming_picks(games, models=MODELS):
    """Fit on every completed game, then score games that haven't been played yet."""
    played = games[games.home_cover.notna() & (games.season > FIRST_SEASON)]
    upcoming = games[games.result.isna() & (games.season == current_season())].copy()
    if upcoming.empty:
        return
    for name, (cols, make) in models.items():
        upcoming[f"p_{name}"] = make().fit(played[cols], played.home_cover).predict_proba(upcoming[cols])[:, 1]
    cols = ["gameday", "week", "away_team", "home_team", "spread_line"] + [f"p_{n}" for n in models]
    upcoming[cols].to_csv(OUT_DIR / "upcoming_picks.csv", index=False)
    next_week = upcoming[upcoming.week == upcoming.week.min()]
    print(f"\nWeek {next_week.week.iloc[0]} model probabilities that the HOME team covers "
          "(spread_line > 0 means home favored):")
    print(next_week[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test", type=int, default=2012, help="first season to backtest")
    parser.add_argument("--refresh", action="store_true", help="re-download all data")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    sched = load_schedules(args.refresh)
    team_games = load_team_games(range(FIRST_SEASON, current_season() + 1), args.refresh)
    games = build_dataset(sched[sched.season >= FIRST_SEASON], team_games)

    preds = walk_forward(games, args.first_test)
    preds.to_csv(OUT_DIR / "backtest_predictions.csv", index=False)
    report(preds).to_csv(OUT_DIR / "backtest_summary.csv", index=False)
    plot(preds)
    upcoming_picks(games)
    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
