"""
Prepares the data for the PSI0325 project (MLP - 2026 World Cup prediction).

Reads results.csv (Kaggle dataset "International football results"),
calculates Elo and recent form sequentially over the ENTIRE history
(from 1872 onward, so it does not start flattened in 2000), then separates:

  - train: matches from 2000-01-01 until the day before the 2026 World Cup
  - test: the 104 matches of the FIFA World Cup 2026 (06/11 to 07/19/2026)

Output: train.csv and test.csv in ./data/, with columns:
  elo_diff, home_flag, form_diff, result (0=home win, 1=draw, 2=away win)
"""

import pandas as pd
from collections import defaultdict, deque

RAW_PATH = "data/raw/results.csv"
TRAIN_CUTOFF = "2026-06-11"  # start of the 2026 World Cup
K_FACTOR = 20
FORM_WINDOW = 10


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def result_label(home_score, away_score):
    if home_score > away_score:
        return 0  # home win
    if home_score == away_score:
        return 1  # draw
    return 2  # away win


def main():
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    elo = defaultdict(lambda: 1500.0)
    recent_results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # points per game

    rows = []
    for _, m in df.iterrows():
        home, away = m["home_team"], m["away_team"]

        # features calculated BEFORE processing the current match result
        elo_diff = elo[home] - elo[away]
        home_flag = 0 if m["neutral"] else 1
        form_home = sum(recent_results[home]) if recent_results[home] else 0
        form_away = sum(recent_results[away]) if recent_results[away] else 0
        form_diff = form_home - form_away

        label = result_label(m["home_score"], m["away_score"])

        if m["date"] >= pd.Timestamp("2000-01-01"):
            rows.append({
                "date": m["date"],
                "home_team": home,
                "away_team": away,
                "tournament": m["tournament"],
                "home_elo": elo[home],
                "away_elo": elo[away],
                "elo_diff": elo_diff,
                "home_flag": home_flag,
                "elo_home_interaction": elo_diff * home_flag,
                "form_diff": form_diff,
                "result": label,
            })

        # update Elo after the match
        exp_home = expected_score(elo[home], elo[away])
        score_home = 1.0 if label == 0 else (0.5 if label == 1 else 0.0)
        elo[home] += K_FACTOR * (score_home - exp_home)
        elo[away] += K_FACTOR * ((1 - score_home) - (1 - exp_home))

        # update recent form
        pts_home = 3 if label == 0 else (1 if label == 1 else 0)
        pts_away = 3 if label == 2 else (1 if label == 1 else 0)
        recent_results[home].append(pts_home)
        recent_results[away].append(pts_away)

    full = pd.DataFrame(rows)

    is_wc26 = (full["tournament"] == "FIFA World Cup") & (full["date"] >= TRAIN_CUTOFF)
    test = full[is_wc26].copy()
    train = full[(full["date"] < TRAIN_CUTOFF) & (~is_wc26)].copy()

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print(f"Training: {len(train)} matches ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test (2026 World Cup): {len(test)} matches")
    print(train["result"].value_counts(normalize=True).rename("training_proportion"))

    # save each team's most recent Elo and form, for use in predict.py
    latest_state = pd.DataFrame([
        {"team": team, "elo": rating, "recent_points": list(recent_results[team])}
        for team, rating in elo.items()
    ])
    latest_state.to_csv("data/processed/team_state.csv", index=False)
    print(f"Team state saved for {len(latest_state)} teams (data/processed/team_state.csv)")


if __name__ == "__main__":
    main()

