"""
Predicts the outcome probabilities for a hypothetical match between
two teams, using each team's most recent Elo and form from
data/processed/team_state.csv.

CLI usage:
    python src/predict.py "Brazil" "Spain" --neutral
    python src/predict.py "Brazil" "Spain"   # Brazil as home team
"""

import argparse
import ast
import numpy as np
import pandas as pd
import torch

from model import MatchOutcomeMLP

CLASS_NAMES = ["home_win", "draw", "away_win"]


def get_team_state(state_df, team_name):
    row = state_df[state_df["team"] == team_name]
    if row.empty:
        raise ValueError(f"Team '{team_name}' not found in team_state.csv")
    elo = row.iloc[0]["elo"]
    recent_points = ast.literal_eval(row.iloc[0]["recent_points"])
    form = sum(recent_points) if recent_points else 0
    return elo, form


def predict_match(home_team, away_team, neutral=False,
                   state_path="data/processed/team_state.csv",
                   model_path="outputs/trained_model.pt"):
    """Returns (home_elo, away_elo, probs) where probs is a length-3
    numpy array in the order [home_win, draw, away_win]."""
    state_df = pd.read_csv(state_path)
    home_elo, home_form = get_team_state(state_df, home_team)
    away_elo, away_form = get_team_state(state_df, away_team)

    home_flag_value = 0 if neutral else 1
    elo_diff_value = home_elo - away_elo
    features = np.array([[
        elo_diff_value,
        home_flag_value,
        elo_diff_value * home_flag_value,
        home_form - away_form,
    ]], dtype=np.float32)

    checkpoint = torch.load(model_path)
    model = MatchOutcomeMLP(input_dim=len(checkpoint["features"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X = (features - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X.astype(np.float32))), dim=1)[0]

    return home_elo, away_elo, probs.numpy()


def predict_and_print(home_team, away_team, neutral=False):
    home_elo, away_elo, probs = predict_match(home_team, away_team, neutral=neutral)
    print(f"{home_team} (Elo {home_elo:.0f}) vs {away_team} (Elo {away_elo:.0f})")
    for name, p in zip(CLASS_NAMES, probs.tolist()):
        print(f"  {name:10s}: {p:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--neutral", action="store_true",
                         help="match played at a neutral venue (no home advantage)")
    args = parser.parse_args()
    predict_and_print(args.home_team, args.away_team, neutral=args.neutral)


if __name__ == "__main__":
    main()