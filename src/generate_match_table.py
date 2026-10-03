"""
Runs the trained MLP on all 104 matches of the 2026 World Cup test set
and builds a comparison table (predicted probabilities vs. actual result),
plus a "highlights" table with the most slide-worthy matches: the final,
the biggest upsets, and the most/least confident predictions.

Outputs:
    outputs/predictions_full.csv       — all 104 matches
    outputs/predictions_highlights.csv — a curated subset for slides
"""

import numpy as np
import pandas as pd
import torch

from model import MatchOutcomeMLP

CLASS_NAMES = ["home_win", "draw", "away_win"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matches to always include in the slide table, as (home_team, away_team)
# — edit this list to change which matches are highlighted.
SELECTED_MATCHES = [
    ("Brazil", "Japan"),      # group stage
    ("Brazil", "Morocco"),    # group stage — ended in a draw, the model's known weak spot
    ("Brazil", "Norway"),     # group stage
    ("France", "Spain"),      # semifinal
    ("France", "England"),    # semifinal, 4-6 — highest-scoring match of the tournament
    ("Spain", "Argentina"),   # final
]

def main():
    checkpoint = torch.load("outputs/trained_model.pt", map_location=DEVICE)
    features = checkpoint["features"]
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"]

    model = MatchOutcomeMLP(input_dim=len(features)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_df = pd.read_csv("data/processed/test.csv")
    X = test_df[features].values.astype(np.float32)
    X = (X - mean) / std

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    result = test_df.copy()
    result["p_home_win"] = probs[:, 0]
    result["p_draw"] = probs[:, 1]
    result["p_away_win"] = probs[:, 2]
    result["predicted"] = [CLASS_NAMES[p] for p in preds]
    result["actual"] = [CLASS_NAMES[r] for r in result["result"]]
    result["correct"] = result["predicted"] == result["actual"]
    # confidence the model had in whichever outcome actually happened
    result["prob_of_actual_outcome"] = probs[np.arange(len(probs)), result["result"].values]

    output_cols = [
        "date", "home_team", "away_team", "actual", "predicted", "correct",
        "p_home_win", "p_draw", "p_away_win", "prob_of_actual_outcome",
    ]
    full_table = result[output_cols].sort_values("date")
    full_table.to_csv("outputs/predictions_full.csv", index=False)
    print(f"Full table saved: outputs/predictions_full.csv ({len(full_table)} matches)")

    # ---- Selected matches for slides ----
    selected_rows = []
    for home, away in SELECTED_MATCHES:
        match = result[(result["home_team"] == home) & (result["away_team"] == away)]
        if match.empty:
            print(f"WARNING: {home} vs {away} not found in test set — skipping")
            continue
        selected_rows.append(match.iloc[0])

    selected_df = pd.DataFrame(selected_rows)[output_cols]
    selected_df.to_csv("outputs/predictions_selected.csv", index=False)
    print(f"Selected matches saved: outputs/predictions_selected.csv")
    print()
    print(selected_df.to_string(index=False))


if __name__ == "__main__":
    main()