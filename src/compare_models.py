"""
Loads both trained models (linear baseline and MLP) and compares
their cross-entropy and accuracy on the 2026 World Cup test set.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import MatchOutcomeMLP
from model_linear import LinearBaseline


def load_and_eval(checkpoint_path, model_class, X_test, y_test_t):
    checkpoint = torch.load(checkpoint_path)
    model = model_class(input_dim=len(checkpoint["features"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X = (X_test - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        ce_loss = nn.CrossEntropyLoss()(logits, y_test_t).item()
        acc = (logits.argmax(dim=1) == y_test_t).float().mean().item()
    return ce_loss, acc


def main():
    test_df = pd.read_csv("data/processed/test.csv")
    features = ["elo_diff", "home_flag", "elo_home_interaction", "form_diff"]
    X_test = test_df[features].values.astype(np.float32)
    y_test_t = torch.from_numpy(test_df["result"].values.astype(np.int64))

    linear_ce, linear_acc = load_and_eval(
        "outputs/trained_model_linear.pt", LinearBaseline, X_test, y_test_t)
    mlp_ce, mlp_acc = load_and_eval(
        "outputs/trained_model.pt", MatchOutcomeMLP, X_test, y_test_t)

    print(f"{'Model':<20}{'Cross-entropy':<16}{'Accuracy':<10}")
    print(f"{'Linear baseline':<20}{linear_ce:<16.4f}{linear_acc:<10.3f}")
    print(f"{'MLP':<20}{mlp_ce:<16.4f}{mlp_acc:<10.3f}")


if __name__ == "__main__":
    main()