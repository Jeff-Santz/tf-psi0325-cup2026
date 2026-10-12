"""
Test-set metrics (2026 World Cup) for the linear baseline and the MLP:
cross-entropy, accuracy, per-class AUC and confusion matrix.

Run from the project root:  python src/linear_metrics.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score

from model import MatchOutcomeMLP
from model_linear import LinearBaseline

CLASS_NAMES = ["home_win", "draw", "away_win"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(checkpoint_path, model_class, test_df):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    features = checkpoint["features"]

    model = model_class(input_dim=len(features)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    X = test_df[features].values.astype(np.float32)
    X = (X - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    y = test_df["result"].values.astype(np.int64)

    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        ce = nn.CrossEntropyLoss()(logits, torch.from_numpy(y).to(DEVICE)).item()

    preds = probs.argmax(axis=1)
    accuracy = (preds == y).mean()
    aucs = [roc_auc_score((y == k).astype(int), probs[:, k]) for k in range(3)]
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    return ce, accuracy, aucs, cm


def main():
    test_df = pd.read_csv("data/processed/test.csv")

    results = {
        "Linear": evaluate("outputs/trained_model_linear.pt", LinearBaseline, test_df),
        "MLP": evaluate("outputs/trained_model.pt", MatchOutcomeMLP, test_df),
    }

    print(f"Test set: {len(test_df)} matches\n")
    print(f"{'Metric':<22}{'Linear':>10}{'MLP':>10}")
    lin, mlp = results["Linear"], results["MLP"]
    print(f"{'Cross-entropy':<22}{lin[0]:>10.4f}{mlp[0]:>10.4f}")
    print(f"{'Accuracy':<22}{lin[1]:>10.3f}{mlp[1]:>10.3f}")
    for k, name in enumerate(CLASS_NAMES):
        print(f"{'AUC ' + name:<22}{lin[2][k]:>10.3f}{mlp[2][k]:>10.3f}")

    for label, (_, _, _, cm) in results.items():
        print(f"\nConfusion matrix, {label} (rows = true, cols = predicted):")
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))


if __name__ == "__main__":
    main()