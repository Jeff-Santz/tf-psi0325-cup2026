"""
Evaluates the trained MLP on data/processed/test.csv (the actual
2026 World Cup matches): cross-entropy, accuracy (secondary metric),
and a calibration curve (predicted probability vs observed frequency).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from model import MatchOutcomeMLP

CLASS_NAMES = ["home_win", "draw", "away_win"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    checkpoint = torch.load("outputs/trained_model.pt", map_location=DEVICE)
    features = checkpoint["features"]
    mean = checkpoint["feature_mean"]
    std = checkpoint["feature_std"]

    model = MatchOutcomeMLP(input_dim=len(features)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df["result"].values.astype(np.int64)

    X_test = (X_test - mean) / std
    X_test_t = torch.from_numpy(X_test).to(DEVICE)
    y_test_t = torch.from_numpy(y_test).to(DEVICE)

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)
        ce_loss = nn.CrossEntropyLoss()(logits, y_test_t).item()
        preds = logits.argmax(dim=1)
        accuracy = (preds == y_test_t).float().mean().item()

    print(f"World Cup 2026 test set ({len(test_df)} matches)")
    print(f"Cross-entropy loss: {ce_loss:.4f}")
    print(f"Accuracy (secondary metric): {accuracy:.3f}")

    plot_calibration_curve(probs.cpu().numpy(), y_test)
    plot_confusion_matrix(y_test, preds.cpu().numpy())
    plot_roc_curves(probs.cpu().numpy(), y_test)


def plot_calibration_curve(probs, y_true, n_bins=10):
    """
    For each class, bins predicted probabilities and compares the
    average predicted probability in each bin to the observed
    frequency of that class among matches in the bin.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_probs = probs[:, class_idx]
        observed = (y_true == class_idx).astype(float)

        bin_ids = np.digitize(class_probs, bin_edges[1:-1])
        pred_means, obs_means = [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            pred_means.append(class_probs[mask].mean())
            obs_means.append(observed[mask].mean())

        ax.plot(pred_means, obs_means, marker="o", label=class_name)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve — 2026 World Cup")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figures/calibration_curve.png", dpi=150)
    print("Calibration curve saved to outputs/figures/calibration_curve.png")


def plot_confusion_matrix(y_true, y_pred):
    """
    3x3 confusion matrix on the World Cup test set. Useful to spot
    systematic bias (e.g. the model never predicting draws).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — 2026 World Cup")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("outputs/figures/confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to outputs/figures/confusion_matrix.png")


def plot_roc_curves(probs, y_true):
    """
    One-vs-rest ROC curve and AUC for each class. Useful to see which
    outcome (e.g. draw) is hardest for the model to discriminate.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="random guess")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        y_binary = (y_true == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (one-vs-rest) — 2026 World Cup")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/figures/roc_curves.png", dpi=150)
    print("ROC curves saved to outputs/figures/roc_curves.png")


if __name__ == "__main__":
    main()