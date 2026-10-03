"""
Trains the MLP on data/processed/train.csv and saves the model
(plus feature normalization stats) to outputs/.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from model import MatchOutcomeMLP

FEATURES = ["elo_diff", "home_flag", "elo_home_interaction", "form_diff"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET = "result"

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1  # held out from train.csv, NOT the World Cup test set
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv("data/processed/train.csv")

    # chronological split: last VAL_FRACTION of train.csv (already sorted by date)
    # used only to monitor overfitting during training, never for model selection
    # against the World Cup itself.
    split_idx = int(len(df) * (1 - VAL_FRACTION))
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train_df[FEATURES].values.astype(np.float32)
    y_train = train_df[TARGET].values.astype(np.int64)
    X_val = val_df[FEATURES].values.astype(np.float32)
    y_val = val_df[TARGET].values.astype(np.int64)

    # standardize features using TRAIN statistics only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    model = MatchOutcomeMLP(input_dim=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "features": FEATURES,
    }, "outputs/trained_model.pt")
    print("Model saved to outputs/trained_model.pt")

    plot_learning_curve(train_loss_history, val_loss_history)


def plot_learning_curve(train_loss_history, val_loss_history):
    """
    Cross-entropy loss per epoch for train and validation sets.
    Used to check convergence and spot overfitting (val_loss rising
    while train_loss keeps falling).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_loss_history, label="train loss", color="tab:blue")
    ax.plot(val_loss_history, label="validation loss", color="tab:red")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Learning curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("outputs/figures/learning_curve.png", dpi=150)
    print("Learning curve saved to outputs/figures/learning_curve.png")


if __name__ == "__main__":
    main()