"""
Sensitivity analysis (one factor at a time) for the linear baseline and the MLP.
Each configuration is trained with several seeds; test metrics are reported as
mean +- std. Exploratory only: results are NOT used to select hyperparameters.

Run from the project root:  python src/sensitivity.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import MatchOutcomeMLP
from model_linear import LinearBaseline

FEATURES = ["elo_diff", "home_flag", "elo_home_interaction", "form_diff"]
VAL_FRACTION = 0.1
WEIGHT_DECAY = 1e-4
SEEDS = [42, 1, 2, 3, 4]
DEVICE = torch.device("cpu")  # tiny models: CPU is usually faster than GPU here

BASE = dict(epochs=100, batch=64, lr=1e-3, init="paper")
CONFIGS = (
    [BASE]
    + [dict(BASE, epochs=e) for e in (50, 200, 400)]
    + [dict(BASE, batch=b) for b in (32, 128, 256)]
    + [dict(BASE, lr=l) for l in (1e-2, 1e-4, 1e-5)]
    + [dict(BASE, init="default")]  # MLP only: PyTorch default init
)


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    split = int(len(train) * (1 - VAL_FRACTION))  # same chronological split as train.py
    train = train.iloc[:split]

    X_tr = train[FEATURES].values.astype(np.float32)
    mean, std = X_tr.mean(axis=0), X_tr.std(axis=0)
    X_te = test[FEATURES].values.astype(np.float32)

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE)
    return (to_t((X_tr - mean) / std), to_t(train["result"].values.astype(np.int64)),
            to_t((X_te - mean) / std), to_t(test["result"].values.astype(np.int64)))


def run(model_class, cfg, seed, data):
    X_tr, y_tr, X_te, y_te = data
    torch.manual_seed(seed)
    model = model_class(input_dim=len(FEATURES)).to(DEVICE)
    if cfg["init"] == "default":
        model.hidden.reset_parameters()
        model.output.reset_parameters()

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    n = X_tr.shape[0]

    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            crit(model(X_tr[idx]), y_tr[idx]).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        ce = crit(logits, y_te).item()
        acc = (logits.argmax(dim=1) == y_te).float().mean().item()
    return ce, acc


def label(cfg):
    changed = [f"{k}={cfg[k]}" for k in ("epochs", "batch", "lr", "init") if cfg[k] != BASE[k]]
    return "base (100 ep, batch 64, lr 1e-3)" if not changed else ", ".join(changed)


def main():
    data = load_data()
    rows = []
    for cfg in CONFIGS:
        for name, cls in (("Linear", LinearBaseline), ("MLP", MatchOutcomeMLP)):
            if cfg["init"] == "default" and name == "Linear":
                continue  # the baseline already uses the PyTorch default init
            res = np.array([run(cls, cfg, s, data) for s in SEEDS])
            rows.append({
                "config": label(cfg), "model": name,
                "ce_mean": res[:, 0].mean(), "ce_std": res[:, 0].std(),
                "acc_mean": res[:, 1].mean(), "acc_std": res[:, 1].std(),
            })
            print(f"{label(cfg):<42}{name:<8}CE {res[:,0].mean():.4f} +- {res[:,0].std():.4f}"
                  f"   acc {res[:,1].mean():.3f} +- {res[:,1].std():.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/sensitivity.csv", index=False)
    print("\nSaved: outputs/sensitivity.csv")


if __name__ == "__main__":
    main()