"""
MLP model definition for the 2026 World Cup outcome prediction task.

Inputs (3 features):
  - elo_diff: Elo rating difference (home - away)
  - home_flag: home advantage indicator (0 if neutral venue, 1 otherwise)
  - form_diff: recent form difference (points in last 10 games, home - away)

Output: 3-class softmax (0 = home win, 1 = draw, 2 = away win)
"""

import torch.nn as nn


class MatchOutcomeMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        # NOTE: no Softmax layer here on purpose — nn.CrossEntropyLoss
        # expects raw logits and applies log-softmax internally.
        # Apply softmax manually only at inference time (see evaluate.py).

    def forward(self, x):
        return self.net(x)
