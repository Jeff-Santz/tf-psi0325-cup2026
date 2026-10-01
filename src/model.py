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
    def __init__(self, input_dim=3, hidden_dim=16, num_classes=3, dropout=0.2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_classes)

        self._init_weights()
        # NOTE: no Softmax layer here on purpose — nn.CrossEntropyLoss
        # expects raw logits and applies log-softmax internally.

    def _init_weights(self):
        # Kaiming (He) initialization: recommended for layers followed
        # by ReLU, since it accounts for ReLU zeroing out ~half the
        # activations and keeps activation variance stable across layers.
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="relu")
        nn.init.zeros_(self.hidden.bias)
        # Output layer feeds into softmax, not ReLU, so a smaller-variance
        # Xavier/Glorot initialization is more appropriate here.
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.output(x)
