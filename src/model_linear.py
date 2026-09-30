"""
Linear baseline for comparison against MatchOutcomeMLP.

This is a multinomial logistic regression expressed as a single
linear layer feeding directly into softmax (via CrossEntropyLoss).
No hidden layer, no non-linear activation.
"""

import torch.nn as nn


class LinearBaseline(nn.Module):
    def __init__(self, input_dim=3, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        # No hidden layer, no activation function: this model can only
        # learn linear decision boundaries between the 3 classes.

    def forward(self, x):
        return self.linear(x)