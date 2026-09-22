"""Small MLP for (src,dst) edge ranking."""
from __future__ import annotations

import torch
import torch.nn as nn


class RankerMLP(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64) -> None:
        super().__init__()
        h2 = max(8, hid // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


__all__ = ["RankerMLP"]
