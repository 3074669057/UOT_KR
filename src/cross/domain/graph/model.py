"""Temporal candidate attention ranker (lightweight GAT-style)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalGraphRanker(nn.Module):
    def __init__(self, in_dim: int, hid: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hid)
        self.attn = nn.MultiheadAttention(embed_dim=hid, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        z, _ = self.attn(h, h, h, need_weights=False)
        return self.ff(z).squeeze(-1)


__all__ = ["TemporalGraphRanker"]
