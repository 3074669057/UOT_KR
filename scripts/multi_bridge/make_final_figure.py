"""Final paper figure: three-bridge split/merge structural recovery, RC-UOT-Q vs one-to-one baselines.

Nature-style: muted colors, panel labels, mean ± std (5 seeds) error bars, editable
source data saved alongside (fig5b_three_bridge_structural_recovery_data.csv,
fig5b_three_bridge_structural_recovery.{png,pdf,svg}).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

bridges = ["Celer", "Multichain", "PolyNetwork"]
methods = ["RC-UOT-Q", "Connector-style", "ABCTracer-style"]

# mean ± std over 5 seeds (from independent verification)
data = {
    "split": {
        "RC-UOT-Q": {"mean": [0.950, 0.971, 1.000], "std": [0.0316, 0.0349, 0.0]},
        "Connector-style": {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        "ABCTracer-style": {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
    },
    "merge": {
        "RC-UOT-Q": {"mean": [0.967, 0.979, 1.000], "std": [0.0432, 0.0255, 0.0]},
        "Connector-style": {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        "ABCTracer-style": {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
    },
}

# source data CSV
rows = []
for metric in ("split", "merge"):
    for m in methods:
        for i, b in enumerate(bridges):
            rows.append({"metric": f"{metric}_recovery", "bridge": b, "method": m,
                         "mean": data[metric][m]["mean"][i], "std": data[metric][m]["std"][i],
                         "n_seeds": 5})
pd.DataFrame(rows).to_csv(OUT / "fig5b_three_bridge_structural_recovery_data.csv", index=False)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

colors = {"RC-UOT-Q": "#2C7BB6", "Connector-style": "#999999", "ABCTracer-style": "#B8B8B8"}
hatches = {"RC-UOT-Q": "", "Connector-style": "//", "ABCTracer-style": ".."}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
x = np.arange(len(bridges))
width = 0.26

for ax, metric, panel in ((axes[0], "split", "a"), (axes[1], "merge", "b")):
    for k, m in enumerate(methods):
        means = data[metric][m]["mean"]
        stds = data[metric][m]["std"]
        off = (k - 1) * (width + 0.02)
        bars = ax.bar(x + off, means, width, yerr=stds, capsize=2.5, error_kw={"elinewidth": 0.8, "capthick": 0.8},
                      color=colors[m], edgecolor="#222222", linewidth=0.5, hatch=hatches[m], label=m)
        for i, v in enumerate(means):
            if v > 0:
                ax.text(x[i] + off, v + max(stds[i], 0.03), f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(bridges, fontsize=8)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(-0.42, 1.16, panel, fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.set_title(metric.capitalize() + " recovery", fontsize=8.5, color="#222222")

axes[0].set_ylabel("Structural recovery (mean ± s.d., 5 seeds)", fontsize=8)
axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.06), ncol=3)

fig.tight_layout(pad=0.6)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"fig5b_three_bridge_structural_recovery.{ext}", dpi=300, bbox_inches="tight")
print("figure written to", OUT)
