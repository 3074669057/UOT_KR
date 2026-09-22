"""S9 post-hoc figures (PDF + 300 dpi PNG, all labels English)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
S9 = R7 / "posthoc_s9_degree_stratification_20260918"
RESULTS = S9 / "results"
FIGURES = S9 / "figures"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans"})

C_CEIL = "#1f77b4"
C_H1 = "#d62728"


def _load() -> dict[str, Any]:
    return json.loads((RESULTS / "s9_analysis.json").read_text(encoding="utf-8"))


def _save(fig, name: str) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf, png = FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


# --------------------------------------------------------------------------- #
# Figure 1 -- dual axis
# --------------------------------------------------------------------------- #

def figure_dual_axis(ana: dict[str, Any]) -> list[str]:
    ex = sorted(ana["exact_degree"], key=lambda r: r["d_max"])
    ce = {r["d_max"]: r for r in ana["oracle_ceiling_exact"]}
    d = np.array([r["d_max"] for r in ex], dtype=float)
    ceil = np.array([ce[r["d_max"]]["mean_ceiling"] for r in ex], dtype=float)
    ceil_lo = np.array([ce[r["d_max"]]["boot_ci_lower"] for r in ex], dtype=float)
    ceil_hi = np.array([ce[r["d_max"]]["boot_ci_upper"] for r in ex], dtype=float)
    h1 = np.array([r["effect"] for r in ex], dtype=float)
    h1_lo = np.array([r["bootstrap"]["ci_lower"] for r in ex], dtype=float)
    h1_hi = np.array([r["bootstrap"]["ci_upper"] for r in ex], dtype=float)
    n = [r["n_templates"] for r in ex]

    fig, ax1 = plt.subplots(figsize=(8.6, 4.6))
    ax2 = ax1.twinx()

    ax1.errorbar(d, ceil, yerr=[ceil - ceil_lo, ceil_hi - ceil],
                 color=C_CEIL, marker="o", ms=5, lw=1.8, capsize=3,
                 label="Oracle 1-to-1 semantic ceiling (label-informed)")
    ax1.set_xlabel("Maximum source fan-out degree  $d_{max}$")
    ax1.set_ylabel("Oracle 1-to-1 semantic ceiling (edge F1)", color=C_CEIL)
    ax1.tick_params(axis="y", labelcolor=C_CEIL)

    ax2.errorbar(d, h1, yerr=[h1 - h1_lo, h1_hi - h1],
                 color=C_H1, marker="s", ms=5, lw=1.8, ls="--", capsize=3,
                 label="UOT-KR - Hungarian edge F1 (paired, post-hoc)")
    ax2.axhline(0.0, color="0.45", lw=1.0, ls=":")
    ax2.set_ylabel("UOT-KR $-$ Hungarian edge F1", color=C_H1)
    ax2.tick_params(axis="y", labelcolor=C_H1)

    # data-driven, independent ranges (no artificial mirroring/crossing)
    ax1.set_ylim(min(ceil_lo.min(), ceil.min()) - 0.05,
                 max(ceil_hi.max(), ceil.max()) + 0.05)
    ax2.set_ylim(min(h1_lo.min(), h1.min()) - 0.04,
                 max(h1_hi.max(), h1.max()) + 0.04)
    ax1.set_xticks(d)

    for xi, ni in zip(d, n):
        ax1.annotate(f"n={ni}", (xi, ax1.get_ylim()[0]),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=6.5, color="0.35")

    h1l, l1 = ax1.get_legend_handles_labels()
    h2l, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1l + h2l, l1 + l2, loc="lower left", fontsize=8, framealpha=0.95)

    fig.suptitle("S9 (post-hoc, not preregistered) - truth-degree stratified "
                 "one-to-one ceiling and paired H1 effect", fontsize=10.5)
    fig.text(0.5, -0.02,
             "The two axes use different numeric scales; the curves are juxtaposed, not "
             "subtractable.\nPost-hoc re-stratification of archived confirmatory results "
             "(block 411-420); no prediction method was re-executed.",
             ha="center", fontsize=7, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _save(fig, "s9_degree_stratified_h1_and_oracle")


# --------------------------------------------------------------------------- #
# Figure 2 -- strata effect estimates
# --------------------------------------------------------------------------- #

def figure_strata(ana: dict[str, Any]) -> list[str]:
    bins = [(r["stratum"], r["n_templates"], r["effect"],
             r["bootstrap"]["ci_lower"], r["bootstrap"]["ci_upper"])
            for r in ana["binary"]]
    tri = [(r["stratum"], r["n_templates"], r["effect"],
            r["bootstrap"]["ci_lower"], r["bootstrap"]["ci_upper"])
           for r in ana["three_bin"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    for ax, data, title, colors in (
            (axes[0], bins, "(a) Binary strata (primary S9)", ["#7f7f7f", "#d62728"]),
            (axes[1], tri, "(b) Three-bin strata", ["#7f7f7f", "#ff7f0e", "#8c564b"])):
        y = np.arange(len(data))
        eff = [r[2] for r in data]
        lo = [r[2] - r[3] for r in data]
        hi = [r[4] - r[2] for r in data]
        ax.barh(y, eff, color=colors, height=0.55)
        ax.errorbar(eff, y, xerr=[lo, hi], fmt="none", ecolor="k", capsize=4, lw=1.3)
        ax.axvline(0, color="0.4", lw=1.1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r[0]}\nn={r[1]}" for r in data], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("UOT-KR $-$ Hungarian edge F1")
        ax.set_title(title, fontsize=9.5)
        for yi, r in zip(y, data):
            ax.annotate(f"{r[2]:+.4f}", (r[2], yi), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=7.5)
    fig.suptitle("S9 (post-hoc, not preregistered) - H1 effect by truth fan-out degree "
                 "stratum", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "s9_binary_threebin_h1")


def build_all() -> dict[str, Any]:
    ana = _load()
    out = {
        "s9_degree_stratified_h1_and_oracle": figure_dual_axis(ana),
        "s9_binary_threebin_h1": figure_strata(ana),
    }
    (FIGURES / "figures_index.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
