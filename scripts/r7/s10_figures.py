"""S10 post-hoc figures (PDF + 300 dpi PNG, all labels English)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R7 = Path(__file__).resolve().parents[2] / "out" / "r7_confirmatory_kernel_ranking_20260917"
S10 = R7 / "posthoc_s10_unmatched_mass_localization_20260919"
RESULTS = S10 / "results"
FIGURES = S10 / "figures"
BRIDGES = ("Celer", "Multi", "Poly")

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans"})

C_U = "#d62728"
C_M = "#1f77b4"
C_BASE = "#7f7f7f"


def _save(fig, name: str) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf, png = FIGURES / f"{name}.pdf", FIGURES / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


def _stratified_roc(src: pd.DataFrame, tpl: pd.DataFrame, n_grid: int = 400
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Template-stratified ROC: thresholds on the within-template delta_s share.

    Each template contributes one TPR step and one FPR step; templates are equally
    weighted (bridge-balanced), which is exactly the definition of the stratified AUC.
    """
    vals = src["delta_s_share"].dropna().to_numpy(float)
    if vals.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    grid = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, n_grid)))
    grid = np.concatenate([[np.inf], grid[::-1], [-np.inf]])
    per_bridge: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {b: [] for b in BRIDGES}
    for (b, _s, _t), g in src.groupby(["bridge", "seed", "template_id"], sort=False):
        u = g.loc[g["truth_unmatched"] == 1, "delta_s_share"].to_numpy(float)
        m = g.loc[g["truth_unmatched"] == 0, "delta_s_share"].to_numpy(float)
        m = m[np.isfinite(m)]
        if u.size != 1 or not np.isfinite(u[0]) or m.size == 0:
            continue
        tp = (u[0] >= grid).astype(float)
        fp = np.array([(m >= t).mean() for t in grid])
        per_bridge[b].append((fp, tp))
    bridge_curves = []
    for b in BRIDGES:
        if per_bridge[b]:
            bridge_curves.append((np.mean([x[0] for x in per_bridge[b]], axis=0),
                                  np.mean([x[1] for x in per_bridge[b]], axis=0)))
    if not bridge_curves:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr = np.mean([c[0] for c in bridge_curves], axis=0)
    tpr = np.mean([c[1] for c in bridge_curves], axis=0)
    order = np.argsort(fpr, kind="stable")
    fpr, tpr = fpr[order], tpr[order]
    keep = np.concatenate([[True], np.diff(fpr) > 0])
    return fpr[keep], tpr[keep]


def figure_source(ana: dict[str, Any], src: pd.DataFrame, tpl: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

    # (a) delta_S share ECDF
    ax = axes[0]
    u = src.loc[src["truth_unmatched"] == 1, "delta_s_share"].dropna().to_numpy(float)
    m = src.loc[src["truth_unmatched"] == 0, "delta_s_share"].dropna().to_numpy(float)
    for v, lab, c in ((m, f"matched sources (n={m.size})", C_M),
                      (u, f"true unmatched source (n={u.size})", C_U)):
        x = np.sort(v)
        ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=c, lw=1.8,
                label=lab)
    ax.set_xlabel(r"$\delta^S$ share within template")
    ax.set_ylabel("ECDF")
    ax.set_title("(a) Source unmatched-mass share", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")

    # (b) template-stratified ROC
    ax = axes[1]
    fpr, tpr = _stratified_roc(src, tpl)
    ax.plot(fpr, tpr, color=C_U, lw=2.0, label="template-stratified ROC")
    ax.plot([0, 1], [0, 1], color=C_BASE, ls="--", lw=1.0, label="chance")
    a = ana["source_side"]["template_stratified_auc"]
    ax.set_xlabel("False positive rate (matched sources)")
    ax.set_ylabel("True positive rate (unmatched source)")
    ax.set_title("(b) Discrimination (not calibration)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.text(0.04, 0.96,
            f"stratified AUC = {a['effect']:.4f}\n95% CI [{a['ci_lower']:.4f}, "
            f"{a['ci_upper']:.4f}]\npooled AUC (diagnostic) = "
            f"{ana['source_side']['pooled_source_auc_DIAGNOSTIC_ONLY']:.4f}\n"
            f"marginal-only null AUC = "
            f"{ana['marginal_null_diagnostic']['source_auc_amount_only_null_bridge_balanced']:.4f}",
            transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(fc="white", ec="0.7", alpha=0.92))

    # (c) Top-1 observed vs chance
    ax = axes[2]
    labels = ["Overall"] + list(BRIDGES)
    obs = [ana["source_side"]["top1_hit"]] + \
        [ana["per_bridge"][b]["source_top1_hit"] for b in BRIDGES]
    ch = [ana["source_side"]["chance_baseline"]] + \
        [ana["per_bridge"][b]["chance_top1"] for b in BRIDGES]
    tie = [ana["source_side"]["tie_aware_top1"]] + \
        [ana["per_bridge"][b]["source_tie_aware_top1"] for b in BRIDGES]
    xs = np.arange(len(labels))
    w = 0.27
    ax.bar(xs - w, obs, w, color=C_U, label="Top-1 hit (deterministic rule)")
    ax.bar(xs, ch, w, color=C_BASE, label="size-adjusted chance (1/n)")
    ax.bar(xs + w, tie, w, color="#2ca02c", label="tie-aware Top-1 (diagnostic)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Source Top-1 localization", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="upper right")
    ax.text(0.02, 0.88, f"permutation p = {ana['source_side']['permutation_p']:.3g}\n"
                        f"(one-sided, n_perm={ana['source_side']['n_perm']})",
            transform=ax.transAxes, fontsize=7, va="top")

    fig.suptitle("S10 (post-hoc, not preregistered) - source-side unmatched-mass "
                 "localization", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "s10_source_unmatched_localization")


def figure_target(ana: dict[str, Any], tpl: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axes[0]
    labels = ["Overall"] + list(BRIDGES)
    obs = [ana["target_side"]["decoy_combined_delta_share"]["mean"]] + \
        [ana["per_bridge"][b]["decoy_combined_delta_share"]["mean"] for b in BRIDGES]
    ch = [ana["target_side"]["chance_baseline"]] + \
        [ana["per_bridge"][b]["chance_decoy_share"] for b in BRIDGES]
    nl = ana["marginal_null_diagnostic"]["decoy_share_amount_only_null"]
    xs = np.arange(len(labels))
    w = 0.3
    ax.bar(xs - w / 2, obs, w, color=C_U, label=r"true decoy pair $\delta^T$ share")
    ax.bar(xs + w / 2, ch, w, color=C_BASE, label="random two-target baseline")
    ax.axhline(nl, color="#2ca02c", ls="--", lw=1.4,
               label=f"marginal-only null = {nl:.4f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"Combined $\delta^T$ share")
    ax.set_title("(a) Decoy-target mass concentration", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.02, 0.06,
            f"permutation p = {ana['target_side']['permutation_p']:.3g} "
            f"(n_perm={ana['target_side']['n_perm']})",
            transform=ax.transAxes, fontsize=7, va="bottom")

    ax = axes[1]
    both = [ana["target_side"]["top2_both_hit"]] + \
        [ana["per_bridge"][b]["decoy_top2_both_hit"] for b in BRIDGES]
    any_ = [ana["target_side"]["top2_any_hit"]] + \
        [ana["per_bridge"][b]["decoy_top2_any_hit"] for b in BRIDGES]
    ax.bar(xs - w / 2, both, w, color="#9467bd", label="both decoys in Top-2")
    ax.bar(xs + w / 2, any_, w, color="#8c564b", label="at least one decoy in Top-2")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Decoy Top-2 diagnostics (secondary)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("S10 (post-hoc, not preregistered) - target-side decoy localization",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "s10_target_decoy_localization")


def figure_overview(ana: dict[str, Any], src: pd.DataFrame, tgt: pd.DataFrame
                    ) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax = axes[0]
    u = src.loc[src["truth_unmatched"] == 1, "delta_s_share"].dropna().to_numpy(float)
    m = src.loc[src["truth_unmatched"] == 0, "delta_s_share"].dropna().to_numpy(float)
    parts = ax.violinplot([m, u], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], (C_M, C_U)):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"matched\n(n={m.size})", f"true unmatched\n(n={u.size})"],
                       fontsize=8)
    ax.set_ylabel(r"$\delta^S$ share within template")
    ax.set_title(r"(a) Source side: $\delta^S$ attribution", fontsize=9.5)

    ax = axes[1]
    d = tgt.loc[tgt["truth_decoy"] == 1, "delta_t_share"].dropna().to_numpy(float)
    n = tgt.loc[tgt["truth_decoy"] == 0, "delta_t_share"].dropna().to_numpy(float)
    parts = ax.violinplot([n, d], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], (C_M, C_U)):
        pc.set_facecolor(c)
        pc.set_alpha(0.55)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"non-decoy\n(n={n.size})", f"decoy\n(n={d.size})"], fontsize=8)
    ax.set_ylabel(r"$\delta^T$ share within template")
    ax.set_title(r"(b) Target side: $\delta^T$ attribution", fontsize=9.5)

    fig.suptitle("S10 (post-hoc, not preregistered) - unmatched-mass attribution "
                 "overview", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "s10_unmatched_mass_attribution_overview")


def build_all() -> dict[str, Any]:
    ana = json.loads((RESULTS / "s10_analysis.json").read_text(encoding="utf-8"))
    src = pd.read_csv(RESULTS / "source_localization_long.csv")
    tgt = pd.read_csv(RESULTS / "target_localization_long.csv")
    tpl = pd.read_csv(RESULTS / "template_localization_summary.csv")
    out = {
        "s10_source_unmatched_localization": figure_source(ana, src, tpl),
        "s10_target_decoy_localization": figure_target(ana, tpl),
        "s10_unmatched_mass_attribution_overview": figure_overview(ana, src, tgt),
    }
    (FIGURES / "figures_index.json").write_text(json.dumps(out, indent=2) + "\n",
                                                encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
