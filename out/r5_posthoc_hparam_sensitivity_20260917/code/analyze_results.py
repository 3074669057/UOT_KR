"""Statistical / interpretive analysis of the R5 post-hoc sensitivity + ablation results.

Reads ONLY the produced CSVs and writes:
    results/analysis_summary.json          machine-readable findings
    results/analysis_tables.md             human-readable tables for the paper sections

No new experiment is executed here; this module only aggregates what was measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917"
RESULTS = EXP / "results"
ABL = EXP / "ablation"
PROV = EXP / "provenance" / "frozen_holdout_points.json"

BRIDGES = ("Celer", "Multi", "Poly")
BRIDGE_LABEL = {"Celer": "Celer cBridge", "Multi": "Multichain", "Poly": "PolyNetwork"}
METHODS = ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4")
DEFAULTS = {"k": 5.0, "epsilon": 0.05, "lambda": 0.5}

# Frozen absolute cost weights of the project; the paper's PRIMARY cost removes `amount` and
# renormalises the rest to sum 1 (dev_candidate.af_common).
FROZEN_ABS_WEIGHTS = {"amount": 0.40, "time": 0.25, "route": 0.15, "risk": 0.15,
                      "evidence": 0.05, "novelty": 0.05}
ABLATION_OMITTED = {"FULL_D6": None, "LOCO_AMOUNT": "amount", "LOCO_TIME": "time",
                    "LOCO_ROUTE": "route", "LOCO_RISK": "risk", "LOCO_EVIDENCE": "evidence",
                    "LOCO_NOVELTY": "novelty"}


def normalised_probe(variant: str) -> dict[str, float]:
    """Reproduce the runner's weight renormalisation, to verify it in the write-up."""
    omitted = ABLATION_OMITTED[variant]
    w = {k: (0.0 if k == omitted else v) for k, v in FROZEN_ABS_WEIGHTS.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    long_df = pd.read_csv(RESULTS / "sweep_long.csv")
    summ = pd.read_csv(RESULTS / "sweep_summary.csv")
    eps_diag = pd.read_csv(RESULTS / "epsilon_solver_diagnostics.csv")
    lam_mass = pd.read_csv(RESULTS / "lambda_unmatched_mass.csv")
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    return long_df, summ, eps_diag, lam_mass, prov


def sweep_slice(df: pd.DataFrame, sweep: str) -> pd.DataFrame:
    if sweep == "k":
        return df[(df["epsilon"] == 0.05) & (df["lambda"] == 0.5)]
    if sweep == "epsilon":
        return df[(df["k"] == 5.0) & (df["lambda"] == 0.5)]
    if sweep == "lambda":
        return df[(df["k"] == 5.0) & (df["epsilon"] == 0.05)]
    raise ValueError(sweep)


def macro_of(df: pd.DataFrame, method: str) -> float:
    """Published convention: unweighted mean over the 15 (bridge, seed) cells."""
    return float(df[df["method"] == method]["macro_edge_f1"].mean())


def paired_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 20000,
                       seed: int = 20240101) -> dict:
    """Two-sided paired sign-flip permutation test on the mean difference (a - b)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n == 0:
        return {"mean_diff": float("nan"), "p_value": float("nan"), "n": 0}
    obs = float(d.mean())
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm = (signs * d[None, :]).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs) - 1e-15).mean())
    return {"mean_diff": obs, "p_value": p, "n": int(n),
            "n_perm": int(n_perm), "rng_seed": int(seed)}


def exact_sign_test(d: np.ndarray) -> dict:
    """Exact two-sided sign test (n<=20)."""
    from math import comb
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return {"n": 0, "n_pos": 0, "p_value": float("nan")}
    k = int((d > 0).sum())
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return {"n": int(n), "n_pos": k, "n_neg": int(n - k),
            "p_value_two_sided": float(min(1.0, 2 * tail))}


def main() -> int:
    long_df, summ, eps_diag, lam_mass, prov = load()
    out: dict = {"generated_from": {
        "sweep_long": "results/sweep_long.csv",
        "sweep_summary": "results/sweep_summary.csv",
        "epsilon_solver_diagnostics": "results/epsilon_solver_diagnostics.csv",
        "lambda_unmatched_mass": "results/lambda_unmatched_mass.csv",
        "ablation_summary": "ablation/cost_component_ablation_summary.csv",
        "frozen_holdout_provenance": "provenance/frozen_holdout_points.json",
    }}
    md: list[str] = []
    md.append("# R5 post-hoc sensitivity & cost-ablation: computed result summary\n")
    md.append("All numbers below are computed directly from the CSVs written by this "
              "experiment. Development seeds: **201-205** only. The frozen confirmatory "
              "holdout (301-305) was **not re-run**.\n")

    # =====================================================================
    # 1. Default point: is it the grid optimum?  (decided by data, not assumed)
    # =====================================================================
    md.append("\n## 1. Default configuration vs the post-hoc grid\n")
    md.append("\nAggregation convention: macro edge F1 = unweighted mean over the 15 "
              "(bridge x seed) cells, matching the published frozen anchors.\n")
    optimum = {}
    for sweep, pcol in (("k", "k"), ("epsilon", "epsilon"), ("lambda", "lambda")):
        sl = sweep_slice(long_df, sweep)
        md.append(f"\n### 1.{sweep} sweep\n\n")
        md.append(f"| {pcol} | RAW macro-F1 | COND macro-F1 | mean sinkhorn iters | all converged |\n")
        md.append("|---|---|---|---|---|\n")
        rows = []
        for v in sorted(sl[pcol].unique()):
            g = sl[sl[pcol] == v]
            r = {m: macro_of(g, m) for m in METHODS}
            it = float(g["sinkhorn_iterations"].mean())
            conv = bool(g["sinkhorn_converged"].all())
            md.append(f"| {v:g} | {r['RAW_UOT_PLAN_D4']:.4f} | "
                      f"{r['CONDITIONAL_UOT_D4']:.4f} | {it:.1f} | {conv} |\n")
            rows.append({"value": float(v), **r, "mean_iters": it, "all_converged": conv})
        for m in METHODS:
            best = max(rows, key=lambda x: x[m])
            dv = [r for r in rows if r["value"] == DEFAULTS[pcol]][0][m]
            optimum[f"{sweep}:{m}"] = {
                "grid_best_value": best["value"],
                "grid_best_f1": best[m],
                "default_f1": dv,
                "default_is_grid_best": bool(abs(best["value"] - DEFAULTS[pcol]) < 1e-12),
                "default_minus_best": dv - best[m],
                "default_rank": 1 + sum(1 for r in rows if r[m] > dv + 1e-12),
                "n_grid_points": len(rows),
            }
            tag = "IS the grid best" if abs(best["value"] - DEFAULTS[pcol]) < 1e-12 \
                else f"is NOT the grid best (best = {best['value']:g}, " \
                     f"delta = {dv - best[m]:+.4f})"
            md.append(f"\n* {m}: default {pcol}={DEFAULTS[pcol]:g} -> {dv:.4f}; "
                      f"grid best {best['value']:g} -> {best[m]:.4f}; "
                      f"rank {optimum[f'{sweep}:{m}']['default_rank']}/{len(rows)}; "
                      f"default **{tag}**.\n")
    out["default_vs_grid_optimum"] = optimum

    # =====================================================================
    # 2. Pairing / stability around the default
    # =====================================================================
    md.append("\n## 2. Stability of the development results around the default\n")
    stability = {}
    for sweep, pcol in (("k", "k"), ("epsilon", "epsilon"), ("lambda", "lambda")):
        sl = sweep_slice(long_df, sweep)
        rec = {}
        for m in METHODS:
            g = sl[sl["method"] == m]
            base = macro_of(g, m)
            # exclude the default point itself
            other = g[g[pcol] != DEFAULTS[pcol]]
            devs = []
            for v in sorted(other[pcol].unique()):
                devs.append(abs(macro_of(other[other[pcol] == v], m) - base))
            rec[m] = {
                "default_f1": base,
                "max_abs_deviation_over_other_grid_points": float(max(devs)),
                "mean_abs_deviation_over_other_grid_points": float(np.mean(devs)),
                "worst_value": float(sorted(other[pcol].unique())[int(np.argmax(devs))]),
            }
        stability[sweep] = rec
        md.append(f"\n* `{sweep}`: RAW max |ΔF1| from default over other grid points = "
                  f"{rec['RAW_UOT_PLAN_D4']['max_abs_deviation_over_other_grid_points']:.4f}; "
                  f"COND = "
                  f"{rec['CONDITIONAL_UOT_D4']['max_abs_deviation_over_other_grid_points']:.4f}\n")
    out["stability_around_default"] = stability

    # =====================================================================
    # 3. RAW vs CONDITIONAL gap as a function of the parameter
    # =====================================================================
    md.append("\n## 3. Paired RAW vs CONDITIONAL gap vs each hyper-parameter\n")
    gaps = {}
    for sweep, pcol in (("k", "k"), ("epsilon", "epsilon"), ("lambda", "lambda")):
        sl = sweep_slice(long_df, sweep)
        rows = []
        md.append(f"\n### 3.{sweep}\n\n| {pcol} | ΔF1 (COND − RAW) | paired p | worsens? |\n|---|---|---|---|\n")
        for v in sorted(sl[pcol].unique()):
            g = sl[sl[pcol] == v]
            piv = g.pivot_table(index=["bridge", "seed"], columns="method",
                                values="macro_edge_f1")
            a = piv["CONDITIONAL_UOT_D4"].to_numpy()
            b = piv["RAW_UOT_PLAN_D4"].to_numpy()
            st = paired_permutation(a, b)
            md.append(f"| {v:g} | {st['mean_diff']:+.4f} | {st['p_value']:.4g} | "
                      f"{'no' if st['mean_diff'] > 0 else 'yes'} |\n")
            rows.append({"value": float(v), **st})
        gaps[sweep] = rows
    out["cond_minus_raw_gap"] = gaps

    # =====================================================================
    # 4. Solver findings (epsilon)
    # =====================================================================
    md.append("\n## 4. Solver stability vs epsilon\n")
    eps_rows = []
    md.append("\n| ε | mean iters | min iters | max iters | converged cells | max final err | "
              "mean δ^S | mean δ^T | mean ΔF1(cond−raw) |\n|---|---|---|---|---|---|---|---|---|\n")
    for v in sorted(eps_diag["epsilon"].unique()):
        g = eps_diag[eps_diag["epsilon"] == v]
        sl = long_df[(long_df["k"] == 5.0) & (long_df["lambda"] == 0.5)
                     & (long_df["epsilon"] == v)]
        gt = g[g["method"] == "CONDITIONAL_UOT_D4"]
        gtl = sl[sl["method"] == "CONDITIONAL_UOT_D4"]
        gaps_v = (macro_of(sl, "CONDITIONAL_UOT_D4") - macro_of(sl, "RAW_UOT_PLAN_D4"))
        rec = {
            "epsilon": float(v),
            "mean_iterations": float(gt["sinkhorn_iterations"].mean()),
            "min_iterations": int(gt["sinkhorn_iterations"].min()),
            "max_iterations": int(gt["sinkhorn_iterations"].max()),
            "n_cells": int(len(gt)),
            "n_converged": int(gt["sinkhorn_converged"].sum()),
            "all_converged": bool(gt["sinkhorn_converged"].all()),
            "max_final_err": float(gt["sinkhorn_final_residual"].max()),
            "any_hit_max_iter": bool(gt["sinkhorn_hit_max_iter"].any()),
            "mean_delta_s_total": float(gtl["delta_s_total"].mean()),
            "mean_delta_t_total": float(gtl["delta_t_total"].mean()),
            "mean_transport_mass": float(gtl["transport_mass_total"].mean()),
            "delta_f1_cond_minus_raw": float(gaps_v),
        }
        eps_rows.append(rec)
        md.append(f"| {v:g} | {rec['mean_iterations']:.1f} | {rec['min_iterations']} | "
                  f"{rec['max_iterations']} | {rec['n_converged']}/{rec['n_cells']} | "
                  f"{rec['max_final_err']:.2e} | {rec['mean_delta_s_total']:.4f} | "
                  f"{rec['mean_delta_t_total']:.4f} | {rec['delta_f1_cond_minus_raw']:+.4f} |\n")
    out["epsilon_solver"] = eps_rows
    it_hi = [r for r in eps_rows if r["epsilon"] == max(r["epsilon"] for r in eps_rows)][0]
    it_lo = [r for r in eps_rows if r["epsilon"] == min(r["epsilon"] for r in eps_rows)][0]
    out["epsilon_iteration_trend"] = {
        "smallest_epsilon": it_lo["epsilon"], "iters_at_smallest": it_lo["mean_iterations"],
        "largest_epsilon": it_hi["epsilon"], "iters_at_largest": it_hi["mean_iterations"],
        "all_converged_everywhere": bool(all(r["all_converged"] for r in eps_rows)),
        "any_hit_max_iter": bool(any(r["any_hit_max_iter"] for r in eps_rows)),
    }

    # =====================================================================
    # 5. Unmatched mass vs lambda
    # =====================================================================
    md.append("\n## 5. Unmatched mass vs lambda (λ = UOT marginal relaxation reg_m)\n")
    md.append("\nδ^S = Σ_i |Σ_j P_ij − a_i| (source marginal violation); "
              "δ^T = Σ_j |Σ_i P_ij − b_j| (target marginal violation); these are the genuine "
              "unbalanced-OT marginal deviations returned by the solver.\n")
    lam_rows = []
    md.append("\n| λ | mean δ^S | mean δ^T | mean δ_total | mean transport mass | "
              "COND F1 | RAW F1 |\n|---|---|---|---|---|---|---|\n")
    for v in sorted(lam_mass["lambda"].unique()):
        g = lam_mass[(lam_mass["lambda"] == v) & (lam_mass["method"] == "CONDITIONAL_UOT_D4")]
        sl = long_df[(long_df["k"] == 5.0) & (long_df["epsilon"] == 0.05)
                     & (long_df["lambda"] == v)]
        rec = {
            "lambda": float(v),
            "mean_delta_s_total": float(g["delta_s_total"].mean()),
            "mean_delta_t_total": float(g["delta_t_total"].mean()),
            "mean_delta_total": float(g["delta_total"].mean()),
            "mean_transport_mass": float(g["transport_mass_total"].mean()),
            "mean_mass_retained_fraction": float(g["mass_retained_fraction"].mean()),
            "mean_iterations": float(g["sinkhorn_iterations"].mean()),
            "f1_conditional": macro_of(sl, "CONDITIONAL_UOT_D4"),
            "f1_raw": macro_of(sl, "RAW_UOT_PLAN_D4"),
        }
        lam_rows.append(rec)
        md.append(f"| {v:g} | {rec['mean_delta_s_total']:.4f} | {rec['mean_delta_t_total']:.4f} | "
                  f"{rec['mean_delta_total']:.4f} | {rec['mean_transport_mass']:.4f} | "
                  f"{rec['f1_conditional']:.4f} | {rec['f1_raw']:.4f} |\n")
    out["lambda_unmatched_mass"] = lam_rows
    f1s = [r["f1_conditional"] for r in lam_rows]
    masses = [r["mean_transport_mass"] for r in lam_rows]
    ds = [r["mean_delta_total"] for r in lam_rows]
    out["lambda_summary"] = {
        "f1_conditional_range": [min(f1s), max(f1s)],
        "f1_conditional_span": max(f1s) - min(f1s),
        "delta_total_range": [min(ds), max(ds)],
        "transport_mass_range": [min(masses), max(masses)],
        "monotone_mass_increase": bool(all(b >= a - 1e-12 for a, b in zip(masses, masses[1:]))),
        "monotone_delta_decrease": bool(all(b <= a + 1e-12 for a, b in zip(ds, ds[1:]))),
    }

    # =====================================================================
    # 6. Cost-component ablation + Multichain specificity
    # =====================================================================
    md.append("\n## 6. Cost-component leave-one-out ablation (CONDITIONAL_UOT_D4)\n")
    ab = pd.read_csv(ABL / "cost_component_ablation_summary.csv")
    ab = ab[ab["method"] == "CONDITIONAL_UOT_D4"].copy()
    md.append("\nΔF1 is relative to the paper's PRIMARY cost = the amount-free renormalised "
              "5-component cost {time, route, risk, evidence, novelty} renormalised to sum 1. "
              "That is exactly the frozen paper configuration and it is the `LOCO_AMOUNT` row "
              "(i.e. the amount component is already absent). Each other row omits ONE "
              "additional component and renormalises the remaining weights to sum 1, so the "
              "cost scale — and therefore the meaning of ε — is unchanged across rows. "
              "`FULL_D6` adds the amount component back as a sixth component (all six "
              "renormalised to sum 1) and is the complete-cost control.\n")
    md.append("\n| variant | omitted | Celer ΔF1 | Multichain ΔF1 | Poly ΔF1 | "
              "mean ΔF1 | seeds<0 / 5 |\n|---|---|---|---|---|---|---|\n")
    piv = ab.pivot_table(index=["variant", "omitted_component"], columns="bridge",
                         values="delta_f1_vs_primary_cost")
    seedneg = ab.pivot_table(index=["variant", "omitted_component"], columns="bridge",
                             values="n_seeds_with_negative_delta")
    abl_rows = []
    for (variant, omitted), row in piv.iterrows():
        label = ("none (paper primary cost)" if variant == "LOCO_AMOUNT"
                 else ("none (six-component control)" if variant == "FULL_D6" else omitted))
        vals = {b: float(row[b]) for b in BRIDGES}
        means = float(np.mean(list(vals.values())))
        neg = {b: int(seedneg.loc[(variant, omitted), b]) for b in BRIDGES}
        md.append(f"| {variant} | {label} | {vals['Celer']:+.4f} | {vals['Multi']:+.4f} | "
                  f"{vals['Poly']:+.4f} | {means:+.4f} | "
                  f"{neg['Celer']}/{neg['Multi']}/{neg['Poly']} |\n")
        abl_rows.append({"variant": variant, "omitted_component": label,
                         "delta_f1_per_bridge": vals, "mean_delta_f1": means,
                         "n_seeds_negative_per_bridge": neg})
    out["cost_ablation"] = {
        "rows": abl_rows,
        "baseline": ("paper primary amount-free renormalised cost (5 components, weight sum 1); "
                     "identified with the LOCO_AMOUNT row"),
        "renormalisation": ("omitting a component sets its weight to 0 and rescales the "
                            "remaining weights so that they sum to 1; the cost scale is "
                            "therefore identical in every row and epsilon keeps its meaning"),
        "weights_sum_check": {v: sum(normalised_probe(v).values()) for v in
                              ("FULL_D6", "LOCO_AMOUNT", "LOCO_TIME", "LOCO_ROUTE",
                               "LOCO_RISK", "LOCO_EVIDENCE", "LOCO_NOVELTY")},
    }

    # FULL_D6 control
    f6 = ab[ab["variant"] == "FULL_D6"].set_index("bridge")
    out["cost_ablation"]["full_d6_control"] = {
        b: {"f1": float(f6.loc[b, "mean_macro_edge_f1"]),
            "delta_vs_primary": float(f6.loc[b, "delta_f1_vs_primary_cost"])}
        for b in BRIDGES}

    # ---- Multichain specificity: paired per-seed deltas ----
    md.append("\n### 6.1 Is any component specifically more important for Multichain?\n")
    seed_df = pd.read_csv(ABL / "cost_component_ablation_long.csv")
    seed_df = seed_df[seed_df["method"] == "CONDITIONAL_UOT_D4"]
    base = (seed_df[seed_df["variant"] == "LOCO_AMOUNT"]
            .set_index(["bridge", "seed"])["macro_edge_f1"])
    spec: dict = {}
    md.append("\n| omitted | Celer ΔF1 | Multichain ΔF1 | Poly ΔF1 | "
              "Multi−Celer (perm p) | Multi−Poly (perm p) | Multi is largest of 3? |\n"
              "|---|---|---|---|---|---|---|\n")
    for variant in ("LOCO_TIME", "LOCO_ROUTE", "LOCO_RISK", "LOCO_EVIDENCE",
                    "LOCO_NOVELTY"):
        per = {}
        for b in BRIDGES:
            g = seed_df[(seed_df["variant"] == variant) & (seed_df["bridge"] == b)]
            g = g.set_index("seed")["macro_edge_f1"].sort_index()
            per[b] = (g - base.loc[b].sort_index()).to_numpy(dtype=float)
        p_mc = paired_permutation(per["Multi"], per["Celer"])
        p_mp = paired_permutation(per["Multi"], per["Poly"])
        sign_mc = exact_sign_test(per["Multi"] - per["Celer"])
        means = {b: float(np.mean(v)) for b, v in per.items()}
        md.append(f"| {variant.replace('LOCO_', '')} | {means['Celer']:+.4f} | "
                  f"{means['Multi']:+.4f} | {means['Poly']:+.4f} | "
                  f"{p_mc['mean_diff']:+.4f} (p={p_mc['p_value']:.3g}) | "
                  f"{p_mp['mean_diff']:+.4f} (p={p_mp['p_value']:.3g}) | "
                  f"{'yes' if means['Multi'] < means['Celer'] and means['Multi'] < means['Poly'] else 'no'} |\n")
        spec[variant] = {
            "delta_f1_per_bridge_mean": means,
            "delta_f1_per_seed": {b: [float(x) for x in v] for b, v in per.items()},
            "multichain_minus_celer": p_mc,
            "multichain_minus_poly": p_mp,
            "multichain_vs_celer_sign_test": sign_mc,
            "multichain_is_largest_loss": bool(means["Multi"] < means["Celer"]
                                               and means["Multi"] < means["Poly"]),
            "significant_at_0.05_vs_both": bool(p_mc["p_value"] < 0.05 and p_mp["p_value"] < 0.05),
        }
    out["multichain_specificity"] = spec

    # =====================================================================
    # 7. Holdout markers vs development default
    # =====================================================================
    md.append("\n## 7. Frozen holdout markers (read-only) vs the development default\n")
    md.append("\nThe holdout markers are pre-existing frozen results at the paper default "
              "(k=5, ε=0.05, λ=0.5); seeds 301-305 were **not** re-run.\n")
    md.append("\n| bridge | method | dev default F1 | frozen holdout F1 | Δ |\n|---|---|---|---|---|\n")
    holdout_cmp = {}
    for p in prov["points"]:
        if p["metric"] != "macro_edge_f1" or p["bridge"] not in BRIDGES:
            continue
        b, m = p["bridge"], p["method"]
        sl = sweep_slice(long_df, "k")
        dev = float(sl[(sl["bridge"] == b) & (sl["method"] == m) & (sl["k"] == 5.0)]
                    ["macro_edge_f1"].mean())
        holdout_cmp[f"{b}:{m}"] = {
            "dev_default_f1": dev, "frozen_holdout_f1": float(p["value"]),
            "delta": float(p["value"]) - dev,
            "holdout_source_path": p["source"]["path"],
            "holdout_source_sha256": p["source"]["sha256"],
            "holdout_source_row": p["source"].get("row"),
        }
        md.append(f"| {BRIDGE_LABEL[b]} | {m} | {dev:.4f} | {p['value']:.4f} | "
                  f"{p['value'] - dev:+.4f} |\n")
    out["holdout_vs_dev_default"] = holdout_cmp

    # =====================================================================
    # 8. Sanity: no silent NaN, complete design
    # =====================================================================
    numeric_cols = ["precision", "recall", "macro_edge_f1", "micro_edge_f1",
                    "sinkhorn_iterations", "sinkhorn_final_residual",
                    "delta_s_total", "delta_t_total", "delta_total"]
    nan_counts = {c: int(long_df[c].isna().sum()) for c in numeric_cols}
    design = {
        "n_rows": int(len(long_df)),
        "n_bridges": int(long_df["bridge"].nunique()),
        "n_seeds": int(long_df["seed"].nunique()),
        "n_configs": int(long_df.groupby(["k", "epsilon", "lambda"]).ngroups),
        "n_methods": int(long_df["method"].nunique()),
        "expected_rows": 3 * 5 * 14 * 2,
        "complete": bool(len(long_df) == 3 * 5 * 14 * 2),
        "nan_counts": nan_counts,
        "any_nan_in_metric_columns": bool(any(v > 0 for v in nan_counts.values())),
        "seeds_present": sorted(int(s) for s in long_df["seed"].unique()),
        "forbidden_seeds_present": sorted(
            int(s) for s in long_df["seed"].unique() if int(s) in (301, 302, 303, 304, 305)),
    }
    out["design_integrity"] = design
    md.append("\n## 8. Design integrity\n")
    md.append(f"\n* sweep_long rows: {design['n_rows']} (expected {design['expected_rows']}) "
              f"-> complete: **{design['complete']}**\n")
    md.append(f"* bridges: {design['n_bridges']}, seeds: {design['seeds_present']}, "
              f"configs: {design['n_configs']}, methods: {design['n_methods']}\n")
    md.append(f"* NaN in metric columns: {nan_counts} "
              f"-> any NaN: **{design['any_nan_in_metric_columns']}**\n")
    md.append(f"* holdout seeds present in results: {design['forbidden_seeds_present']} "
              f"(must be empty)\n")

    # =====================================================================
    # 9. Answer the two propositions explicitly
    # =====================================================================
    a_true = True
    b_true = all(not v["default_is_grid_best"] for v in optimum.values())
    conclusion = {
        "proposition_A_default_not_chosen_from_this_grid": {
            "holds": a_true,
            "evidence": ("The defaults (k=5, eps=0.05, lam=0.5) are the frozen preregistered "
                         "values: holdout_common.identity_check enforces reg==0.05 and "
                         "reg_m==0.5, and the decoder rank constant K5==5 appears in "
                         "holdout_common.py, dev_candidate2/cp_common.py and "
                         "baseline_mechanism/common.py. This grid was executed afterwards and "
                         "was never used to select anything; the main-experiment defaults were "
                         "not modified (validation check 14)."),
        },
        "proposition_B_default_is_not_the_grid_optimum": {
            "holds": b_true,
            "detail": {key: {"default_is_grid_best": v["default_is_grid_best"],
                             "grid_best_value": v["grid_best_value"],
                             "default_f1": v["default_f1"],
                             "default_minus_best": v["default_minus_best"]}
                       for key, v in optimum.items()},
            "note": ("This is a statement about the DEVELOPMENT grid only. It is not a claim "
                     "about the holdout and it does not license any change of the paper's "
                     "defaults."),
        },
    }
    out["propositions"] = conclusion

    (RESULTS / "analysis_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESULTS / "analysis_tables.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "wrote": ["results/analysis_summary.json", "results/analysis_tables.md"],
        "proposition_B_default_is_grid_optimum": b_true,
        "design_complete": design["complete"],
        "any_nan": design["any_nan_in_metric_columns"],
        "forbidden_seeds_present": design["forbidden_seeds_present"],
        "multichain_specificity_significant": {
            k: v["significant_at_0.05_vs_both"] for k, v in spec.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
