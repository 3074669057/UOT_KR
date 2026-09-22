"""Layer-2 main experiment: three bridges x five methods, 48 templates x seeds 42-46.

Fairness: all methods operate on the IDENTICAL frozen grid (C_effective, flow orders,
marginals) loaded from the frozen faithful pipeline per-seed artifacts.
- RC-UOT-Q: frozen transport matrix decoded with decode_correspondence semantics at 1e-9
  (the frozen decode; the plan is NOT re-solved).
- Balanced-OT: same C_effective + same risk/evidence-weighted marginals (unit-normalized),
  strictly balanced entropic OT (reg=0.05), decoded with the SAME 1e-9 threshold rule.
- Threshold-MM: same C_effective with the calibration-selected GLOBAL cutoff.
- Connector-style / ABCTracer-style: project's existing per-source top-1 rules on the
  shared decomposition (amount; 0.75*amount + 0.25*time), scoped per template.
All methods are evaluated by the SAME unified evaluator (exact split/merge, edge P/R/F1,
degree accuracy, FP count, coverage) with the SAME ground truth.

Usage:
  python run_main_baseline_study.py [--bridges Celer,Multi,Poly] [--seeds 42,43,44,45,46]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import (  # noqa: E402
    BRIDGES, FROZEN_PARAMS, STUDY, TEST_SEEDS,
    bootstrap_ci, decode_abctracer, decode_connector, decode_plan, decode_threshold_mm,
    evaluate_method, load_frozen_instance, solve_balanced_ot, truth_structure, tpl_of,
)

OUT = STUDY / "per_seed"


def load_calibration() -> dict[str, Any]:
    sel = json.loads((STUDY / "calibration" / "selected_threshold.json").read_text(encoding="utf-8"))
    return {"tau": float(sel["global_tau"]), "cutoff": float(sel["global_cutoff_cost"]), "meta": sel}


def run_seed(bridge: str, seed: int, cal: dict[str, Any]) -> dict[str, Any]:
    inst = load_frozen_instance(bridge, seed)
    truth = truth_structure(inst["labels"])
    root = OUT / bridge / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)

    method_edges: dict[str, list[tuple[str, str]]] = {}
    # RC-UOT-Q — frozen plan, frozen decode (threshold 1e-9)
    method_edges["RC-UOT-Q"] = decode_plan(inst["P"], inst["sids"], inst["tids"],
                                           threshold=FROZEN_PARAMS["uot_decode_threshold"])
    # Threshold-MM — same C, calibration-selected GLOBAL cutoff
    method_edges["Threshold-MM"] = decode_threshold_mm(inst, cal["cutoff"])
    # Balanced-OT — same C + same marginals, strictly balanced
    bot = solve_balanced_ot(inst["C"], inst["a_rw"], inst["b_ev"], reg=FROZEN_PARAMS["uot_reg"])
    np.savez(root / "balanced_ot_transport.npz", P=bot["P"])
    method_edges["Balanced-OT"] = decode_plan(bot["P"], inst["sids"], inst["tids"],
                                              threshold=FROZEN_PARAMS["uot_decode_threshold"])
    # One-to-one baselines — project's existing rules on the shared decomposition
    method_edges["Connector-style"] = decode_connector(inst)
    method_edges["ABCTracer-style"] = decode_abctracer(inst)

    per_method: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for method, edges in method_edges.items():
        df, summ = evaluate_method(inst, truth, edges)
        summ["method"] = method
        summ["n_edges_total"] = len(edges)
        per_method[method] = summ
        for _, r in df.iterrows():
            all_rows.append({
                "bridge": bridge, "seed": seed, "method": method, **r.to_dict(),
            })
        edge_df = pd.DataFrame([{"src_flow_id": s, "dst_flow_id": d} for s, d in edges])
        edge_df.to_csv(root / f"edges_{method.replace(' ', '_').replace('-', '_')}.csv", index=False)

    tpl_df = pd.DataFrame(all_rows)
    tpl_df.to_csv(root / "templates_per_method.csv", index=False)

    # per-template degree annotations + ground truth (for verification)
    degree_rows: list[dict[str, Any]] = []
    for t, tr in sorted(truth.items()):
        out_d_gt = {}
        in_d_gt = {}
        for s, d in tr["positive"]:
            out_d_gt[s] = out_d_gt.get(s, 0) + 1
            in_d_gt[d] = in_d_gt.get(d, 0) + 1
        degree_rows.append({
            "bridge": bridge, "seed": seed, "template_id": t,
            "ground_truth_edges": json.dumps(sorted(tr["positive"])),
            "source_degree_gt": json.dumps(out_d_gt), "target_degree_gt": json.dumps(in_d_gt),
            "unmatched_src": json.dumps(sorted(tr["unmatched_src"])),
            "hidden_dst": json.dumps(sorted(tr["hidden_dst"])),
        })
    pd.DataFrame(degree_rows).to_csv(root / "template_truth.csv", index=False)

    summary = {
        "bridge": bridge, "seed": seed,
        "calibration_tau": cal["tau"], "calibration_cutoff": cal["cutoff"],
        "frozen_source": str(inst["root"]),
        "balanced_ot": {"converged": bot["converged"], "final_err": bot["final_err"],
                        "row_residual": bot["row_residual"], "col_residual": bot["col_residual"],
                        "transported_mass": float(bot["P"].sum())},
        "rc_uot_q_transported_mass": float(inst["P"].sum()),
        "rc_uot_q_unmatched_source_mass": float(np.clip(inst["a_rw"].sum() - inst["P"].sum(axis=1).sum(), 0, None)),
        "per_method_summary": per_method,
    }
    (root / "seed_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridges", default=None)
    ap.add_argument("--seeds", default=None)
    cli = ap.parse_args()
    bridges = tuple(cli.bridges.split(",")) if cli.bridges else BRIDGES
    seeds = tuple(int(x) for x in cli.seeds.split(",")) if cli.seeds else TEST_SEEDS

    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calibration()
    print(f"global tau={cal['tau']} cutoff={cal['cutoff']:.6f}", flush=True)

    seed_summaries: list[dict[str, Any]] = []
    for br in bridges:
        for seed in seeds:
            print(f"[main] {br} seed {seed} ...", flush=True)
            s = run_seed(br, seed, cal)
            seed_summaries.append(s)
            for m, ms in s["per_method_summary"].items():
                print(f"  {m:16s} split_ex={ms['split_exact']:.3f} merge_ex={ms['merge_exact']:.3f} "
                      f"P={ms['edge_precision']:.4f} R={ms['edge_recall']:.4f} F1={ms['edge_f1']:.4f} "
                      f"FP={ms['edge_fp_total']}", flush=True)

    # collect per-seed per-template rows for aggregation
    tpl_rows: list[dict[str, Any]] = []
    for br in bridges:
        for seed in seeds:
            p = OUT / br / f"seed_{seed}" / "templates_per_method.csv"
            if p.is_file():
                tpl_rows.extend(pd.read_csv(p, dtype={"bridge": str, "method": str}).to_dict("records"))
    tpl_all = pd.DataFrame(tpl_rows)
    tpl_all.to_csv(OUT / "structural_per_template_all.csv", index=False)

    agg = STUDY / "aggregated"
    agg.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, Any]] = []
    for s in seed_summaries:
        for m, ms in s["per_method_summary"].items():
            per_seed_rows.append({"bridge": s["bridge"], "seed": s["seed"], "method": m, **{
                k: v for k, v in ms.items() if k != "n_templates"}})
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(agg / "per_seed_metrics.csv", index=False)

    def _fmt_stat(vals: pd.Series) -> dict[str, float]:
        v = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(dtype=float)
        lo, hi = bootstrap_ci(v)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "ci95_lo": lo, "ci95_hi": hi, "n_seeds": int(len(v))}

    main_rows: list[dict[str, Any]] = []
    for br in bridges:
        sub = tpl_all[tpl_all["bridge"] == br]
        for m in sorted(sub["method"].unique()):
            ms = sub[sub["method"] == m]
            row: dict[str, Any] = {"bridge": br, "method": m,
                                   "n_templates_per_seed": int(ms.groupby("seed").size().max())}
            for col in ("split_exact", "merge_exact", "edge_precision", "edge_recall", "edge_f1",
                        "coverage"):
                seed_means = ms.groupby("seed")[col].mean()
                st = _fmt_stat(seed_means)
                row[f"{col}_mean"] = st["mean"]; row[f"{col}_std"] = st["std"]
                row[f"{col}_ci95_lo"] = st["ci95_lo"]; row[f"{col}_ci95_hi"] = st["ci95_hi"]
                row[f"{col}_n_seeds"] = st["n_seeds"]
            for col in ("deg_acc_split", "deg_acc_merge"):
                seed_means = ms.groupby("seed")[col].mean()
                st = _fmt_stat(seed_means)
                row[f"{col}_mean"] = st["mean"]; row[f"{col}_std"] = st["std"]
            deg = pd.concat([ms["deg_acc_split"], ms["deg_acc_merge"]], axis=0)
            row["degree_acc_mean"] = float(deg.mean()); row["degree_acc_std"] = float(deg.std(ddof=1))
            fp = ms.groupby("seed")["edge_fp"].mean()
            st = _fmt_stat(fp)
            row["fp_edges_mean"] = st["mean"]; row["fp_edges_std"] = st["std"]
            row["fp_edges_ci95_lo"] = st["ci95_lo"]; row["fp_edges_ci95_hi"] = st["ci95_hi"]
            npred = ms.groupby("seed")["n_pred_edges"].mean()
            st = _fmt_stat(npred)
            row["n_pred_edges_mean"] = st["mean"]; row["n_pred_edges_std"] = st["std"]
            # hierarchical bootstrap (seeds outer, templates inner) for key metrics
            for col in ("split_exact", "merge_exact", "edge_f1"):
                seed_means = ms.groupby("seed")[col].mean()
                lo, hi = hierarchical_bootstrap_ci(ms, col)
                row[f"{col}_hier_ci95_lo"] = lo; row[f"{col}_hier_ci95_hi"] = hi
            main_rows.append(row)
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(agg / "main_structural_comparison.csv", index=False)
    write_main_md(main_df)
    print(main_df[["bridge", "method", "split_exact_mean", "merge_exact_mean", "edge_f1_mean",
                   "fp_edges_mean"]].to_string(index=False))
    return 0


def hierarchical_bootstrap_ci(ms: pd.DataFrame, col: str, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.RandomState(1234)
    seeds = list(ms["seed"].unique())
    per = {s: pd.to_numeric(ms.loc[ms["seed"] == s, col], errors="coerce").dropna().to_numpy()
           for s in seeds}
    means = []
    for _ in range(n_boot):
        acc = []
        for s in rng.choice(seeds, size=len(seeds), replace=True):
            arr = per[s]
            if arr.size:
                acc.extend(rng.choice(arr, size=arr.size, replace=True))
        means.append(float(np.mean(acc)) if acc else 0.0)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_main_md(main_df: pd.DataFrame) -> None:
    lines = ["# Main structural comparison — three bridges x five methods",
             "",
             "48 templates x seeds 42-46 per bridge. Mean +/- std over the 5 seeds "
             "(per-seed value = mean over 48 templates); 95% bootstrap CI (seed resampling).",
             "RC-UOT-Q = frozen faithful run decoded at 1e-9. Balanced-OT = same C, same marginals,",
             "strictly balanced, same decode. Threshold-MM = same C, calibration GLOBAL tau. "
             "Connector/ABCTracer = project's existing per-source top-1 rules.",
             "",
             "| Bridge | Method | Split exact | Merge exact | Edge P | Edge R | Edge F1 | Degree acc | FP edges | Pred edges |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in main_df.iterrows():
        lines.append(
            f"| {r['bridge']} | {r['method']} | {r['split_exact_mean']:.3f} ± {r['split_exact_std']:.3f} | "
            f"{r['merge_exact_mean']:.3f} ± {r['merge_exact_std']:.3f} | {r['edge_precision_mean']:.3f} ± {r['edge_precision_std']:.3f} | "
            f"{r['edge_recall_mean']:.3f} ± {r['edge_recall_std']:.3f} | {r['edge_f1_mean']:.3f} ± {r['edge_f1_std']:.3f} | "
            f"{r['degree_acc_mean']:.3f} ± {r['degree_acc_std']:.3f} | {r['fp_edges_mean']:.1f} ± {r['fp_edges_std']:.1f} | "
            f"{r['n_pred_edges_mean']:.1f} ± {r['n_pred_edges_std']:.1f} |")
    lines += ["", "*Bootstrap CIs per metric are stored in `main_structural_comparison.csv` "
              "(seed-level and hierarchical).*"]
    (STUDY / "aggregated" / "main_structural_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
