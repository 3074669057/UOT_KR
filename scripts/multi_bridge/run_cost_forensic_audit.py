"""COST / FEATURE REPRESENTATION forensic audit (development seeds 201-205 ONLY).

1. Per-GT-edge vs confuser forensic table with per-component margins (cost/).
2. Feature scale audit (feature_scale/).
3. Amount semantics audit incl. double-counting check and split/merge child-parent
   penalty analysis (amount_semantics/).
4. Leave-one-component-out DIAGNOSTIC (component_ablation/): cost-space ranking only;
   causal-style ablation, never a weight search, never method selection.

Nothing here modifies any cost, weight, parameter, or frozen artifact.
"""
from __future__ import annotations

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

from diag.ctd_common import (  # noqa: E402
    BRIDGES, COMP_KEYS, CTD, DEV_SEEDS, FROZEN_WEIGHTS, cost_d4_edges, load_dev_cell,
)
from decoder_audit.da_common import evaluate_edges  # noqa: E402

WEIGHT_KEY = {"amount_cost": "amount", "time_cost": "time", "route_cost": "route",
              "risk_cost": "risk", "evidence_cost": "evidence",
              "address_novelty_cost": "novelty"}
LOCO = {"FULL": None, "NO_AMOUNT": "amount_cost", "NO_TIME": "time_cost",
        "NO_ROUTE": "route_cost", "NO_RISK": "risk_cost",
        "NO_EVIDENCE": "evidence_cost", "NO_NOVELTY": "address_novelty_cost"}


def main() -> int:
    for sub in ("cost", "feature_scale", "amount_semantics", "component_ablation"):
        (CTD / sub).mkdir(parents=True, exist_ok=True)

    forensic_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    amount_rows: list[dict[str, Any]] = []
    loco_rows: list[dict[str, Any]] = []

    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(bridge, seed)
            C = cell["C"]
            comps = cell["components"]
            truth = cell["truth"]
            sids, tids = cell["sids"], cell["tids"]
            n, m = C.shape
            idx_s = {s: i for i, s in enumerate(sids)}
            idx_t = {t: j for j, t in enumerate(tids)}
            # amounts from flows.json
            flows = json.loads((cell["root"] / "flows.json").read_text(encoding="utf-8"))
            amt_s = {str(f["flow_id"]): float(f.get("amount_usd", 0.0)) for f in flows["src"]}
            amt_t = {str(f["flow_id"]): float(f.get("amount_usd", 0.0)) for f in flows["dst"]}

            # --- feature scale (pool all cells) ---
            tot = C.ravel()
            scale_rows.append({"bridge": bridge, "seed": seed, "component": "total_cost",
                               **{q: float(np.percentile(tot, qq)) for q, qq in
                                  (("min", 0), ("p05", 5), ("p25", 25), ("median", 50),
                                   ("p75", 75), ("p95", 95), ("max", 100))},
                               "mean": float(tot.mean()), "std": float(tot.std()),
                               "weighted_share": 1.0})
            wsum = 0.0
            for k in COMP_KEYS:
                w = FROZEN_WEIGHTS[WEIGHT_KEY[k]]
                arr = comps[k].ravel() * w
                wsum += float(arr.sum())
            for k in COMP_KEYS:
                w = FROZEN_WEIGHTS[WEIGHT_KEY[k]]
                arr = comps[k].ravel()
                wc = arr * w
                scale_rows.append({"bridge": bridge, "seed": seed, "component": k,
                                   **{q: float(np.percentile(arr, qq)) for q, qq in
                                      (("min", 0), ("p05", 5), ("p25", 25), ("median", 50),
                                       ("p75", 75), ("p95", 95), ("max", 100))},
                                   "mean": float(arr.mean()), "std": float(arr.std()),
                                   "weighted_share": float(wc.sum() / max(wsum, 1e-12)),
                                   "weighted_mean": float(wc.mean()),
                                   "var_contribution": float(np.var(wc) / max(np.var(tot), 1e-12))})

            # --- per-GT-edge forensic + amount semantics ---
            for t, tr in truth.items():
                for s, d in tr["positive"]:
                    i = idx_s[s]
                    j = idx_t[d]
                    role = ("split" if (s, d) in tr["split"]
                            else "merge" if (s, d) in tr["merge"] else "decoy")
                    non_gt = [jj for jj in range(m) if (i, jj) not in
                              {(idx_s[ss], idx_t[dd]) for ss, dd in tr["positive"]}]
                    jc = int(min(non_gt, key=lambda jj: (C[i, jj], jj))) if non_gt else -1
                    row = {"bridge": bridge, "seed": seed, "template_id": t,
                           "structural_role": role, "src": s, "dst": d,
                           "gt_total_cost": float(C[i, j]),
                           "gt_row_rank": int(np.argsort(np.lexsort((np.arange(m), C[i])))[
                               np.where(np.lexsort((np.arange(m), C[i])) == j)[0][0]] + 1)}
                    if jc >= 0:
                        conf_role = tids[jc].split("__")[-1]
                        row.update({
                            "confuser_cost": float(C[i, jc]),
                            "cost_margin": float(C[i, jc] - C[i, j]),
                            "confuser_id": tids[jc], "confuser_role": conf_role,
                            "confuser_same_template": cell["tpl_t"][jc] == cell["tpl_s"][i],
                            "confuser_class": _classify(conf_role, cell["tpl_t"][jc], cell["tpl_s"][i]),
                        })
                        for k in COMP_KEYS:
                            row[f"gt_{k}"] = float(comps[k][i, j])
                            row[f"conf_{k}"] = float(comps[k][i, jc])
                            row[f"margin_{k}"] = float(comps[k][i, jc] - comps[k][i, j])
                        row["gt_amount"] = amt_s.get(s, float("nan"))
                        row["gt_dst_amount"] = amt_t.get(d, float("nan"))
                        row["conf_amount"] = amt_t.get(tids[jc], float("nan"))
                    forensic_rows.append(row)
                    # amount semantics
                    if role == "split":
                        parent = amt_s.get(s, 0.0)
                        sib = [amt_t.get(dd, 0.0) for dd in tr["split_dst"]]
                        full_d = [amt_t.get(dd, 0.0) for dd in (tr["merge_dst"] | set(tids[jj] for jj in range(m)
                                   if "__noise_dst_" in tids[jj] and cell["tpl_t"][jj] == t))]
                        amount_rows.append({
                            "bridge": bridge, "seed": seed, "template_id": t, "role": "split",
                            "parent_amount": parent, "children_sum": float(np.sum(sib)),
                            "child_amounts": sib,
                            "parent_to_child_cost": float(comps["amount_cost"][i, idx_t[list(tr["split_dst"])[0]]]),
                            "child_cost_gt_mean": float(np.mean([comps["amount_cost"][i, idx_t[dd]]
                                                                 for dd in tr["split_dst"]])),
                            "full_confuser_cost_mean": float(np.mean([
                                comps["amount_cost"][i, idx_t[dd]] for dd in
                                (tr["merge_dst"] | {dd for dd in tr["all_dst"]
                                                    if "__noise_dst_" in dd and dd in idx_t})])),
                        })
                    if role == "merge":
                        parent_dst = amt_t.get(d, 0.0)
                        child_srcs = [amt_s.get(ss, 0.0) for ss in tr["merge_src"]]
                        amount_rows.append({
                            "bridge": bridge, "seed": seed, "template_id": t, "role": "merge",
                            "parent_amount": parent_dst, "children_sum": float(np.sum(child_srcs)),
                            "child_amounts": child_srcs,
                            "parent_to_child_cost": float(comps["amount_cost"][idx_s[list(tr["merge_src"])[0]], j]),
                        })

            # --- LOCO (leave-one-component-out, cost space only, DIAGNOSTIC) ---
            for vname, drop_key in LOCO.items():
                Cx = C.copy()
                if drop_key is not None:
                    Cx = C - FROZEN_WEIGHTS[WEIGHT_KEY[drop_key]] * comps[drop_key]
                edges = cost_d4_edges(Cx, sids, tids)
                df, summ = evaluate_edges(cell, edges)
                gt_rank_rows = []
                gt_mutual = 0
                n_gt = 0
                cr = _col_ranks(Cx)
                for t, tr in truth.items():
                    for s, d in tr["positive"]:
                        i = idx_s[s]
                        j = idx_t[d]
                        n_gt += 1
                        order = np.lexsort((np.arange(m), Cx[i]))
                        r = int(np.where(order == j)[0][0] + 1)
                        gt_rank_rows.append(r)
                        if r <= 5 and int(np.where(np.lexsort((np.arange(n), Cx[:, j])) == i)[0][0] + 1) <= 5:
                            gt_mutual += 1
                loco_rows.append({
                    "bridge": bridge, "seed": seed, "variant": vname,
                    "gt_row_rank_mean": float(np.mean(gt_rank_rows)),
                    "gt_mutual_top5_retention": gt_mutual / max(n_gt, 1),
                    "d4_precision": summ["edge_precision"], "d4_recall": summ["edge_recall"],
                    "d4_f1": summ["edge_f1"], "fp_per_template": summ["edge_fp_total"] / 48,
                    "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                })
            print(f"[forensic] {bridge} seed {seed} done", flush=True)

    # write + aggregate
    fore = pd.DataFrame(forensic_rows)
    fore.to_csv(CTD / "raw" / "gt_vs_confuser_forensic.csv", index=False)
    fore_agg = fore.groupby(["bridge"]).agg(
        n_gt_edges=("gt_total_cost", "count"),
        frac_confuser_cheaper=("cost_margin", lambda s: float((s < 0).mean())),
        median_cost_margin=("cost_margin", "median"), mean_cost_margin=("cost_margin", "mean"),
        **{f"frac_gt_loses_{k}": (f"margin_{k}", lambda s, k=k: float((s < 0).mean()))
           for k in COMP_KEYS},
        **{f"median_margin_{k}": (f"margin_{k}", "median") for k in COMP_KEYS},
    ).reset_index()
    fore_agg.to_csv(CTD / "cost" / "gt_vs_confuser_aggregated.csv", index=False)
    # split vs merge breakdown
    role_agg = fore.groupby(["bridge", "structural_role"]).agg(
        frac_confuser_cheaper=("cost_margin", lambda s: float((s < 0).mean())),
        median_cost_margin=("cost_margin", "median"),
        **{f"frac_gt_loses_{k}": (f"margin_{k}", lambda s, k=k: float((s < 0).mean()))
           for k in COMP_KEYS}).reset_index()
    role_agg.to_csv(CTD / "cost" / "gt_vs_confuser_by_role.csv", index=False)
    cls_agg = fore.groupby(["bridge", "confuser_class"]).agg(
        n=("gt_total_cost", "count"), median_cost_margin=("cost_margin", "median"),
        frac_cheaper=("cost_margin", lambda s: float((s < 0).mean()))).reset_index()
    cls_agg.to_csv(CTD / "cost" / "confuser_class_breakdown.csv", index=False)

    sc = pd.DataFrame(scale_rows)
    sc.to_csv(CTD / "raw" / "feature_scale.csv", index=False)
    sc_agg = sc.groupby(["bridge", "component"]).mean(numeric_only=True).reset_index()
    sc_agg.to_csv(CTD / "feature_scale" / "feature_scale_aggregated.csv", index=False)

    am = pd.DataFrame(amount_rows)
    am.to_csv(CTD / "raw" / "amount_semantics.csv", index=False)
    am_agg = am.groupby(["bridge", "role"]).agg(
        n=("parent_amount", "count"),
        mean_children_sum_over_parent=("children_sum", lambda s: float((s / am.loc[s.index, "parent_amount"]).mean())),
        mean_parent_to_child_amount_cost=("parent_to_child_cost", "mean"),
    ).reset_index()
    if "child_cost_gt_mean" in am.columns:
        am_agg2 = am.groupby(["bridge", "role"]).agg(
            mean_gt_child_cost=("child_cost_gt_mean", "mean"),
            mean_full_confuser_cost=("full_confuser_cost_mean", "mean")).reset_index()
        am_agg = am_agg.merge(am_agg2, on=["bridge", "role"], how="left")
    am_agg.to_csv(CTD / "amount_semantics" / "amount_semantics_aggregated.csv", index=False)
    # marginal encodes amount?
    marg_rows = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_dev_cell(bridge, seed)
            flows = json.loads((cell["root"] / "flows.json").read_text(encoding="utf-8"))
            amt_s = np.array([float(f.get("amount_usd", 0.0)) for f in flows["src"]])
            amt_t = np.array([float(f.get("amount_usd", 0.0)) for f in flows["dst"]])
            marg_rows.append({
                "bridge": bridge, "seed": seed,
                "corr_a_amount": float(np.corrcoef(cell["a_rw"], amt_s)[0, 1])
                if amt_s.std() > 0 and cell["a_rw"].std() > 0 else float("nan"),
                "corr_b_amount": float(np.corrcoef(cell["b_ev"], amt_t)[0, 1])
                if amt_t.std() > 0 and cell["b_ev"].std() > 0 else float("nan"),
            })
    pd.DataFrame(marg_rows).to_csv(CTD / "amount_semantics" / "marginal_amount_correlation.csv",
                                   index=False)

    lo = pd.DataFrame(loco_rows)
    lo.to_csv(CTD / "raw" / "loco.csv", index=False)
    lo_agg = lo.groupby(["bridge", "variant"]).mean(numeric_only=True).reset_index()
    lo_agg.to_csv(CTD / "component_ablation" / "loco_aggregated.csv", index=False)
    print(lo_agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(fore_agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


def _classify(conf_role: str, conf_tpl: str, gt_tpl: str) -> str:
    if conf_tpl != gt_tpl:
        return "cross_template"
    if "merge_dst" in conf_role:
        return "same_template_merge_dst"
    if "split_" in conf_role and conf_role.endswith(("_a", "_b")):
        return "same_template_split_child"
    if "noise_dst" in conf_role:
        return "same_template_noise_dst"
    if "hidden" in conf_role:
        return "same_template_hidden_dst"
    return f"other_{conf_role}"


def _col_ranks(C: np.ndarray) -> np.ndarray:
    R = np.zeros_like(C, dtype=int)
    for j in range(C.shape[1]):
        order = np.lexsort((np.arange(C.shape[0]), C[:, j]))
        R[order, j] = np.arange(1, C.shape[0] + 1)
    return R


if __name__ == "__main__":
    raise SystemExit(main())
