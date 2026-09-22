"""ONE-SHOT holdout runner — complete execution body + dev preflight mode.

GUARDS (evaluated BEFORE any data access):
  - explicit --execute-holdout flag required for the holdout path;
  - preregistration hash verification (candidate + all prereg documents + this script);
  - frozen config identity (k=5, reg=0.05, reg_m=0.5, weights, bridges, methods).
Any failure ABORTS before holdout access.

--preflight-dev is STRUCTURALLY incapable of touching seeds 301-305: it only reads the
existing development artifacts (201-205) and reproduces the known development anchors.
It is SOFTWARE VALIDATION ONLY — never candidate selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from holdout.holdout_common import (  # noqa: E402
    AF, BRIDGES, CP, DEV_SEEDS, EXPECTED_DEV_ANCHORS, HOLD_SEEDS, METHODS, PRE, REPO,
    TDS, all_method_edges, conditional_edges, evaluate_gates, identity_check,
    paired_bootstrap_per_bridge, rank_desc, sha256, verify_prereg_hashes,
)
from decoder_audit.da_common import evaluate_edges  # noqa: E402

RESULT_ROOT = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_holdout_results"


def _cell_from_existing_dev(bridge: str, seed: int) -> dict[str, Any]:
    from baseline_mechanism.common import tpl_maps, truth_structure
    from dev_candidate.af_common import load_dev_cell
    cell = load_dev_cell(bridge, seed)
    c = np.load(AF / "plans" / bridge / f"seed_{seed}" / "costs.npz", allow_pickle=False)
    u = np.load(AF / "plans" / bridge / f"seed_{seed}" / "uot_primary.npz", allow_pickle=False)
    b = np.load(AF / "plans" / bridge / f"seed_{seed}" / "bot_primary.npz", allow_pickle=False)
    cell["C_primary"] = np.asarray(c["C_primary"], dtype=float)
    cell["P_uot"] = np.asarray(u["P"], dtype=float)
    cell["P_bot"] = np.asarray(b["P"], dtype=float)
    cell["truth"] = truth_structure(cell["labels"])
    return cell


def _build_holdout_cell(bridge: str, seed: int) -> dict[str, Any]:
    """Generate ONE holdout cell with the exact faithful generator + frozen pipeline.
    Called ONLY behind --execute-holdout."""
    from baseline_mechanism.common import FROZEN, FROZEN_PARAMS, N_TEMPLATES, tpl_maps, truth_structure
    from cross.domain.evaluation.semi_synthetic_flows import build_semi_synthetic_from_flow_labels
    from cross.domain.evaluation.synthetic_segment_subgraph import write_synthetic_subgraph_segment_csvs
    from cross.domain.labels.uot_flow_loader import flows_from_segment_export_csv
    from cross.domain.uot.cost_matrix import build_cost_matrix_decomposed, default_cost_weights
    from cross.domain.uot.uot_solver import _evidence_weighted_target_mass, _risk_weighted_source_mass
    from dev_candidate.af_common import build_amount_free_costs
    from diag.ctd_common import solve_bot_log, solve_uot_log

    root = RESULT_ROOT / "cells" / bridge / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    pool = FROZEN / "feature_stats" / bridge
    labels_pool = pd.read_csv(pool / "flow_labels.csv", dtype=str, keep_default_na=False)
    labels_pool.to_csv(root / "flow_labels_pool.csv", index=False)
    stats_path = root / "flow_label_stats.json"
    stats_path.write_text(json.dumps({"predominantly_one_to_one": True,
                                      "n_pool_pairs": int(len(labels_pool))}), encoding="utf-8")
    build_semi_synthetic_from_flow_labels(root / "flow_labels_pool.csv", stats_path, root,
                                          seed=seed, max_seeds=N_TEMPLATES, force=True)
    hints = json.loads((root / "labels" / "synthetic_uot_eval_metrics.json").read_text(encoding="utf-8"))
    clones = hints.get("segment_clone_records") or []
    se = root / "flow_segments_eth_synth.csv"
    sb = root / "flow_segments_bnb_synth.csv"
    write_synthetic_subgraph_segment_csvs(pool / "flow_segments_eth.csv",
                                          pool / "flow_segments_bnb.csv", clones, se, sb)
    eth = flows_from_segment_export_csv(se, chain="ETH")
    bnb = flows_from_segment_export_csv(sb, chain="BNB")
    decomp = build_cost_matrix_decomposed(
        eth, bnb, weights=default_cost_weights(), use_graph=False,
        max_delay_sec=FROZEN_PARAMS["uot_max_delay_sec"],
        causal_violation_penalty=FROZEN_PARAMS["uot_causal_violation_penalty"])
    C_full = np.maximum(np.asarray(decomp["C"], dtype=float)
                        + np.asarray(decomp["bridge_prior_bonus"], dtype=float), 0.0)
    components = {k: np.asarray(decomp[k], dtype=float) for k in
                  ("time_cost", "route_cost", "risk_cost", "evidence_cost",
                   "address_novelty_cost")}
    C_primary = build_amount_free_costs(components)["primary"]
    sids = [str(f.get("flow_id")) for f in eth]
    tids = [str(f.get("flow_id")) for f in bnb]
    a0, a_rw = _risk_weighted_source_mass(eth, lambda_risk=FROZEN_PARAMS["uot_lambda_risk"])
    b0, b_ev = _evidence_weighted_target_mass(bnb)
    u = solve_uot_log(a_rw, b_ev, C_primary, FROZEN_PARAMS["uot_reg"], FROZEN_PARAMS["uot_reg_m"])
    bot = solve_bot_log(a_rw, b_ev, C_primary, FROZEN_PARAMS["uot_reg"])
    labels = pd.read_csv(root / "labels" / "synthetic_flow_labels.csv", dtype=str,
                         keep_default_na=False)
    np.savez(root / "grid.npz", C_primary=C_primary, P_uot=u["P"], P_bot=bot["P"],
             a_rw=a_rw, b_ev=b_ev)
    (root / "solver.json").write_text(json.dumps({
        "uot_converged": u["converged"], "uot_final_err": u["final_err"],
        "bot_converged": bot["converged"],
        "bot_row_residual": bot["row_residual"], "bot_col_residual": bot["col_residual"],
    }, indent=2) + "\n", encoding="utf-8")
    labels.to_csv(root / "labels.csv", index=False)
    return {
        "bridge": bridge, "seed": seed, "sids": sids, "tids": tids,
        "C_primary": C_primary, "P_uot": u["P"], "P_bot": bot["P"],
        "a_rw": a_rw, "b_ev": b_ev, "labels": labels,
        "truth": truth_structure(labels), "root": root,
        "solver": {"uot_converged": u["converged"], "uot_final_err": u["final_err"],
                   "bot_converged": bot["converged"],
                   "bot_row_residual": bot["row_residual"], "bot_col_residual": bot["col_residual"]},
    }


def _compute_cell(cell: dict[str, Any], out_root: Path) -> dict[str, Any]:
    np.savez(out_root / "cell_inputs.npz", C_primary=cell["C_primary"],
             P_uot=cell["P_uot"], P_bot=cell["P_bot"],
             sids=np.array(cell["sids"], dtype=object),
             tids=np.array(cell["tids"], dtype=object))
    cell["labels"].to_csv(out_root / "labels.csv", index=False)
    edges_by_method = all_method_edges(cell)
    tpl_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {"bridge": cell["bridge"], "seed": cell["seed"]}
    for name in METHODS:
        edges = edges_by_method[name]
        df, summ = evaluate_edges(cell, edges)
        df["bridge"] = cell["bridge"]
        df["seed"] = cell["seed"]
        df["method"] = name
        tpl_rows.append(df)
        pd.DataFrame([{"src_flow_id": s, "dst_flow_id": d} for s, d in edges]).to_csv(
            out_root / f"edges_{name}.csv", index=False)
        per_seed[name] = {
            "edge_f1": summ["edge_f1"], "edge_precision": summ["edge_precision"],
            "edge_recall": summ["edge_recall"], "fp_per_template": summ["edge_fp_total"] / 48,
            "fn_per_template": summ["edge_fn_total"] / 48,
        }
    pd.concat(tpl_rows, ignore_index=True).to_csv(out_root / "per_template.csv", index=False)
    # mechanism rows: kernel vs plan mutual-top5 AND directional row/col top-5 for GT edges
    K = np.exp(-cell["C_primary"] / 0.05)
    rr_k = rank_desc(K, 1)
    cr_k = rank_desc(K, 0)
    rr_raw = rank_desc(cell["P_uot"], 1)
    cr_raw = rank_desc(cell["P_uot"], 0)
    S_row, S_col = _cond_scores(cell["P_uot"])
    rr_cond = rank_desc(S_row, 1)
    cr_cond = rank_desc(S_col, 0)
    gt = []
    for t, tr in cell["truth"].items():
        for s, d in tr["positive"]:
            gt.append((cell["sids"].index(s), cell["tids"].index(d)))
    gt = list(set(gt))
    mech_rows = []
    for i, j in gt:
        s, d = cell["sids"][i], cell["tids"][j]
        gt_role = None
        for t, tr in cell["truth"].items():
            if (s, d) in tr["split"]:
                gt_role = "split"
            elif (s, d) in tr["merge"]:
                gt_role = "merge"
            elif (s, d) in tr["decoy"]:
                gt_role = "decoy"
        dst_role = d.split("__")[-1]
        mech_rows.append({
            "gt_role": gt_role, "dst_role": dst_role,
            "cost_row5": int(rr_k[i, j] <= 5), "cost_col5": int(cr_k[i, j] <= 5),
            "raw_row5": int(rr_raw[i, j] <= 5), "raw_col5": int(cr_raw[i, j] <= 5),
            "cond_row5": int(rr_cond[i, j] <= 5), "cond_col5": int(cr_cond[i, j] <= 5),
            "cost_mutual5": int(rr_k[i, j] <= 5 and cr_k[i, j] <= 5),
            "raw_mutual5": int(rr_raw[i, j] <= 5 and cr_raw[i, j] <= 5),
            "cond_mutual5": int(rr_cond[i, j] <= 5 and cr_cond[i, j] <= 5),
        })
    mech = pd.DataFrame(mech_rows)
    mech.to_csv(out_root / "mechanism.csv", index=False)
    return {"per_seed": per_seed, "n_templates": len(cell["truth"])}


def _cond_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S_row = np.zeros_like(P)
    S_col = np.zeros_like(P)
    S_row[:, c > 0] = P[:, c > 0] / c[None, c > 0]
    S_col[r > 0, :] = P[r > 0, :] / r[r > 0, None]
    return S_row, S_col


def _statistics(per_seed_rows: list[dict[str, Any]], mech_frames: list[pd.DataFrame],
                solver_cells: list[dict[str, Any]], cells_root: Path) -> dict[str, Any]:
    ps = pd.DataFrame(per_seed_rows)
    macro = {}
    for m in METHODS:
        bm = ps[ps["method"] == m].groupby("bridge")["edge_f1"].mean()
        macro[m] = float(bm.mean())
    keys = ["bridge", "seed", "template_id"]
    tp = pd.concat([pd.read_csv(p) for p in sorted(cells_root.rglob("per_template.csv"))],
                   ignore_index=True)
    def _paired(a: str, b: str) -> dict[str, Any]:
        fa = tp[tp["method"] == a][keys + ["edge_f1"]].rename(columns={"edge_f1": "x"})
        fb = tp[tp["method"] == b][keys + ["edge_f1"]].rename(columns={"edge_f1": "y"})
        m = fa.merge(fb, on=keys)
        m["d"] = m["x"] - m["y"]
        by_bridge = {b2: g["d"].to_numpy(dtype=float) for b2, g in m.groupby("bridge")}
        return paired_bootstrap_per_bridge(by_bridge)
    d_primary = _paired("CONDITIONAL_UOT_D4", "RAW_UOT_PLAN_D4")
    d_cost = _paired("CONDITIONAL_UOT_D4", "AMOUNT_FREE_COST_D4")
    d_support = _paired("CONDITIONAL_UOT_D4", "SUPPORT_PLUS_K_D4")
    d_bot = _paired("CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4")
    # mechanism macro directions (cond - raw; harmful rates should DECREASE -> negative)
    mech = pd.concat(mech_frames, ignore_index=True)
    def harmful(dim: str, col: str) -> float:
        sub = mech[mech[f"cost_{dim}"] == 1]
        return float((sub[col] == 0).mean()) if len(sub) else float("nan")
    def harmful_role(col: str, role: str) -> float:
        sub = mech[(mech["cost_mutual5"] == 1) & (mech["gt_role"] == role)]
        return float((sub[col] == 0).mean()) if len(sub) else float("nan")
    mechanism = {
        "row_harmful_delta": harmful("row5", "cond_row5") - harmful("row5", "raw_row5"),
        "col_harmful_delta": harmful("col5", "cond_col5") - harmful("col5", "raw_col5"),
        "split_child_delta": harmful_role("cond_mutual5", "split") - harmful_role("raw_mutual5", "split"),
        "merge_dst_delta": harmful_role("cond_mutual5", "merge") - harmful_role("raw_mutual5", "merge"),
        "retention_delta": float((mech["cond_mutual5"] == 1).mean()
                                 - (mech["raw_mutual5"] == 1).mean()),
    }
    solver = {"all_converged": bool(all(c["solver"]["uot_converged"] and c["solver"]["bot_converged"]
                                        for c in solver_cells)),
              "n_cells": len(solver_cells)}
    stats = {
        "macro_f1": macro,
        "d_primary": d_primary, "d_cost": d_cost, "d_support": d_support, "d_bot": d_bot,
        "fp_raw": float(ps[ps["method"] == "RAW_UOT_PLAN_D4"]["fp_per_template"].mean()),
        "fp_cond": float(ps[ps["method"] == "CONDITIONAL_UOT_D4"]["fp_per_template"].mean()),
        "fn_raw": float(ps[ps["method"] == "RAW_UOT_PLAN_D4"]["fn_per_template"].mean()),
        "fn_cond": float(ps[ps["method"] == "CONDITIONAL_UOT_D4"]["fn_per_template"].mean()),
        "mechanism": mechanism, "solver": solver,
    }
    stats["decision"] = evaluate_gates(stats, mechanism, solver)
    return stats


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-holdout", action="store_true",
                    help="REQUIRED for the holdout path; refuses otherwise")
    ap.add_argument("--preflight-dev", action="store_true",
                    help="software validation on development artifacts 201-205 only")
    cli = ap.parse_args()

    errors = verify_prereg_hashes() + identity_check({"seeds": HOLD_SEEDS})
    if errors:
        for e in errors:
            print(f"[GATE-FAIL] {e}", flush=True)
        raise SystemExit("ABORT BEFORE HOLDOUT ACCESS: preregistration/config identity failed")

    if cli.execute_holdout and cli.preflight_dev:
        raise SystemExit("refusing: mutually exclusive modes")

    if cli.execute_holdout:
        if RESULT_ROOT.exists():
            raise SystemExit("refusing: holdout result directory already exists "
                             "(one-shot run only)")
        RESULT_ROOT.mkdir(parents=True)
        per_seed_rows, mech_frames, solver_cells = [], [], []
        for bridge in BRIDGES:
            for seed in HOLD_SEEDS:
                cell = _build_holdout_cell(bridge, seed)
                out = RESULT_ROOT / "cells" / bridge / f"seed_{seed}"
                r = _compute_cell(cell, out)
                r["per_seed"]["method"] = None
                for name in METHODS:
                    per_seed_rows.append({"bridge": bridge, "seed": seed, "method": name,
                                          **r["per_seed"][name]})
                mech_frames.append(pd.read_csv(out / "mechanism.csv"))
                solver_cells.append(cell)
        stats = _statistics(per_seed_rows, mech_frames, solver_cells, RESULT_ROOT)
        (RESULT_ROOT / "statistics.json").write_text(json.dumps(stats, indent=2, default=str)
                                                     + "\n", encoding="utf-8")
        print(json.dumps(stats["decision"], indent=2, default=str))
        return 0

    if cli.preflight_dev:
        import shutil
        tmp = PRE / "preflight_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        per_seed_rows, mech_frames, solver_cells = [], [], []
        for bridge in BRIDGES:
            for seed in DEV_SEEDS:
                cell = _cell_from_existing_dev(bridge, seed)
                out = tmp / "cells" / bridge / f"seed_{seed}"
                out.mkdir(parents=True, exist_ok=True)
                r = _compute_cell(cell, out)
                for name in METHODS:
                    per_seed_rows.append({"bridge": bridge, "seed": seed, "method": name,
                                          **r["per_seed"][name]})
                mech_frames.append(pd.read_csv(out / "mechanism.csv"))
                solver_cells.append({"solver": {"uot_converged": True, "bot_converged": True}})
        stats = _statistics(per_seed_rows, mech_frames, solver_cells, tmp)
        macro = stats["macro_f1"]
        report = {"reproduced_macro_f1": macro, "anchors": EXPECTED_DEV_ANCHORS}
        # NOTE: the anchors are rounded to 4 decimals in the development report; the
        # validation tolerance is therefore 5e-4 (software check only, not a scientific
        # threshold).
        worst = max(abs(macro[m] - EXPECTED_DEV_ANCHORS[m]) for m in METHODS)
        report["max_abs_deviation_from_anchor"] = float(worst)
        report["anchor_tolerance"] = 5e-4
        report["preflight_pass"] = bool(worst < 5e-4)
        report["delta_primary_macro"] = stats["d_primary"]["macro_mean"]
        report["decision"] = stats["decision"]
        (PRE / "preflight_report.json").write_text(json.dumps(report, indent=2, default=str)
                                                   + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["preflight_pass"] else 1

    print("REFUSING: holdout execution requires explicit human approval "
          "(--execute-holdout); this run entered the GUARD PATH only.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
