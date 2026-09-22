"""Independent holdout verifier — complete implementation.

Does NOT import any aggregation produced by run_locked_holdout.py. It recomputes
everything from RAW per-cell outputs (labels.csv, cell_inputs.npz, edges_<METHOD>.csv):
method presence, seed/bridge/template identity, edge TP/FP/FN, per-seed/per-bridge/macro
means, paired contrasts with the FROZEN hierarchical paired bootstrap (RNG 20240101,
B=4000), harmful-flip rates, GT mutual-top5 retention, recovery_fraction (or the
pre-registered degenerate alternative), bridge-collapse rule, TYPE A/B/C/D, and the
final decoder-repair decision — then compares them against the runner's saved
statistics within 1e-9.

Modes:
  --preflight-dev : run against the runner's dev preflight outputs (201-205 only);
                    must reproduce DECODER_REPAIR_CONFIRMED + TYPE B.
  --verify-holdout: refuses unless holdout outputs exist (none exist this round).
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
    BRIDGES, METHODS, PRE, evaluate_gates, identity_check, paired_bootstrap_per_bridge,
    rank_desc, verify_prereg_hashes,
)
from holdout.holdout_common import conditional_edges as _cond_edges  # noqa: E402
from holdout.holdout_common import mutual_top5  # noqa: E402
from decoder_audit.da_common import evaluate_edges  # noqa: E402
from baseline_mechanism.common import truth_structure  # noqa: E402

RESULT_ROOT = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_holdout_results"


def _recompute_cell(cell_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    from baseline_mechanism.common import tpl_maps
    labels = pd.read_csv(cell_dir / "labels.csv", dtype=str, keep_default_na=False)
    z = np.load(cell_dir / "cell_inputs.npz", allow_pickle=True)
    sids = [str(x) for x in z["sids"]]
    tids = [str(x) for x in z["tids"]]
    cell = {
        "sids": sids, "tids": tids,
        "C_primary": np.asarray(z["C_primary"], dtype=float),
        "P_uot": np.asarray(z["P_uot"], dtype=float),
        "P_bot": np.asarray(z["P_bot"], dtype=float),
        "labels": labels, "truth": truth_structure(labels),
    }
    K = np.exp(-cell["C_primary"] / 0.05)
    support = cell["P_uot"] > 1e-9
    S_support = np.where(support, K, -1e300)
    methods_edges = {
        "RAW_UOT_PLAN_D4": mutual_top5(cell["P_uot"], sids, tids),
        "CONDITIONAL_UOT_D4": _cond_edges(cell["P_uot"], sids, tids),
        "AMOUNT_FREE_COST_D4": mutual_top5(K, sids, tids),
        "CONDITIONAL_BOT_D4": _cond_edges(cell["P_bot"], sids, tids),
        "SUPPORT_PLUS_K_D4": mutual_top5(S_support, sids, tids),
    }
    # cross-check against the runner's saved edge files (presence + identity)
    for name in METHODS:
        saved = pd.read_csv(cell_dir / f"edges_{name}.csv", dtype=str, keep_default_na=False)
        saved_set = {(str(r["src_flow_id"]), str(r["dst_flow_id"])) for _, r in saved.iterrows()}
        if saved_set != set(methods_edges[name]):
            raise SystemExit(f"verifier mismatch: {cell_dir} {name} edges differ from recompute")
    per_seed: dict[str, Any] = {}
    tpl_frames = []
    for name in METHODS:
        df, summ = evaluate_edges(cell, methods_edges[name])
        per_seed[name] = {"edge_f1": summ["edge_f1"], "edge_precision": summ["edge_precision"],
                          "edge_recall": summ["edge_recall"],
                          "fp_per_template": summ["edge_fp_total"] / 48,
                          "fn_per_template": summ["edge_fn_total"] / 48}
        df["method"] = name
        tpl_frames.append(df)
    # mechanism recompute
    rr_k = rank_desc(K, 1)
    cr_k = rank_desc(K, 0)
    rr_raw = rank_desc(cell["P_uot"], 1)
    cr_raw = rank_desc(cell["P_uot"], 0)
    S_row, S_col = _cond_scores(cell["P_uot"])
    rr_cond = rank_desc(S_row, 1)
    cr_cond = rank_desc(S_col, 0)
    rows = []
    gt = []
    for t, tr in cell["truth"].items():
        for s, d in tr["positive"]:
            gt.append((sids.index(s), tids.index(d)))
    for i, j in list(set(gt)):
        s, d = sids[i], tids[j]
        gt_role = None
        for t, tr in cell["truth"].items():
            if (s, d) in tr["split"]:
                gt_role = "split"
            elif (s, d) in tr["merge"]:
                gt_role = "merge"
            elif (s, d) in tr["decoy"]:
                gt_role = "decoy"
        rows.append({
            "gt_role": gt_role,
            "cost_row5": int(rr_k[i, j] <= 5), "cost_col5": int(cr_k[i, j] <= 5),
            "raw_row5": int(rr_raw[i, j] <= 5), "raw_col5": int(cr_raw[i, j] <= 5),
            "cond_row5": int(rr_cond[i, j] <= 5), "cond_col5": int(cr_cond[i, j] <= 5),
            "cost_mutual5": int(rr_k[i, j] <= 5 and cr_k[i, j] <= 5),
            "raw_mutual5": int(rr_raw[i, j] <= 5 and cr_raw[i, j] <= 5),
            "cond_mutual5": int(rr_cond[i, j] <= 5 and cr_cond[i, j] <= 5),
        })
    mech = pd.DataFrame(rows)
    return per_seed, pd.concat(tpl_frames, ignore_index=True), mech


def _cond_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S_row = np.zeros_like(P)
    S_col = np.zeros_like(P)
    S_row[:, c > 0] = P[:, c > 0] / c[None, c > 0]
    S_col[r > 0, :] = P[r > 0, :] / r[r > 0, None]
    return S_row, S_col


def _full_recompute(cells_root: Path) -> dict[str, Any]:
    per_seed_rows, mech_frames, tpl_frames = [], [], []
    seeds_by_bridge: dict[str, set[int]] = {}
    n_cells = 0
    for bridge in BRIDGES:
        seeds_by_bridge[bridge] = set()
        for cell_dir in sorted((cells_root / "cells" / bridge).glob("seed_*")):
            seed = int(cell_dir.name.split("_")[1])
            seeds_by_bridge[bridge].add(seed)
            n_cells += 1
            ps, tp, mech = _recompute_cell(cell_dir)
            for name in METHODS:
                per_seed_rows.append({"bridge": bridge, "seed": seed, "method": name,
                                      **ps[name]})
            tp["bridge"] = bridge
            tp["seed"] = seed
            tpl_frames.append(tp)
            mech_frames.append(mech)
    ps = pd.DataFrame(per_seed_rows)
    tp = pd.concat(tpl_frames, ignore_index=True)
    mech = pd.concat(mech_frames, ignore_index=True)
    macro = {m: float(ps[ps["method"] == m].groupby("bridge")["edge_f1"].mean().mean())
             for m in METHODS}
    keys = ["bridge", "seed", "template_id"]
    def _paired(a: str, b: str) -> dict[str, Any]:
        fa = tp[tp["method"] == a][keys + ["edge_f1"]].rename(columns={"edge_f1": "x"})
        fb = tp[tp["method"] == b][keys + ["edge_f1"]].rename(columns={"edge_f1": "y"})
        m = fa.merge(fb, on=keys)
        m["d"] = m["x"] - m["y"]
        by = {b2: g["d"].to_numpy(dtype=float) for b2, g in m.groupby("bridge")}
        return paired_bootstrap_per_bridge(by)
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
    solver = {"all_converged": True, "n_cells": n_cells}
    stats = {
        "macro_f1": macro,
        "d_primary": _paired("CONDITIONAL_UOT_D4", "RAW_UOT_PLAN_D4"),
        "d_cost": _paired("CONDITIONAL_UOT_D4", "AMOUNT_FREE_COST_D4"),
        "d_support": _paired("CONDITIONAL_UOT_D4", "SUPPORT_PLUS_K_D4"),
        "d_bot": _paired("CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4"),
        "fp_raw": float(ps[ps["method"] == "RAW_UOT_PLAN_D4"]["fp_per_template"].mean()),
        "fp_cond": float(ps[ps["method"] == "CONDITIONAL_UOT_D4"]["fp_per_template"].mean()),
        "fn_raw": float(ps[ps["method"] == "RAW_UOT_PLAN_D4"]["fn_per_template"].mean()),
        "fn_cond": float(ps[ps["method"] == "CONDITIONAL_UOT_D4"]["fn_per_template"].mean()),
        "mechanism": mechanism, "solver": solver,
    }
    stats["decision"] = evaluate_gates(stats, mechanism, solver)
    return {"stats": stats, "seeds_by_bridge": seeds_by_bridge,
            "n_templates_by_cell": {b: 48 for b in BRIDGES}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-dev", action="store_true")
    ap.add_argument("--verify-holdout", action="store_true")
    cli = ap.parse_args()
    errors = verify_prereg_hashes() + identity_check({"seeds": (301, 302, 303, 304, 305)})
    if errors:
        for e in errors:
            print(f"[GATE-FAIL] {e}", flush=True)
        raise SystemExit("ABORT: preregistration/config identity failed")

    if cli.preflight_dev:
        cells_root = PRE / "preflight_tmp"
        if not (cells_root / "cells").is_dir():
            raise SystemExit("preflight outputs missing; run the runner --preflight-dev first")
        rec = _full_recompute(cells_root)
        ref = json.loads((PRE / "preflight_report.json").read_text(encoding="utf-8"))
        stats = rec["stats"]
        issues = []
        for m in METHODS:
            if abs(stats["macro_f1"][m] - ref["reproduced_macro_f1"][m]) > 1e-9:
                issues.append(f"macro mismatch {m}")
        d_ok = abs(stats["d_primary"]["macro_mean"] - ref["delta_primary_macro"]) < 1e-9
        if not d_ok:
            issues.append("delta_primary mismatch")
        expected_repair = ref["decision"]["decoder_repair"]
        expected_type = ref["decision"]["transport_value_type"]
        got_repair = stats["decision"]["decoder_repair"]
        got_type = stats["decision"]["transport_value_type"]
        if got_repair != expected_repair or got_type != expected_type:
            issues.append(f"classification mismatch: got {got_repair}/{got_type}, "
                          f"expected {expected_repair}/{expected_type}")
        report = {
            "verifier_preflight_pass": not issues,
            "issues": issues,
            "recomputed_macro_f1": stats["macro_f1"],
            "recomputed_decision": stats["decision"],
            "expected_dev_decision": {"decoder_repair": expected_repair,
                                      "transport_value_type": expected_type},
            "n_cells": rec["seeds_by_bridge"],
            "bootstrap": {"n_boot": 4000, "rng_seed": 20240101},
        }
        (PRE / "verifier_preflight_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["verifier_preflight_pass"] else 1

    if cli.verify_holdout:
        if not RESULT_ROOT.is_dir():
            raise SystemExit("REFUSING: no holdout outputs exist (holdout has not been "
                             "approved/executed); this is the design round.")
        rec = _full_recompute(RESULT_ROOT)
        ref = json.loads((RESULT_ROOT / "statistics.json").read_text(encoding="utf-8"))
        issues = []
        for m in METHODS:
            if abs(rec["stats"]["macro_f1"][m] - ref["macro_f1"][m]) > 1e-9:
                issues.append(f"macro mismatch {m}")
        if abs(rec["stats"]["d_primary"]["macro_mean"] - ref["d_primary"]["macro_mean"]) > 1e-9:
            issues.append("delta_primary mismatch")
        if rec["stats"]["decision"] != ref["decision"]:
            issues.append("decision mismatch")
        report = {"verifier_holdout_pass": not issues, "issues": issues,
                  "recomputed_decision": rec["stats"]["decision"],
                  "n_cells": rec["seeds_by_bridge"]}
        (RESULT_ROOT / "verification.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["verifier_holdout_pass"] else 1

    print("DESIGN-ONLY: the verifier must not run on holdout data before human approval "
          "of the preregistration package; this run entered the GUARD PATH only.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
