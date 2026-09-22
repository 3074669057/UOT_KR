"""One-shot runner for the real-anchor external validation (DESIGN-ONLY round).

Guard hierarchy (all enforced here, all negative-tested):
  1. GUARD PATH by default: no flag -> refuse (design-only round).
  2. --preflight-toy: full pipeline on toy fixtures only; allowed anytime.
  3. --execute-real-anchor-validation: requires LOCK_STATUS == approved, hash-gate
     PASS, dataset identity PASS, isolation scan PASS, result dir absent.
This script is self-contained: it never imports the 301-305 holdout package and
refuses any path/config containing holdout artifacts or seed ids 301-305.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tifs_external.real_anchor_common import (
    FROZEN, cluster_bootstrap, conditional_edges, evaluate_components, mutual_top5,
    one_to_one_edges, scan_forbidden, threshold_mm_edges, verify_hash_manifest,
)

REPO = Path(__file__).resolve().parents[3]
PKG_V1 = REPO / "out" / "multi_bridge_expansion" / "tifs_real_anchor_external_validation_preregistration"  # archived record
PKG = REPO / "out" / "multi_bridge_expansion" / "tifs_real_anchor_external_validation_preregistration_v2"
LOCK_FILE = PKG / "LOCK_STATUS.md"
MANIFEST = PKG / "HASH_MANIFEST_v2.json"
RESULT_ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_real_anchor_external_validation_results"
LABELS = REPO / "out" / "paper_full_pipeline_run" / "label_layer_v1" / "flow_labels.csv"

EXPECTED_IDENTITY = {
    "n_rows": 7128, "n_one_to_one": 5155, "n_fanout_rows": 1961, "n_merge_rows": 12,
    "n_fanout_components": 568, "n_fanout_degree_le_5": 508, "n_fanout_degree_gt_5": 60,
    "n_merge_components": 6, "support_sum": 7296,
    "n_fanout_addresses": 5, "n_fanout_iso_weeks": 2,
    "boundary_audit_value": 1639015242,  # v1 cross-check value; retained for history
}

METHODS = ["RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4",
           "THRESHOLD_MM", "CONNECTOR_STYLE", "ABCTRACER_STYLE"]


def _read_lock_status() -> str:
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- Status:"):
            value = stripped.split(":", 1)[1].strip()
            return "approved" if value == "approved" else value
    return "unknown"


def _check_isolation(config_paths: list[str]) -> list[str]:
    hits = scan_forbidden(*config_paths, str(RESULT_ROOT), str(PKG), str(LABELS))
    return hits


def _identity_check() -> list[str]:
    errors: list[str] = []
    df = pd.read_csv(LABELS)
    fl = df
    errors += [] if len(fl) == EXPECTED_IDENTITY["n_rows"] else ["n_rows"]
    pat = fl["pattern_type"].value_counts().to_dict()
    if pat.get("one_to_one") != EXPECTED_IDENTITY["n_one_to_one"]:
        errors.append("n_one_to_one")
    if pat.get("many_to_one") != EXPECTED_IDENTITY["n_fanout_rows"]:
        errors.append("n_fanout_rows")
    if pat.get("one_to_many") != EXPECTED_IDENTITY["n_merge_rows"]:
        errors.append("n_merge_rows")
    if float(fl["support_tx_pair_count"].sum()) != EXPECTED_IDENTITY["support_sum"]:
        errors.append("support_sum")
    src_deg = fl.groupby("src_flow_id")["dst_flow_id"].nunique()
    dst_deg = fl.groupby("dst_flow_id")["src_flow_id"].nunique()
    fan_components = src_deg[src_deg >= 2]
    merge_components = dst_deg[dst_deg >= 2]
    if len(fan_components) != EXPECTED_IDENTITY["n_fanout_components"]:
        errors.append("n_fanout_components")
    if int((fan_components <= FROZEN["k"]).sum()) != EXPECTED_IDENTITY["n_fanout_degree_le_5"]:
        errors.append("n_fanout_degree_le_5")
    if int((fan_components > FROZEN["k"]).sum()) != EXPECTED_IDENTITY["n_fanout_degree_gt_5"]:
        errors.append("n_fanout_degree_gt_5")
    if len(merge_components) != EXPECTED_IDENTITY["n_merge_components"]:
        errors.append("n_merge_components")
    return errors


# --------------------------------------------------------------------------- #
# Toy fixture                                                                 #
# --------------------------------------------------------------------------- #

def build_toy() -> dict:
    """Deterministic toy cell: 8 sources x 16 destinations with planted structure.

    Exercises the v2 all-population primary: fan-out components of degree 2, 3 and 6
    (the degree-6 component stays in the primary population; the D4 decoders
    structurally cannot recover it exactly — known representational limit).
    Address clusters: A(s0), B(s1, s6), C(s5) -> 3 clusters for the bootstrap.
    """
    rng = np.random.RandomState(7)
    n, m = 8, 16
    base = rng.rand(n, m)
    gt = {
        "s0": {"dsts": ["d0", "d1"], "t0": 100.0, "addr": "A"},              # deg 2
        "s1": {"dsts": ["d2", "d3", "d4"], "t0": 200.0, "addr": "B"},        # deg 3
        "s2": {"dsts": ["d5"], "t0": 300.0, "addr": "A"},                    # deg 1 (not primary)
        "s3": {"dsts": ["d6"], "t0": 400.0, "addr": "B"},                    # deg 1
        "s4": {"dsts": ["d7"], "t0": 500.0, "addr": "A"},                    # deg 1
        "s5": {"dsts": [f"d{i}" for i in range(8, 14)], "t0": 600.0, "addr": "C"},  # deg 6
        "s6": {"dsts": ["d14", "d15"], "t0": 700.0, "addr": "B"},            # deg 2
        # s7: unmatched (no GT edges)
    }
    C = base.copy()
    for i, s in enumerate([f"s{i}" for i in range(n)]):
        for d in gt.get(s, {}).get("dsts", []):
            C[i, int(d[1:])] = 0.01
    a = np.ones(n) / n
    b = np.ones(m) / m
    return {"C": C, "a": a, "b": b, "gt": gt,
            "srcs": [f"s{i}" for i in range(n)], "dsts": [f"d{j}" for j in range(m)]}


def _run_cell(cell: dict, B: int) -> dict:
    import ot
    C, a, b = cell["C"], cell["a"], cell["b"]
    P_uot = ot.sinkhorn_unbalanced(a, b, C, FROZEN["reg"], FROZEN["reg_m"])
    P_bot = ot.sinkhorn(a, b, C, FROZEN["reg"])
    # all-population primary: EVERY fan-out component (degree >= 2), no k filter
    gt_primary = {s: v for s, v in cell["gt"].items() if len(v["dsts"]) >= 2}
    ordered = sorted(gt_primary, key=lambda s: gt_primary[s]["t0"])
    n_calib = max(int(np.floor(len(ordered) * FROZEN["tmm_calibration_frac"])), 1)
    calib_srcs = set(ordered[:n_calib])
    tau = _calibrate_tau(C, cell["gt"], calib_srcs)
    edges = {
        "RAW_UOT_PLAN_D4": mutual_top5(P_uot),
        "CONDITIONAL_UOT_D4": conditional_edges(P_uot),
        "CONDITIONAL_BOT_D4": conditional_edges(P_bot),
        "THRESHOLD_MM": threshold_mm_edges(C, tau),
        "CONNECTOR_STYLE": one_to_one_edges(C, "connector"),
        "ABCTRACER_STYLE": one_to_one_edges(C, "abctracer"),
    }
    sids = cell["srcs"]
    dids = cell["dsts"]
    str_edges = {m: {(sids[i], dids[j]) for (i, j) in e} for m, e in edges.items()}
    evals = {m: evaluate_components(gt_primary, str_edges[m]) for m in METHODS}
    values = {m: {s: v["per_component"][s]["recovered"] for s in gt_primary}
              for m, v in evals.items()}
    per = evals["CONDITIONAL_UOT_D4"]["per_component"]
    addr_ids = {"A": 0, "B": 1, "C": 2}
    clusters = {s: addr_ids[gt_primary[s]["addr"]] for s in gt_primary}
    per_addr = {}
    for s in gt_primary:
        per_addr.setdefault(gt_primary[s]["addr"], []).append(values["CONDITIONAL_UOT_D4"][s])
    stats = {
        "recovery_rate": {m: float(np.mean([values[m][s] for s in gt_primary])) for m in METHODS},
        "n_primary_components": len(gt_primary),
        "n_degree_gt_k_in_primary": sum(1 for s in gt_primary if len(gt_primary[s]["dsts"]) > FROZEN["k"]),
        "n_calib_components": len(calib_srcs),
        "per_address_recovery": {a: float(np.mean(v)) for a, v in per_addr.items()},
        "d_primary": cluster_bootstrap(per, "CONDITIONAL_UOT_D4", "RAW_UOT_PLAN_D4",
                                       values, clusters=clusters, B=B),
        "d_tmm": cluster_bootstrap(per, "CONDITIONAL_UOT_D4", "THRESHOLD_MM",
                                   values, clusters=clusters, B=B),
        "d_bot": cluster_bootstrap(per, "CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4",
                                   values, clusters=clusters, B=B),
        "edges": {m: sorted(list(str_edges[m])) for m in METHODS},
        "uot_max_rowdev": float(np.max(np.abs(P_uot.sum(axis=1) - a))),
        "uot_max_coldev": float(np.max(np.abs(P_uot.sum(axis=0) - b))),
    }
    return stats


def _calibrate_tau(C: np.ndarray, gt: dict, calib_srcs: set[str]) -> float:
    """Toy calibration: grid-quantile tau maximizing edge F1 on the CALIBRATION-slice
    GT only (v2 rule: chronologically earliest 25% of fan-out components)."""
    best_tau, best_f1 = 0.0, -1.0
    gt_edges = {(s, d) for s, v in gt.items() for d in v["dsts"] if s in calib_srcs}
    for q in FROZEN["tau_grid_quantiles"]:
        tau = float(np.quantile(C, q))
        pred = threshold_mm_edges(C, tau)
        tp = len(gt_edges & pred)
        f1 = 2 * tp / max(len(pred) + len(gt_edges), 1)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Real-anchor external validation runner")
    ap.add_argument("--execute-real-anchor-validation", action="store_true",
                    help="REQUIRED for the real path; refuses otherwise")
    ap.add_argument("--preflight-toy", action="store_true",
                    help="machinery validation on toy fixtures only (allowed anytime)")
    ap.add_argument("--boot-b", type=int, default=None)
    cli = ap.parse_args()

    if cli.execute_real_anchor_validation:
        # ---- real path: every gate must pass ----
        if _read_lock_status() != "approved":
            print("ABORT: LOCK_STATUS is not 'approved'. Design-only round; no execution.")
            return 3
        iso = _check_isolation([str(PKG), str(LABELS), str(RESULT_ROOT)])
        if iso:
            print(f"ABORT: forbidden holdout dependency tokens found: {iso}")
            return 4
        errs = verify_hash_manifest(MANIFEST, REPO)
        if errs:
            print("ABORT: hash gate failed: " + "; ".join(errs))
            return 5
        errs = _identity_check()
        if errs:
            print("ABORT: dataset identity mismatch: " + ", ".join(errs))
            return 6
        if RESULT_ROOT.exists():
            print("ABORT: result directory already exists (one-shot rule).")
            return 7
        print("ABORT: real execution logic completes in the approval round (design-only).")
        return 8
    elif cli.preflight_toy:
        print("PREFLIGHT-TOY: running full machinery on toy fixtures only.")
        hits = scan_forbidden(str(PKG), str(RESULT_ROOT))
        if hits:
            print(f"ABORT: forbidden tokens in package paths: {hits}")
            return 4
        errs = verify_hash_manifest(MANIFEST, REPO)
        if errs:
            print("ABORT: hash gate failed: " + "; ".join(errs))
            return 5
        B = cli.boot_b if cli.boot_b is not None else 100
        stats = _run_cell(build_toy(), B=B)
        print(json.dumps({k: v for k, v in stats.items() if k != "edges"},
                         indent=2, default=float))
        print("PREFLIGHT-TOY: MACHINERY OK (toy numbers carry no interpretation).")
        return 0
    else:
        print("GUARD PATH ONLY: this round is design-only. Execution requires "
              "--execute-real-anchor-validation AND LOCK_STATUS approved.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
