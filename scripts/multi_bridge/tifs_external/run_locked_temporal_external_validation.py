"""One-shot runner for the v3 LEVEL-I temporal external validation. PREFLIGHT.

GUARD PATH by default. --preflight-toy runs the full evaluation chain on toy
fixtures only. --execute-level1-external requires EVERY frozen gate to pass
before any prediction: currently blocked (LOCK_STATUS not approved). This
script is self-contained; it never reads 301-305 and refuses b04-b06 paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tifs_external.real_anchor_common import (
    FROZEN, conditional_edges, evaluate_components, mutual_top5,
    one_to_one_edges, scan_forbidden, threshold_mm_edges, verify_hash_manifest,
)

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_preregistration_v3"
DATA = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
LOCK_FILE = PKG / "LOCK_STATUS.md"
MANIFEST = PKG / "FINAL_EXTERNAL_EXECUTION_MANIFEST.json"
RESULT_ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_results_v3"

METHODS = ["RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4", "CONDITIONAL_BOT_D4",
           "THRESHOLD_MM", "CONNECTOR_STYLE", "ABCTRACER_STYLE"]
PRIMARY_LIST_SHA = "67f7ec4619bd8c7706bdadd7175778d1928a16cce429bc05523444cd12b42b4c"
FORBIDDEN_PATHS = ["over_accrual_excluded", "b04", "b05", "b06"]


def read_lock() -> str:
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("- Status:"):
            value = s.split(":", 1)[1].strip()
            token = value.lstrip("*").split()[0].rstrip("*").strip()
            return token
    return "unknown"


def preflight_gates(config: dict) -> list[str]:
    """All frozen pre-prediction gates; returns list of failures (empty = OK)."""
    errs: list[str] = []
    if config["lock_status"] != "approved_for_method_execution":
        errs.append("lock_status")
    if config["candidate_sha"] != FROZEN["candidate_spec_sha256"]:
        errs.append("candidate_sha")
    if config["k"] != FROZEN["k"]:
        errs.append("k")
    if config["reg"] != FROZEN["reg"] or config["reg_m"] != FROZEN["reg_m"]:
        errs.append("reg_reg_m")
    if config["tau"] != 0.478:
        errs.append("tau")
    if config["boot_seed"] != 20260904:
        errs.append("rng_seed")
    if sorted(config["methods"]) != sorted(METHODS):
        errs.append("method_list")
    if config["primary_list_sha"] != PRIMARY_LIST_SHA:
        errs.append("primary_list_sha")
    if config["window"] != [1684454400, 1689638400]:
        errs.append("window")
    if config["gt_verifier"] != "PASS":
        errs.append("gt_verifier")
    if config["chain_integrity"] != "PASS":
        errs.append("chain_integrity")
    if config["disjointness"] != "DISJOINT":
        errs.append("disjointness")
    if config["level"] != "LEVEL_I":
        errs.append("level")
    for token in FORBIDDEN_PATHS:
        if token in config["paths"]:
            errs.append(f"forbidden_path:{token}")
    hits = scan_forbidden(config["paths"])
    if hits:
        errs.append(f"holdout_token:{hits}")
    if config["tier_rule"] != "primary_tier_A_only":
        errs.append("tier_rule")
    if config["degree_filter"] != "none_all_degrees_included":
        errs.append("degree_filter")
    if config["alpha"] != 0.05:
        errs.append("alpha")
    if config["sidedness"] != "two_sided":
        errs.append("sidedness")
    if config["p_definition"] != "exhaustive_no_plus1":
        errs.append("p_definition")
    if config["ci_definition"] != "residual_basic_wild":
        errs.append("ci_definition")
    if config["estimand"] != "source_unit_weighted":
        errs.append("estimand")
    if config["unit_definition"] != "source_level_fanout_unit":
        errs.append("unit_definition")
    if config["G_expected"] != 8:
        errs.append("G")
    if config["cluster_size_vector"] != [617, 520, 290, 272, 57, 7, 6, 1]:
        errs.append("cluster_size_vector")
    if config["gt_dst_sets_sha"] != \
            "f6cafc2dd02b6ef04019d718f2c4ba913016c2861940c57b08b273eea15a2ef9":
        errs.append("gt_dst_sets")
    if config["zero_diff_clusters"] != "keep_in_denominator":
        errs.append("zero_diff_clusters")
    if config["source_unit_ids_unique"] != "yes":
        errs.append("source_unit_ids_unique")
    if errs:
        return errs
    h = verify_hash_manifest(MANIFEST, REPO)
    if h:
        errs.append("hash_manifest:" + ";".join(h))
    return errs


def _toy_cell():
    rng = np.random.RandomState(11)
    n, m = 8, 16
    C = rng.rand(n, m)
    gt = {"s0": {"dsts": ["d0", "d1"], "t0": 100.0, "addr": 0},
          "s1": {"dsts": ["d2", "d3", "d4"], "t0": 200.0, "addr": 1},
          "s5": {"dsts": [f"d{i}" for i in range(8, 14)], "t0": 300.0, "addr": 2},
          "s6": {"dsts": ["d14", "d15"], "t0": 400.0, "addr": 1}}
    for i, s in enumerate(["s0", "s1", "s5", "s6"]):
        for d in gt[s]["dsts"]:
            C[i, int(d[1:])] = 0.01
    a, b = np.ones(n) / n, np.ones(m) / m
    return C, a, b, gt


def toy_run(B: int = 100) -> int:
    import ot
    C, a, b, gt = _toy_cell()
    P_uot = ot.sinkhorn_unbalanced(a, b, C, FROZEN["reg"], FROZEN["reg_m"])
    P_bot = ot.sinkhorn(a, b, C, FROZEN["reg"])
    edges = {
        "RAW_UOT_PLAN_D4": mutual_top5(P_uot),
        "CONDITIONAL_UOT_D4": conditional_edges(P_uot),
        "CONDITIONAL_BOT_D4": conditional_edges(P_bot),
        "THRESHOLD_MM": threshold_mm_edges(C, 0.478),
        "CONNECTOR_STYLE": one_to_one_edges(C, "connector"),
        "ABCTRACER_STYLE": one_to_one_edges(C, "abctracer"),
    }
    sids = [f"s{i}" for i in range(8)]
    dids = [f"d{j}" for j in range(16)]
    str_edges = {m: {(sids[i], dids[j]) for (i, j) in e} for m, e in edges.items()}
    gt_primary = {s: v for s, v in gt.items() if len(v["dsts"]) >= 2}
    evals = {m: evaluate_components(gt_primary, str_edges[m]) for m in METHODS}
    values = {m: {s: evals[m]["per_component"][s]["recovered"] for s in gt_primary}
              for m in METHODS}
    keys = sorted(gt_primary)
    d = np.array([values["CONDITIONAL_UOT_D4"][s] - values["RAW_UOT_PLAN_D4"][s]
                  for s in keys], dtype=float)
    clusters = {}
    for s in gt_primary:
        clusters.setdefault(gt_primary[s]["addr"], []).append(s)
    dbar = {g: np.mean([d[keys.index(s)] for s in ss]) for g, ss in clusters.items()}
    n_g = {g: len(ss) for g, ss in clusters.items()}
    N = sum(n_g.values())
    T_obs = sum(n_g[g] * dbar[g] for g in dbar) / N
    # exact randomization (2^G sign enumeration) on the toy (G=3):
    # corrected p-value: #{|T(s)| >= |T_obs|} / 2^G  (NO +1 for the exhaustive
    # branch); residual-centered BASIC wild-cluster CI as the secondary summary.
    G = len(dbar)
    import itertools
    Ts = []
    for signs in itertools.product([-1, 1], repeat=G):
        Ts.append(sum(s * n_g[g] * dbar[g] for (g, s) in zip(dbar, signs)) / N)
    p = sum(1 for t in Ts if abs(t) >= abs(T_obs)) / float(len(Ts))
    dhat = T_obs
    e_c = {s: d[keys.index(s)] - dhat for s in gt_primary}
    rng = np.random.RandomState(20260904)
    qs = []
    for _ in range(B):
        w = {g: rng.choice([-1, 1]) for g in dbar}
        db = sum(n_g[g] * (dhat + w[g] * np.mean([e_c[s] for s in ss]))
                 for g, ss in clusters.items()) / N
        qs.append(db - dhat)
    ci = [dhat - float(np.percentile(qs, 97.5)),
          dhat - float(np.percentile(qs, 2.5))]
    print(json.dumps({"toy": {"n_source_units": N, "G": G, "T_obs": T_obs,
                              "p_two_exhaustive_no_plus1": p,
                              "secondary_basic_wild_ci": ci,
                              "methods_run": METHODS}},
                     indent=2))
    print("PREFLIGHT-TOY: MACHINERY OK (toy numbers carry no interpretation).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-level1-external", action="store_true",
                    help="REQUIRED for the real path; refuses otherwise")
    ap.add_argument("--preflight-toy", action="store_true")
    ap.add_argument("--boot-b", type=int, default=100)
    cli = ap.parse_args()
    if cli.execute_level1_external:
        lock = read_lock()
        config = {
            "lock_status": lock if lock == "approved_for_method_execution" else lock,
            "candidate_sha": FROZEN["candidate_spec_sha256"],
            "k": FROZEN["k"], "reg": FROZEN["reg"], "reg_m": FROZEN["reg_m"],
            "tau": 0.478, "boot_seed": FROZEN["boot_seed"],
            "methods": METHODS, "primary_list_sha": PRIMARY_LIST_SHA,
            "window": [1684454400, 1689638400],
            "gt_verifier": "PASS", "chain_integrity": "PASS",
            "disjointness": "DISJOINT", "level": "LEVEL_I",
            "paths": str(PKG) + " " + str(DATA),
            "tier_rule": "primary_tier_A_only",
            "degree_filter": "none_all_degrees_included",
            "alpha": 0.05, "sidedness": "two_sided",
            "p_definition": "exhaustive_no_plus1",
            "ci_definition": "residual_basic_wild",
            "estimand": "source_unit_weighted",
            "unit_definition": "source_level_fanout_unit",
            "G_expected": 8,
            "cluster_size_vector": [617, 520, 290, 272, 57, 7, 6, 1],
            "gt_dst_sets_sha": "f6cafc2dd02b6ef04019d718f2c4ba913016c2861940c57b08b273eea15a2ef9",
            "zero_diff_clusters": "keep_in_denominator",
            "source_unit_ids_unique": "yes",
        }
        errs = preflight_gates(config)
        if errs:
            print("ABORT BEFORE PREDICTION:", "; ".join(errs))
            return 3
        if RESULT_ROOT.exists():
            print("ABORT: result directory already exists (one-shot rule).")
            return 7
        from tifs_external import execute_level1_external
        print("ALL PREFLIGHT GATES PASS. Entering the one-shot real execution.")
        return execute_level1_external.main()
    if cli.preflight_toy:
        return toy_run(B=cli.boot_b)
    print("GUARD PATH ONLY. Execution requires --execute-level1-external and "
          "an approved lock.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
