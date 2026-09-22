"""Frozen execution machinery shared by the holdout runner and the independent verifier.

Scientific definitions are FROZEN by the preregistration package; this module only
implements them deterministically. Statistics: paired hierarchical bootstrap with
B = 4000 replicates and RNG seed 20240101 (frozen in HOLDOUT_ANALYSIS_PLAN.md).
No scientific threshold lives in code that is not in the preregistration documents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

PRE = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_holdout_preregistration"
CP = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_candidate_dev"
AF = REPO / "out" / "multi_bridge_expansion" / "amount_free_candidate_dev"
TDS = REPO / "out" / "multi_bridge_expansion" / "transport_dual_scaling_diagnosis"

BRIDGES = ("Celer", "Multi", "Poly")
DEV_SEEDS = (201, 202, 203, 204, 205)
HOLD_SEEDS = (301, 302, 303, 304, 305)  # NEVER touched without --execute-holdout
METHODS = ("RAW_UOT_PLAN_D4", "CONDITIONAL_UOT_D4", "AMOUNT_FREE_COST_D4",
           "CONDITIONAL_BOT_D4", "SUPPORT_PLUS_K_D4")
K5 = 5
N_BOOT = 4000
BOOT_SEED = 20240101
RECOVERY_THRESHOLD = 0.75
BRIDGE_COLLAPSE_BOUND = -0.005
FP_FN_FACTOR = 3.0
DEGENERATE_COST_CI_BOUND = -0.01
UOT_CONV_ERR = 1e-7
BOT_CONV_RESID = 1e-6

EXPECTED_DEV_ANCHORS = {
    "RAW_UOT_PLAN_D4": 0.2374, "CONDITIONAL_UOT_D4": 0.3129,
    "AMOUNT_FREE_COST_D4": 0.3172, "CONDITIONAL_BOT_D4": 0.3131,
    "SUPPORT_PLUS_K_D4": 0.3116,
}


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_prereg_hashes(repo: Path = REPO) -> list[str]:
    """Return a list of hash-mismatch errors (empty = all good). Aborts BEFORE any data
    access. Path resolution: 'file' entries are repo-relative; 'scripts/...' entries are
    repo-relative; everything else lives in the preregistration directory."""
    errors: list[str] = []
    mani = json.loads((PRE / "HASH_MANIFEST.json").read_text(encoding="utf-8"))
    for rel, e in mani["managed_hashes"].items():
        if "file" in e:
            p = repo / e["file"]
        elif rel.startswith("scripts/"):
            p = repo / rel
        else:
            p = PRE / rel
        if not p.is_file():
            errors.append(f"missing file: {rel}")
            continue
        if sha256(p) != e["sha256"]:
            errors.append(f"hash mismatch: {rel}")
    return errors


def identity_check(params: dict[str, Any] | None = None) -> list[str]:
    """Verify frozen config identity (k, reg, reg_m, seeds, bridges, methods, weights).
    params may override for negative tests only."""
    errors: list[str] = []
    p = params or {}
    if p.get("k", K5) != 5:
        errors.append(f"k changed: {p.get('k')}")
    if p.get("reg", 0.05) != 0.05:
        errors.append(f"reg changed: {p.get('reg')}")
    if p.get("reg_m", 0.5) != 0.5:
        errors.append(f"reg_m changed: {p.get('reg_m')}")
    if "seeds" in p and tuple(p["seeds"]) != (301, 302, 303, 304, 305):
        errors.append(f"seeds changed: {p.get('seeds')}")
    if tuple(p.get("bridges", BRIDGES)) != ("Celer", "Multi", "Poly"):
        errors.append(f"bridges changed: {p.get('bridges')}")
    if tuple(p.get("methods", METHODS)) != METHODS:
        errors.append(f"methods changed: {p.get('methods')}")
    from dev_candidate.af_common import KEPT_ABS_WEIGHTS, primary_weights
    w = primary_weights()
    for k2, v in KEPT_ABS_WEIGHTS.items():
        if abs(w[k2] - v / 0.65) > 1e-12:
            errors.append(f"weight changed: {k2}")
    from decoder_audit.da_common import FROZEN_PARAMS
    if FROZEN_PARAMS["uot_reg"] != 0.05 or FROZEN_PARAMS["uot_reg_m"] != 0.5:
        errors.append("FROZEN_PARAMS changed")
    return errors


# --------------------------------------------------------------------------- #
# Frozen decode rules (identical to the candidate spec)
# --------------------------------------------------------------------------- #

def rank_desc(S: np.ndarray, axis: int) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if axis == 1:
        R = np.zeros_like(S, dtype=int)
        for i in range(S.shape[0]):
            order = np.lexsort((np.arange(S.shape[1]), -S[i]))
            R[i, order] = np.arange(1, S.shape[1] + 1)
        return R
    R = np.zeros_like(S, dtype=int)
    for j in range(S.shape[1]):
        order = np.lexsort((np.arange(S.shape[0]), -S[:, j]))
        R[order, j] = np.arange(1, S.shape[0] + 1)
    return R


def conditional_scores(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S_row = np.zeros_like(P, dtype=float)
    S_col = np.zeros_like(P, dtype=float)
    S_row[:, c > 0] = P[:, c > 0] / c[None, c > 0]
    S_col[r > 0, :] = P[r > 0, :] / r[r > 0, None]
    return S_row, S_col


def conditional_edges(P: np.ndarray, sids: list[str], tids: list[str],
                      k: int = K5) -> list[tuple[str, str]]:
    S_row, S_col = conditional_scores(P)
    rr = rank_desc(S_row, 1)
    cr = rank_desc(S_col, 0)
    out = []
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                out.append((sids[i], tids[j]))
    return out


def mutual_top5(S: np.ndarray, sids: list[str], tids: list[str],
                k: int = K5) -> list[tuple[str, str]]:
    rr = rank_desc(S, 1)
    cr = rank_desc(S, 0)
    out = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if rr[i, j] <= k and cr[i, j] <= k:
                out.append((sids[i], tids[j]))
    return out


def all_method_edges(cell: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """The five preregistered methods for one cell. cell needs: sids, tids, C_primary,
    P_uot, P_bot (plus K built internally)."""
    C = cell["C_primary"]
    K = np.exp(-C / 0.05)
    support = cell["P_uot"] > 1e-9
    S_support = np.where(support, K, -1e300)
    return {
        "RAW_UOT_PLAN_D4": mutual_top5(cell["P_uot"], cell["sids"], cell["tids"], K5),
        "CONDITIONAL_UOT_D4": conditional_edges(cell["P_uot"], cell["sids"], cell["tids"], K5),
        "AMOUNT_FREE_COST_D4": mutual_top5(K, cell["sids"], cell["tids"], K5),
        "CONDITIONAL_BOT_D4": conditional_edges(cell["P_bot"], cell["sids"], cell["tids"], K5),
        "SUPPORT_PLUS_K_D4": mutual_top5(S_support, cell["sids"], cell["tids"], K5),
    }


# --------------------------------------------------------------------------- #
# Frozen statistics (HOLDOUT_ANALYSIS_PLAN.md)
# --------------------------------------------------------------------------- #

def paired_bootstrap_per_bridge(values_by_bridge: dict[str, np.ndarray],
                                n_boot: int = N_BOOT) -> dict[str, Any]:
    """Frozen paired hierarchical bootstrap: resample template pairs WITHIN each bridge,
    then average bridge means. Returns macro + per-bridge means and 95% CIs."""
    rng = np.random.RandomState(BOOT_SEED)
    per_bridge: dict[str, Any] = {}
    boot_macro: list[float] = []
    for b, v in sorted(values_by_bridge.items()):
        v = np.asarray(v, dtype=float)
        per_bridge[b] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n_pairs": int(v.size),
        }
        boots = np.array([rng.choice(v, size=v.size, replace=True).mean()
                          for _ in range(n_boot)])
        per_bridge[b]["ci95_lo"] = float(np.percentile(boots, 2.5))
        per_bridge[b]["ci95_hi"] = float(np.percentile(boots, 97.5))
        boot_macro.append(boots)
    macro_boots = np.mean(np.array(boot_macro), axis=0)
    means = np.array([per_bridge[b]["mean"] for b in sorted(per_bridge)])
    return {
        "per_bridge": per_bridge,
        "macro_mean": float(means.mean()),
        "macro_ci95_lo": float(np.percentile(macro_boots, 2.5)),
        "macro_ci95_hi": float(np.percentile(macro_boots, 97.5)),
        "n_boot": n_boot, "rng_seed": BOOT_SEED,
    }


def evaluate_gates(stats: dict[str, Any], mechanism: dict[str, Any],
                   solver: dict[str, Any], degenerate: bool = False) -> dict[str, Any]:
    """Frozen gates A-E + recovery handling + TYPE classification (HOLDOUT_DECISION_RULES)."""
    d_primary = stats["d_primary"]
    d_cost = stats["d_cost"]
    d_support = stats["d_support"]
    f1 = stats["macro_f1"]
    gate_a = bool(d_primary["macro_ci95_lo"] > 0)
    # mechanism gate B (macro-level, all five directions)
    gate_b = bool(mechanism["row_harmful_delta"] < 0
                  and mechanism["col_harmful_delta"] < 0
                  and mechanism["split_child_delta"] < 0
                  and mechanism["merge_dst_delta"] < 0
                  and mechanism["retention_delta"] > 0)
    D = f1["AMOUNT_FREE_COST_D4"] - f1["RAW_UOT_PLAN_D4"]
    degenerate = degenerate or D <= 0
    if not degenerate:
        recovery = (f1["CONDITIONAL_UOT_D4"] - f1["RAW_UOT_PLAN_D4"]) / D
        gate_c = recovery >= RECOVERY_THRESHOLD
        recovery_label = "recovery_fraction"
    else:
        recovery = float("nan")
        gate_c = bool(d_cost["macro_ci95_lo"] >= DEGENERATE_COST_CI_BOUND
                      and d_primary["macro_ci95_lo"] > 0)
        recovery_label = "RECOVERY_DEGENERATE_ALTERNATIVE_PASS"
    # D: bridge collapse (no per-bridge paired mean below the bound)
    per_b = d_primary["per_bridge"]
    gate_d = bool(all(per_b[b]["mean"] >= BRIDGE_COLLAPSE_BOUND for b in per_b))
    # E: FP/FN + convergence
    fp_ratio = stats["fp_cond"] / max(stats["fp_raw"], 1e-12)
    fn_ratio = stats["fn_cond"] / max(stats["fn_raw"], 1e-12)
    gate_e = bool(fp_ratio <= FP_FN_FACTOR and fn_ratio <= FP_FN_FACTOR
                  and solver["all_converged"])
    gates = {"A": gate_a, "B": gate_b, "C": gate_c, "D": gate_d, "E": gate_e}
    repair = all(gates.values())
    # TYPE classification (independent of repair decision)
    if repair and d_support["macro_ci95_lo"] > 0 and d_cost["macro_ci95_lo"] > 0:
        ttype = "TYPE A"
    elif repair and (d_cost["macro_ci95_lo"] >= -0.01):
        ttype = "TYPE B"
    elif gate_a and gate_b and (not gate_c):
        ttype = "TYPE C"
    else:
        ttype = "TYPE D"
    # bridge heterogeneity
    heterogeneous = bool(any(per_b[b]["ci95_hi"] < 0 for b in per_b) and d_primary["macro_ci95_lo"] > 0)
    return {
        "gates": gates, "decoder_repair": "DECODER_REPAIR_CONFIRMED" if repair else "NOT_CONFIRMED",
        "transport_value_type": ttype,
        "recovery": {recovery_label: float(recovery) if not degenerate else None,
                     "denominator_D": float(D), "degenerate": degenerate,
                     "gate_c": gate_c},
        "fp_fn": {"fp_ratio": float(fp_ratio), "fn_ratio": float(fn_ratio)},
        "heterogeneous_effect": heterogeneous,
    }
