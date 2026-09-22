"""Repair validation on NON-v4 data only (development-exposed dev cells
201-205 and toy fixtures). NO v3, NO b04-b06, NO 301-305.

Gates:
  P1: positive-support wrapper equivalence (old vs repaired) on dev cells with
      strictly positive test marginals: plan max-abs discrepancy, decoder
      output identity, evaluation identity.
  P2: zero-support toy tests (8 frozen cases).
  P3: historical zero-mass diagnostic on dev cells (data-exposed).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tifs_external.solver_support import (  # noqa: E402
    old_bot, old_uot, support_aware_bot, support_aware_uot,
)
from tifs_external.execute_level1_external import (  # noqa: E402
    mutual_top5, conditional_edges_arrays,
)

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
PRE = REPO / "out" / "multi_bridge_expansion" / "conditional_plan_holdout_preregistration" / "preflight_tmp" / "cells"
REG, REG_M = 0.05, 0.5
OUT = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_preregistration_v3" / "repair_validation"
OUT.mkdir(parents=True, exist_ok=True)


def dev_cells():
    cells = []
    for bridge in ("Celer", "Multi", "Poly"):
        for seed in (201, 202, 203, 204, 205):
            f = PRE / bridge / f"seed_{seed}" / "cell_inputs.npz"
            if f.is_file():
                d = np.load(f, allow_pickle=True)
                cells.append((f"{bridge}_{seed}", d["C_primary"], d["P_uot"],
                              d["P_bot"]))
    return cells


def equivalence_tests() -> dict:
    results = []
    for name, C, P_uot, P_bot in dev_cells():
        a = np.maximum(P_uot.sum(axis=1), 1e-9)
        b = np.maximum(P_uot.sum(axis=0), 1e-9)
        a = a / a.sum()
        b = b / b.sum()
        if not (np.all(a > 0) and np.all(b > 0)):
            continue  # zero-support cells go to the diagnostic/test gate
        try:
            P_old_u = old_uot(a, b, C, REG, REG_M)
            P_new_u = support_aware_uot(a, b, C, REG, REG_M)
            P_old_b = old_bot(a, b, C, REG)
            P_new_b = support_aware_bot(a, b, C, REG)
        except Exception as exc:  # noqa: BLE001
            results.append({"cell": name, "error": str(exc)[:100]})
            continue
        du = float(np.max(np.abs(P_old_u - P_new_u)))
        db = float(np.max(np.abs(P_old_b - P_new_b)))
        edges_old = mutual_top5(P_old_u, 5)
        edges_new = mutual_top5(P_new_u, 5)
        Sr = P_new_u / np.maximum(P_new_u.sum(axis=0, keepdims=True), 1e-300)
        Sc = P_new_u / np.maximum(P_new_u.sum(axis=1, keepdims=True), 1e-300)
        ce_new = conditional_edges_arrays(Sr, Sc, 5)
        Sr = P_old_u / np.maximum(P_old_u.sum(axis=0, keepdims=True), 1e-300)
        Sc = P_old_u / np.maximum(P_old_u.sum(axis=1, keepdims=True), 1e-300)
        ce_old = conditional_edges_arrays(Sr, Sc, 5)
        results.append({"cell": name,
                        "uot_max_abs_diff": du,
                        "bot_max_abs_diff": db,
                        "raw_decoder_identical": edges_old == edges_new,
                        "conditional_decoder_identical": ce_old == ce_new})
    return results


def zero_support_toy_tests() -> dict:
    rng = np.random.RandomState(42)
    cases = {
        "one_zero_src": {"zs": [0], "zd": []},
        "multi_zero_src": {"zs": [0, 1], "zd": []},
        "one_zero_dst": {"zs": [], "zd": [0]},
        "multi_zero_dst": {"zs": [], "zd": [0, 1]},
        "all_zero_src": {"zs": list(range(6)), "zd": []},
        "all_zero_dst": {"zs": [], "zd": list(range(8))},
        "mixed_sparse": {"zs": [0, 2], "zd": [1, 5]},
        "zero_mass_on_gt_unit": {"zs": [3], "zd": []},
    }
    out = {}
    for name, spec in cases.items():
        n, m = 6, 8
        C = rng.rand(n, m)
        a = rng.rand(n)
        b = rng.rand(m)
        for i in spec["zs"]:
            a[i] = 0.0
        for j in spec["zd"]:
            b[j] = 0.0
        a = a / max(a.sum(), 1e-12)
        b = b / max(b.sum(), 1e-12)
        P = support_aware_uot(a, b, C, REG, REG_M)
        Pb = support_aware_bot(a, b, C, REG)
        ok = {
            "no_nan": not np.isnan(P).any() and not np.isnan(Pb).any(),
            "zero_src_rows_zero": bool(np.all(P[spec["zs"], :] == 0)) if spec["zs"] else True,
            "zero_dst_cols_zero": bool(np.all(P[:, spec["zd"]] == 0)) if spec["zd"] else True,
            "all_zero_side_plan": (np.all(P == 0) if
                                   (set(spec["zs"]) == set(range(n)) or
                                    set(spec["zd"]) == set(range(m))) else True),
        }
        Ip = [i for i in range(n) if a[i] > 0]
        Jp = [j for j in range(m) if b[j] > 0]
        if Ip and Jp:
            import ot
            Pr = ot.sinkhorn_unbalanced(a[Ip], b[Jp], C[np.ix_(Ip, Jp)], REG, REG_M)
            ok["positive_block_matches_reduced"] = bool(
                np.allclose(P[np.ix_(Ip, Jp)], Pr, atol=1e-12))
        edges = mutual_top5(P, 5)
        ok["decoder_deterministic"] = (mutual_top5(P, 5) == edges)
        out[name] = ok
    return out


def historical_zero_mass_diagnostic() -> dict:
    diag = []
    for name, C, P_uot, P_bot in dev_cells():
        a = P_uot.sum(axis=1)
        b = P_uot.sum(axis=0)
        diag.append({"cell": name,
                     "n_zero_src_mass": int(np.sum(a <= 1e-300)),
                     "n_zero_dst_mass": int(np.sum(b <= 1e-300)),
                     "n": C.shape[0], "m": C.shape[1]})
    return diag


def main() -> int:
    eq = equivalence_tests()
    zt = zero_support_toy_tests()
    diag = historical_zero_mass_diagnostic()
    eq_pass = (len(eq) > 0 and
               all(r.get("uot_max_abs_diff", 9) < 1e-9 and
                   r.get("bot_max_abs_diff", 9) < 1e-9 and
                   r.get("raw_decoder_identical") is True and
                   r.get("conditional_decoder_identical") is True
                   for r in eq))
    zt_pass = all(all(v for v in case.values()) for case in zt.values())
    report = {
        "positive_support_equivalence": {
            "n_cells": len(eq), "cells": eq, "PASS": eq_pass,
            "max_uot_diff": max((r["uot_max_abs_diff"] for r in eq), default=None),
            "max_bot_diff": max((r["bot_max_abs_diff"] for r in eq), default=None),
            "decoder_discrepancy_count": sum(
                1 for r in eq if not r.get("raw_decoder_identical")
                or not r.get("conditional_decoder_identical")),
        },
        "zero_support_toy_tests": {"cases": zt, "PASS": zt_pass},
        "historical_zero_mass_diagnostic": diag,
        "POSITIVE_SUPPORT_EQUIVALENCE": "PASS" if eq_pass else "FAIL",
        "ZERO_SUPPORT_TESTS": "PASS" if zt_pass else "FAIL",
    }
    (OUT / "repair_validation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if (eq_pass and zt_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
