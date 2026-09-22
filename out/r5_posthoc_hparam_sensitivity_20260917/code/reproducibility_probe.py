"""Reproduction probe (run during preflight; development seeds only).

Establishes the two facts the sweep depends on:
  1. the frozen development cells contain everything needed, so no data has to be
     regenerated (the risk-weighted/evidence-weighted marginals recomputed from the frozen
     `flows.json` reproduce the frozen `transport_uot.npz` marginals with 0.0 error, and the
     frozen flow ids match the frozen id arrays exactly);
  2. the frozen cost decomposition closes exactly, i.e.
     `C_effective == sum_k w_k * component_k + bridge_prior_bonus` with the frozen absolute
     weights, and the paper's primary amount-free renormalised cost reproduces the frozen
     `amount_free_candidate_dev` C_primary with 0.0 error.

NEVER touches seeds 301-305.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO / "src"), str(REPO / "scripts" / "multi_bridge")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cross.domain.uot.uot_solver import (  # noqa: E402
    _evidence_weighted_target_mass, _risk_weighted_source_mass,
)
from dev_candidate.af_common import KEPT_ABS_WEIGHTS, primary_weights  # noqa: E402
from dev_candidate2.cp_common import AF  # noqa: E402

CTD = REPO / "out" / "multi_bridge_expansion" / "cost_transport_diagnosis"
BRIDGES = ("Celer", "Multi", "Poly")
DEV_SEEDS = (201, 202, 203, 204, 205)
FULL_W = {"amount": 0.40, "time": 0.25, "route": 0.15, "risk": 0.15,
          "evidence": 0.05, "novelty": 0.05}
COMP = {"amount": "amount_cost", "time": "time_cost", "route": "route_cost",
        "risk": "risk_cost", "evidence": "evidence_cost", "novelty": "address_novelty_cost"}


def main() -> int:
    rows = []
    worst = {"marginal": 0.0, "closure": 0.0, "primary": 0.0, "ids": 0}
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            root = CTD / "plans" / "dev" / bridge / f"seed_{seed}"
            fl = json.loads((root / "flows.json").read_text(encoding="utf-8"))
            eth, bnb = fl["src"], fl["dst"]
            a0, a_rw = _risk_weighted_source_mass(eth, lambda_risk=0.25)
            b0, b_ev = _evidence_weighted_target_mass(bnb)

            u = np.load(root / "transport_uot.npz", allow_pickle=True)
            c = np.load(root / "cost.npz", allow_pickle=False)
            c2 = np.load(AF / "plans" / bridge / f"seed_{seed}" / "costs.npz",
                         allow_pickle=False)
            ids = np.load(root / "ids.npz", allow_pickle=True)
            sids = [str(x) for x in ids["sids"]]
            tids = [str(x) for x in ids["tids"]]

            d_marg = max(float(np.abs(a_rw - np.asarray(u["a_rw"])).max()),
                         float(np.abs(b_ev - np.asarray(u["b_ev"])).max()),
                         float(np.abs(a0 - np.asarray(u["a0"])).max()),
                         float(np.abs(b0 - np.asarray(u["b0"])).max()))
            d_ids = int(sids != [str(f.get("flow_id")) for f in eth]) + \
                int(tids != [str(f.get("flow_id")) for f in bnb])

            Ce = np.asarray(c["C_effective"], dtype=float)
            recon = np.asarray(c["bridge_prior_bonus"], dtype=float).copy()
            for name, w in FULL_W.items():
                recon = recon + w * np.asarray(c[COMP[name]], dtype=float)
            d_closure = float(np.abs(recon - Ce).max())

            pw = primary_weights()
            rec_p = np.zeros_like(Ce)
            for name, w in pw.items():
                rec_p = rec_p + w * np.asarray(c[COMP[name]], dtype=float)
            d_primary = float(np.abs(rec_p - np.asarray(c2["C_primary"], dtype=float)).max())

            rows.append({
                "bridge": bridge, "seed": seed,
                "max_abs_err_recomputed_marginals_vs_frozen": d_marg,
                "flow_id_mismatches": d_ids,
                "max_abs_err_cost_decomposition_closure": d_closure,
                "max_abs_err_primary_vs_frozen_amount_free": d_primary,
                "bridge_prior_bonus_nonzero_entries": int(
                    (np.asarray(c["bridge_prior_bonus"]) != 0).sum()),
                "n_sources": len(sids), "n_targets": len(tids),
            })
            worst["marginal"] = max(worst["marginal"], d_marg)
            worst["closure"] = max(worst["closure"], d_closure)
            worst["primary"] = max(worst["primary"], d_primary)
            worst["ids"] += d_ids

    out = {
        "purpose": ("preflight reproduction check on frozen development cells; no data was "
                    "regenerated and no holdout seed was touched"),
        "development_seeds": list(DEV_SEEDS),
        "holdout_seeds_touched": [],
        "frozen_primary_weights": primary_weights(),
        "frozen_kept_abs_weights": KEPT_ABS_WEIGHTS,
        "worst_case_errors": worst,
        "pass": bool(worst["marginal"] < 1e-12 and worst["closure"] < 1e-10
                     and worst["primary"] < 1e-12 and worst["ids"] == 0),
        "cells": rows,
    }
    dest = REPO / "out" / "r5_posthoc_hparam_sensitivity_20260917" / "00_preflight" / \
        "reproducibility_probe.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(dest.name), "worst_case_errors": worst,
                      "pass": out["pass"]}, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
