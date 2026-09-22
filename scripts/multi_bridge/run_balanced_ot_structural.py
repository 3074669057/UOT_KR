"""Balanced-OT baseline for the structural mechanism study.

Strictly balanced entropic OT on the SAME ``C_effective`` cost matrix as RC-UOT-Q:

    min_P  <C, P> + reg * H(P)   s.t.   P 1 = a,  P^T 1 = b

- reg = 0.05 (identical to the frozen RC-UOT-Q entropic regularization).
- NO reg_m, NO unbalanced relaxation, NO dustbin/dummy node, NO partial OT.
- Feasibility rule: each marginal set is normalized to unit total mass
  (the task's specified rule when source and destination totals differ).
- Decoding: exactly the RC-UOT-Q transport-to-correspondence rule
  (``decode_correspondence`` semantics: edge iff transport mass >= threshold,
  threshold = 1e-9). Hungarian is NEVER applied.

CLI (inspection / small grids):
  python run_balanced_ot_structural.py --cost cost.npz --a a.npy --b b.npy --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from baseline_mechanism.common import decode_plan, solve_balanced_ot  # noqa: E402


def run_balanced_ot(C: np.ndarray, a: np.ndarray, b: np.ndarray, reg: float = 0.05) -> dict[str, Any]:
    return solve_balanced_ot(C, a, b, reg=reg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--a", required=True, help=".npy source marginal (unit-normalized inside)")
    ap.add_argument("--b", required=True, help=".npy target marginal (unit-normalized inside)")
    ap.add_argument("--sids", default=None)
    ap.add_argument("--tids", default=None)
    ap.add_argument("--decode-threshold", type=float, default=1e-9)
    ap.add_argument("--out-dir", required=True)
    cli = ap.parse_args()

    C = np.load(cli.cost)
    C = C["C_effective"] if "C_effective" in C.files else C["C"]
    a = np.load(cli.a)
    b = np.load(cli.b)
    res = run_balanced_ot(np.asarray(C, dtype=float), np.asarray(a, dtype=float),
                          np.asarray(b, dtype=float))
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "balanced_ot_transport.npz", P=res["P"])
    sids = [str(x) for x in np.load(cli.sids)["ids"]] if cli.sids else [str(i) for i in range(res["P"].shape[0])]
    tids = [str(x) for x in np.load(cli.tids)["ids"]] if cli.tids else [str(j) for j in range(res["P"].shape[1])]
    edges = decode_plan(res["P"], sids, tids, threshold=float(cli.decode_threshold))
    with open(out / "balanced_ot_summary.json", "w", encoding="utf-8") as f:
        json.dump({"n_edges_decoded": len(edges), "transported_mass": float(res["P"].sum()),
                   "converged": res["converged"], "final_err": res["final_err"],
                   "row_residual": res["row_residual"], "col_residual": res["col_residual"],
                   "kappa": res["kappa"]}, f, indent=2)
    with open(out / "balanced_ot_edges.csv", "w", encoding="utf-8") as f:
        f.write("src_flow_id,dst_flow_id,transport_mass\n")
        idx_s = {s: i for i, s in enumerate(sids)}
        idx_t = {t: j for j, t in enumerate(tids)}
        for s, d in edges:
            f.write(f"{s},{d},{res['P'][idx_s[s], idx_t[d]]:.12g}\n")
    print(f"balanced-OT: transported={res['P'].sum():.6f} converged={res['converged']} "
          f"decoded_edges={len(edges)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
