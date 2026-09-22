"""Supplementary analyses for the baseline + mechanism study.

1. Top-k-MM (structure-aware heuristic, supplementary only): per-source top-k cheapest
   targets by C_eff, k in {1,2,3,4}. NOT used to select anything on the test seeds; all k
   reported. Top-2 carries a structural prior for this benchmark (1->2 / 2->1), so it is
   flagged structure-aware and placed in supplementary/.
2. Decode-threshold sensitivity for the frozen RC-UOT-Q plan and the Balanced-OT plan on
   Celer seed 42: edge P/R/F1 and exact recovery as a function of the mass threshold.
   This documents WHY the frozen 1e-9 decode yields dense edge sets; the frozen operating
   point (1e-9) is not changed.
3. Balanced-OT plan density statistics on all main-run grids.
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

from baseline_mechanism.common import (  # noqa: E402
    BRIDGES, FROZEN_PARAMS, STUDY, TEST_SEEDS,
    decode_plan, evaluate_method, load_frozen_instance, solve_balanced_ot, truth_structure,
)

OUT = STUDY / "supplementary"


def topk_edges(inst: dict[str, Any], k: int) -> list[tuple[str, str]]:
    C = inst["C"]
    sids, tids = inst["sids"], inst["tids"]
    edges: list[tuple[str, str]] = []
    for i in range(C.shape[0]):
        order = np.argsort(C[i], kind="stable")[:k]
        for j in order:
            edges.append((sids[i], tids[int(j)]))
    return edges


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Top-k-MM on the frozen main grids ------------------------------
    rows: list[dict[str, Any]] = []
    for br in BRIDGES:
        for seed in TEST_SEEDS:
            inst = load_frozen_instance(br, seed)
            truth = truth_structure(inst["labels"])
            for k in (1, 2, 3, 4):
                edges = topk_edges(inst, k)
                _df, summ = evaluate_method(inst, truth, edges)
                rows.append({"bridge": br, "seed": seed, "method": f"Top-{k}-MM", "k": k,
                             "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                             "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                             "edge_f1": summ["edge_f1"], "fp_per_template": summ["edge_fp_total"] / 48,
                             "n_pred_edges": summ["n_pred_edges"],
                             "structure_aware_flag": bool(k == 2)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "top_k_mm.csv", index=False)
    agg = df.groupby(["bridge", "method"]).agg(
        split_exact_mean=("split_exact", "mean"), merge_exact_mean=("merge_exact", "mean"),
        edge_precision_mean=("edge_precision", "mean"), edge_recall_mean=("edge_recall", "mean"),
        edge_f1_mean=("edge_f1", "mean"), fp_per_template_mean=("fp_per_template", "mean"),
    ).reset_index()
    agg.to_csv(OUT / "top_k_mm_aggregated.csv", index=False)

    # ---- 2. decode-threshold sensitivity (Celer seed 42) -------------------
    inst = load_frozen_instance("Celer", 42)
    truth = truth_structure(inst["labels"])
    bot = solve_balanced_ot(inst["C"], inst["a_rw"], inst["b_ev"], reg=FROZEN_PARAMS["uot_reg"])
    thr_rows: list[dict[str, Any]] = []
    for thr in (1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.003, 0.01, 0.03, 0.1):
        for name, P in (("RC-UOT-Q", inst["P"]), ("Balanced-OT", bot["P"])):
            edges = decode_plan(P, inst["sids"], inst["tids"], threshold=thr)
            _df, summ = evaluate_method(inst, truth, edges)
            thr_rows.append({"method": name, "threshold": thr, "n_edges": len(edges),
                             "split_exact": summ["split_exact"], "merge_exact": summ["merge_exact"],
                             "edge_precision": summ["edge_precision"], "edge_recall": summ["edge_recall"],
                             "edge_f1": summ["edge_f1"], "fp_total": summ["edge_fp_total"],
                             "n_pred_per_template": summ["n_pred_edges"]})
    tdf = pd.DataFrame(thr_rows)
    tdf.to_csv(OUT / "decode_threshold_sensitivity.csv", index=False)

    # ---- 3. balanced-OT density on all main grids --------------------------
    dens_rows: list[dict[str, Any]] = []
    for br in BRIDGES:
        for seed in TEST_SEEDS:
            i = load_frozen_instance(br, seed)
            b = solve_balanced_ot(i["C"], i["a_rw"], i["b_ev"], reg=FROZEN_PARAMS["uot_reg"])
            P = b["P"]
            dens_rows.append({"bridge": br, "seed": seed, "cells": int(P.size),
                              "cells_ge_1e-9": int((P >= 1e-9).sum()),
                              "cells_ge_1e-6": int((P >= 1e-6).sum()),
                              "cells_ge_1e-4": int((P >= 1e-4).sum()),
                              "cells_ge_1e-3": int((P >= 1e-3).sum()),
                              "converged": b["converged"], "final_err": b["final_err"],
                              "row_residual": b["row_residual"], "col_residual": b["col_residual"],
                              "transported_mass": float(P.sum())})
    ddf = pd.DataFrame(dens_rows)
    ddf.to_csv(OUT / "balanced_ot_density.csv", index=False)

    def df_to_md(frame: pd.DataFrame) -> str:
        f2 = frame.copy()
        for c in f2.columns:
            if pd.api.types.is_float_dtype(f2[c]):
                f2[c] = f2[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        lines = ["| " + " | ".join(str(c) for c in f2.columns) + " |",
                 "| " + " | ".join("---" for _ in f2.columns) + " |"]
        for _, r in f2.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
        return "\n".join(lines)

    with open(OUT / "supplementary_summary.md", "w", encoding="utf-8") as f:
        f.write("# Supplementary analyses\n\n")
        f.write("## Top-k-MM (structure-aware heuristic, supplementary)\n\n"
                "Per-source top-k cheapest targets by C_eff, k in {1,2,3,4}, no test-set tuning "
                "(all k reported). Top-2 is flagged structure-aware because this benchmark's true "
                "structures are exactly 1->2 / 2->1. Data: `top_k_mm.csv`.\n\n")
        f.write(df_to_md(agg))
        f.write("\n\n## Decode-threshold sensitivity (Celer seed 42)\n\n"
                "The frozen RC-UOT-Q decode (1e-9) and the Balanced-OT decode are dense: at 1e-9 "
                "the RC-UOT-Q plan emits ~18.8k edges and the balanced plan ~30k edges on the "
                "288x288 grid, which is why strict exact recovery is 0 for the transport methods "
                "at the frozen operating point. This table documents how the edge metrics move "
                "with the mass threshold; the frozen operating point is NOT changed by this study.\n\n")
        f.write(df_to_md(tdf))
        f.write("\n\n## Balanced-OT plan density (all main grids)\n\n")
        f.write(df_to_md(ddf))
        f.write("\n")
    print("supplementary artifacts written under", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
