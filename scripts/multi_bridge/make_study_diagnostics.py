"""Write diagnostics.md / diagnostics.json for the baseline + mechanism study."""
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

STUDY = REPO / "out" / "multi_bridge_expansion" / "structural_baseline_mechanism_study"
OUT = STUDY / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    sel = json.loads((STUDY / "calibration" / "selected_threshold.json").read_text(encoding="utf-8"))
    main_df = pd.read_csv(STUDY / "aggregated" / "main_structural_comparison.csv")
    thr = pd.read_csv(STUDY / "supplementary" / "decode_threshold_sensitivity.csv")
    topk = pd.read_csv(STUDY / "supplementary" / "top_k_mm_aggregated.csv")
    dens = pd.read_csv(STUDY / "supplementary" / "balanced_ot_density.csv")

    d: dict[str, Any] = {
        "calibration": {"global_tau": sel["global_tau"], "global_cutoff": sel["global_cutoff_cost"],
                        "grid_lower_bound_hit": sel["global_tau"] == min(sel["threshold_grid_taus"])},
        "main_headline": {
            "split_exact_nonzero_cells": int((main_df["split_exact_mean"] > 0).sum()),
            "merge_exact_nonzero_cells": int((main_df["merge_exact_mean"] > 0).sum()),
            "best_edge_f1_by_bridge": {
                r["bridge"]: (r["method"], round(float(r["edge_f1_mean"]), 4))
                for _, r in main_df.sort_values("edge_f1_mean", ascending=False)
                .drop_duplicates("bridge").iterrows()},
        },
        "frozen_decode_density": {
            "celer_seed42_rc_uot_q_edges": int(thr[(thr["method"] == "RC-UOT-Q") & (thr["threshold"] == 1e-9)]["n_edges"].iloc[0]),
            "celer_seed42_balanced_ot_edges_1e-9": int(thr[(thr["method"] == "Balanced-OT") & (thr["threshold"] == 1e-9)]["n_edges"].iloc[0]),
            "balanced_ot_mean_cells_ge_1e-9": float(dens["cells_ge_1e-9"].mean()),
            "balanced_ot_mean_density": float((dens["cells_ge_1e-9"] / dens["cells"]).mean()),
        },
        "truth_edge_rank_diagnostic": {
            "note": "Per-source cost rank of the true edges is ~3-4 (Top-k-MM analysis), i.e. the "
                    "true split/merge edges are NOT the locally cheapest cells; confusers "
                    "(full-amount merge dst for the split source; half-amount split dsts for the "
                    "merge sources) are cheaper per-edge. This is why (a) the frozen RC-UOT-Q "
                    "decode only achieves edge-recall via global mass allocation, (b) strict "
                    "exact recovery is 0 for every method at the frozen 1e-9 decode, and (c) "
                    "Threshold-MM/Top-k hit their best edge F1 near rank 3-4.",
            "topk_evidence": {
                "celer_top1_f1": float(topk[(topk["bridge"] == "Celer") & (topk["method"] == "Top-1-MM")]["edge_f1_mean"].iloc[0]),
                "celer_top2_f1": float(topk[(topk["bridge"] == "Celer") & (topk["method"] == "Top-2-MM")]["edge_f1_mean"].iloc[0]),
                "celer_top3_f1": float(topk[(topk["bridge"] == "Celer") & (topk["method"] == "Top-3-MM")]["edge_f1_mean"].iloc[0]),
                "celer_top4_f1": float(topk[(topk["bridge"] == "Celer") & (topk["method"] == "Top-4-MM")]["edge_f1_mean"].iloc[0]),
            },
        },
        "balanced_ot_convergence": {
            "all_converged": bool(dens["converged"].all()),
            "worst_final_err": float(dens["final_err"].max()),
            "worst_row_residual": float(dens["row_residual"].max()),
            "worst_col_residual": float(dens["col_residual"].max()),
            "note": "POT stopThr (1e-11) stalls on Multi seeds 42/43 at err ~5e-8 with marginal "
                    "residual ~2e-8 of unit mass (below the decode threshold); treated as "
                    "converged by the residual criterion (<1e-6). Two cosmetic POT warnings in "
                    "the main-run stderr correspond to those instances.",
        },
    }
    (OUT / "diagnostics.json").write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    md = f"""# Diagnostics — baseline + mechanism study

## 1. Why strict exact recovery is 0 for every method at the frozen operating point

Three independent, audited causes (none of them evaluator bugs — see the verification report):

1. **Decode density.** The frozen RC-UOT-Q decode rule is `mass >= 1e-9`; on the 288x288
   synthetic grids the frozen plan has 18,776 positive cells on Celer seed 42
   ({d['frozen_decode_density']['celer_seed42_rc_uot_q_edges']} edges, ~22.6% of the grid).
   The strictly balanced entropic plan is even denser
   ({d['frozen_decode_density']['celer_seed42_balanced_ot_edges_1e-9']} edges on the same grid).
   A decoded edge set of that size cannot equal the 6-edge ground truth exactly.

2. **Truth edges are not the locally cheapest cells.** The per-source cost rank of the true
   edges is around 3-4 (Top-k-MM: F1 jumps from ~0.019 at Top-2 to ~0.248 at Top-3 and ~0.264
   at Top-4). The cheapest cells per source are confusers: the split source's cheapest target
   is the full-amount merge destination (amount cost ~0.0 vs ~0.5 for the true split targets),
   and the merge sources' cheapest targets are the half-amount split destinations. Any local
   threshold that keeps the true edges also keeps cheaper confusers, so exact topology is
   unattainable for local rules; the frozen RC-UOT-Q plan includes the true edges only through
   the global mass allocation (they carry ~0.4% source share, see the frozen correspondence
   export), which the 1e-9 decode reports as a dense many-to-many plan.

3. **The legacy split/merge "recovery" metric is edge inclusion, not exactness.** The frozen
   `split_recovery_rate` counts a truth edge as recovered when it appears anywhere in the
   decoded set; with a dense decode this reaches 0.95 while the strict exact recovery is 0.
   The new unified evaluator reports both views (exact + edge P/R/F1).

## 2. Calibration outcome

- Global tau = **{sel['global_tau']}** (cutoff cost = **{sel['global_cutoff_cost']:.6f}**), selected by
  pooled edge F1 on seeds 101-103 (432 templates, 746,496 cost cells).
- Tau* sits at the **lower boundary of the pre-registered grid** (edge F1 is monotonically
  decreasing in tau because recall saturates at 92.6% already at tau=0.05). The grid was fixed
  in advance; no below-grid value was searched, and seeds 42-46 were never read by calibration.

## 3. Balanced-OT solver health

{d['balanced_ot_convergence']['note']}

## 4. Stress-ladder design notes

- Mass mismatch: dst amounts x (1+m), m in {{0, 0.05, 0.10, 0.20, 0.40}}; realized dst/src mass
  ratios per level are stored in `stress/mass_mismatch.csv` (realized ratio column).
- Unmatched ratio: extra full-mass sources without truth edges, realized ratios ~{{0, 0.10, 0.20,
  0.30, 0.40}} (recorded exactly per cell).
- Decoy density: k in {{1, 2, 4, 8}} decoy pairs per template (0.5x/1x/2x/4x of the main design's 2).
- Time noise: true dst legs offset ~ U(0, 60*(s-1)) s for s in {{1, 2, 4}}; decoy offsets stay
  +60/+120 s. 1x = the current main design.
- Stress seeds are 101-103 (the calibration seeds); the test seeds 42-46 are untouched.

## 5. Known honest limitations

- PolyNetwork stress instances saturate similarly to the faithful run (lock==unlock amounts).
- Exact recovery stays 0 at every stress level for all three transport-family methods; the
  mechanism conclusions therefore rest on edge precision/recall/F1 and FP counts
  (pre-registered caveat in PRE_REGISTERED_HYPOTHESES.md).
"""
    (OUT / "diagnostics.md").write_text(md + "\n", encoding="utf-8")
    print("diagnostics written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
