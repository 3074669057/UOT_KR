"""Support vs ranking attribution (DIAGNOSTIC ONLY; dev 201-205).

Four-level attribution on the amount-free kernel/plans:
  COST_D4          : D4@5 mutual top-k on the raw kernel K (no transport).
  SUPPORT_ONLY_D4  : transport support (P > 1e-9) decides ELIGIBILITY; ranking by K;
                     same D4@5 structural rule.
  PLAN_RANK_ONLY   : D4@5 mutual top-k on the plan P with the SAME eligible universe as
                     COST_D4 (all cells) — identical to the standard UOT_PLAN_D4, because
                     the locked D4 never prunes by support. Recorded as an explicit
                     identity note.
  FULL_PLAN_D4     : same as PLAN_RANK_ONLY (the actual deployed decoder).
Support contribution = SUPPORT_ONLY - COST; ranking contribution = PLAN_RANK_ONLY - COST.
"""
from __future__ import annotations

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

from decoder_audit.da_common import evaluate_edges  # noqa: E402
from diag2.tds_common import (  # noqa: E402
    BRIDGES, DEV_SEEDS, K5, REG, TDS, d4_edges_on_score, gt_cells, load_cell, rank_asc,
)


def main() -> int:
    out = TDS / "support_vs_ranking"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for bridge in BRIDGES:
        for seed in DEV_SEEDS:
            cell = load_cell(bridge, seed)
            C = cell["C_primary"]
            K = np.exp(-C / REG)
            for plan_name, P in (("UOT", cell["P_uot"]), ("BOT", cell["P_bot"])):
                # COST_D4
                edges_cost = d4_edges_on_score(K, cell["sids"], cell["tids"], K5)
                # SUPPORT_ONLY: rank by K restricted to the transport support
                support = P > 1e-9
                S_support = np.where(support, -K, 1e12)
                edges_support = d4_edges_on_score(-S_support, cell["sids"], cell["tids"], K5)
                # PLAN_RANK_ONLY == FULL_PLAN_D4 (D4 on P; no support pruning in D4)
                edges_plan = d4_edges_on_score(P, cell["sids"], cell["tids"], K5)
                for name, edges in (("COST_D4", edges_cost), ("SUPPORT_ONLY_D4", edges_support),
                                    ("PLAN_RANK_ONLY_D4", edges_plan),
                                    ("FULL_PLAN_D4", edges_plan)):
                    _df, summ = evaluate_edges(cell, edges)
                    rows.append({
                        "bridge": bridge, "seed": seed, "plan": plan_name, "variant": name,
                        "edge_f1": summ["edge_f1"], "edge_precision": summ["edge_precision"],
                        "edge_recall": summ["edge_recall"],
                        "fp_per_template": summ["edge_fp_total"] / 48,
                        "pred_edges_per_template": summ["n_pred_edges"],
                    })
            print(f"[support-vs-rank] {bridge} seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "support_vs_ranking.csv", index=False)
    agg = df.groupby(["bridge", "plan", "variant"]).mean(numeric_only=True).reset_index()
    agg.to_csv(out / "support_vs_ranking_aggregated.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
