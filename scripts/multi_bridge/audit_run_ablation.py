"""PHASE 7: three-bridge feature ablations (leave-one-out + minimum defensible set)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "multi_bridge") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, run_variant, weights_loo, weights_minimum_defensible  # noqa: E402


def main() -> None:
    rows: list[dict] = []
    for comp in ("amount", "time", "route", "risk", "evidence", "novelty"):
        rows += run_variant(f"loo_{comp}", weights=weights_loo(comp))
    rows += run_variant("minimum_defensible_amount_time_route", weights=weights_minimum_defensible())
    df = pd.DataFrame(rows)
    (AUDIT / "ablation").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "ablation" / "structural_ablation_per_seed.csv", index=False)
    agg = df.groupby(["variant", "bridge"]).agg(
        split_mean=("split_recovery", "mean"), split_std=("split_recovery", "std"),
        merge_mean=("merge_recovery", "mean"), merge_std=("merge_recovery", "std"),
        n=("seed", "count"),
    ).reset_index()
    agg.to_csv(AUDIT / "ablation" / "structural_ablation_aggregated.csv", index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
