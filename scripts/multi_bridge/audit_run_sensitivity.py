"""PHASE 8: UOT parameter sensitivity (one-factor-at-a-time around frozen; all reported)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "multi_bridge") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, run_variant  # noqa: E402


def main() -> None:
    rows: list[dict] = []
    rows += run_variant("sens_reg_0.025", reg=0.025, reg_m=0.5)
    rows += run_variant("sens_reg_0.10", reg=0.10, reg_m=0.5)
    rows += run_variant("sens_regm_0.25", reg=0.05, reg_m=0.25)
    rows += run_variant("sens_regm_1.0", reg=0.05, reg_m=1.0)
    df = pd.DataFrame(rows)
    (AUDIT / "sensitivity").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "sensitivity" / "uot_sensitivity_per_seed.csv", index=False)
    agg = df.groupby(["variant", "bridge"]).agg(
        split_mean=("split_recovery", "mean"), split_std=("split_recovery", "std"),
        merge_mean=("merge_recovery", "mean"), merge_std=("merge_recovery", "std"),
        n=("seed", "count"),
    ).reset_index()
    agg.to_csv(AUDIT / "sensitivity" / "uot_sensitivity_aggregated.csv", index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
