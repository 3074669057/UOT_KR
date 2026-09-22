"""PHASE 4: USD price perturbation robustness (pre-registered, all values reported)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "multi_bridge") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, run_variant  # noqa: E402


def main() -> None:
    rows: list[dict] = []
    # uniform scaling (invariance sanity: ratio-based amount cost + normalized masses)
    for f in (0.8, 0.9, 1.1, 1.2):
        rows += run_variant(f"price_uniform_{f}", variant={"price_uniform": f})
    # per-token random multiplicative perturbation (asset-relative stress), 3 random seeds
    for rs in (1, 2, 3):
        rows += run_variant(f"price_per_token_rand{rs}", variant={"price_per_token_seed": rs})
    df = pd.DataFrame(rows)
    (AUDIT / "price_robustness").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "price_robustness" / "price_perturbation_per_seed.csv", index=False)
    agg = df.groupby(["variant", "bridge"]).agg(
        split_mean=("split_recovery", "mean"), split_std=("split_recovery", "std"),
        merge_mean=("merge_recovery", "mean"), merge_std=("merge_recovery", "std"),
        n=("seed", "count"),
    ).reset_index()
    agg.to_csv(AUDIT / "price_robustness" / "price_perturbation_aggregated.csv", index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
