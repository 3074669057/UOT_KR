"""PHASE 5: PolyNetwork saturation diagnosis — single-feature, difficulty variants."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "multi_bridge") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, run_variant, weights_only  # noqa: E402

POLY = ("Poly",)


def main() -> None:
    rows: list[dict] = []
    # single-feature-only (which feature alone can do the job)
    for comp in ("amount", "time", "route", "risk", "evidence", "novelty"):
        rows += run_variant(f"poly_only_{comp}", bridges=POLY, weights=weights_only(comp))
    # difficulty: timestamp noise
    for sig in (60.0, 120.0, 240.0):
        rows += run_variant(f"poly_tsnoise_{int(sig)}", bridges=POLY, variant={"ts_noise_std": sig})
    # difficulty: amount noise
    for lv in (0.05, 0.10, 0.20):
        rows += run_variant(f"poly_amtnoise_{int(lv*100)}pct", bridges=POLY, variant={"amt_noise_level": lv})
    # difficulty: decoy density
    for k in (2, 4):
        rows += run_variant(f"poly_decoy_{k}x", bridges=POLY, variant={"decoy_mult": k})
    # difficulty: unmatched ratio
    for k in (2, 3):
        rows += run_variant(f"poly_unmatched_{k}x", bridges=POLY, variant={"unmatched_mult": k})
    df = pd.DataFrame(rows)
    (AUDIT / "poly_saturation").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "poly_saturation" / "poly_variants_per_seed.csv", index=False)
    agg = df.groupby(["variant", "bridge"]).agg(
        split_mean=("split_recovery", "mean"), split_std=("split_recovery", "std"),
        merge_mean=("merge_recovery", "mean"), merge_std=("merge_recovery", "std"),
        n=("seed", "count"),
    ).reset_index()
    agg.to_csv(AUDIT / "poly_saturation" / "poly_variants_aggregated.csv", index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
