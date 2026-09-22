"""PHASE 6: Celer frozen-vs-new counterfactuals (which change flipped the one edge)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "multi_bridge") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, run_variant  # noqa: E402

CELER = ("Celer",)


def main() -> None:
    rows: list[dict] = []
    rows += run_variant("celer_counter_decoy_off", bridges=CELER, variant={"decoy_off": True})
    rows += run_variant("celer_counter_aml_zero", bridges=CELER, variant={"aml_zero": True})
    rows += run_variant("celer_counter_evidence_const", bridges=CELER, variant={"evidence_const": True})
    df = pd.DataFrame(rows)
    (AUDIT / "celer_regression_delta").mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT / "celer_regression_delta" / "counterfactual_per_seed.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
