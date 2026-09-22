"""PHASE 6 (close): combined counterfactual (decoy_off+aml_zero+evidence_const) = full frozen config."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "multi_bridge") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from audit_runner import AUDIT, run_variant  # noqa: E402


def main() -> None:
    rows = run_variant(
        "celer_counter_all_frozen",
        bridges=("Celer",),
        variant={"decoy_off": True, "aml_zero": True, "evidence_const": True},
    )
    df = pd.DataFrame(rows)
    df.to_csv(AUDIT / "celer_regression_delta" / "counterfactual_all_frozen_per_seed.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
