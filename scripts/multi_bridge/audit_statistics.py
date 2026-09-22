"""PHASE 11: statistical reporting — per-seed values, bootstrap 95% CI (seed-level unit)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges_audit"
MAIN = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
BRIDGES = ("Celer", "Multi", "Poly")


def bootstrap_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    return {"ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)),
            "n_boot": n_boot}


def main() -> None:
    rows: list[dict] = []
    out_rows: list[dict] = []
    for br in BRIDGES:
        sp, mg = [], []
        for seed in (42, 43, 44, 45, 46):
            ev = json.loads((MAIN / "per_seed" / br / f"seed_{seed}" / "eval" / "uot_evaluation_metrics.json").read_text(encoding="utf-8"))
            sp.append(ev["split_recovery_rate"])
            mg.append(ev["merge_recovery_rate"])
            rows.append({"bridge": br, "method": "RC-UOT-Q", "seed": seed,
                         "split_recovery": ev["split_recovery_rate"], "merge_recovery": ev["merge_recovery_rate"]})
        sp_a, mg_a = np.asarray(sp), np.asarray(mg)
        out_rows.append({
            "bridge": br, "method": "RC-UOT-Q",
            "split_mean": float(sp_a.mean()), "split_std": float(sp_a.std(ddof=1)),
            "split_ci95_low": bootstrap_ci(sp_a)["ci95_low"], "split_ci95_high": bootstrap_ci(sp_a)["ci95_high"],
            "merge_mean": float(mg_a.mean()), "merge_std": float(mg_a.std(ddof=1)),
            "merge_ci95_low": bootstrap_ci(mg_a)["ci95_low"], "merge_ci95_high": bootstrap_ci(mg_a)["ci95_high"],
            "n_seeds": 5, "statistical_unit": "seed (48 templates per seed)",
        })
    (AUDIT / "statistics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AUDIT / "statistics" / "rc_uot_q_per_seed.csv", index=False)
    df = pd.DataFrame(out_rows)
    df.to_csv(AUDIT / "statistics" / "bootstrap_ci.csv", index=False)
    print(df.to_string())

    # stronger-baseline paired (by seed) comparison vs RC-UOT-Q, if the baseline job finished
    sb = AUDIT / "stronger_baselines" / "stronger_baselines_per_seed.csv"
    if sb.is_file():
        sb_df = pd.read_csv(sb)
        rc = pd.DataFrame(rows)
        paired_rows = []
        for variant, g in sb_df.groupby("variant"):
            for br in BRIDGES:
                gb = g[g["bridge"] == br]
                rb = rc[rc["bridge"] == br]
                m = gb.merge(rb[["seed", "split_recovery", "merge_recovery"]], on="seed",
                             suffixes=("_base", "_rc"))
                for metric in ("split", "merge"):
                    d = m[f"{metric}_recovery_rc"] - m[f"{metric}_recovery_base"]
                    paired_rows.append({
                        "bridge": br, "variant": variant, "metric": f"{metric}_recovery",
                        "mean_diff_rc_minus_base": float(d.mean()),
                        "paired_seed_diff": [round(float(x), 4) for x in d],
                        "n_seeds": len(d),
                    })
        pd.DataFrame(paired_rows).to_csv(AUDIT / "statistics" / "paired_vs_stronger_baselines.csv", index=False)
        print("\npaired diffs (RC-UOT-Q minus baseline, per seed unit):")
        print(pd.DataFrame(paired_rows).to_string())


if __name__ == "__main__":
    main()
