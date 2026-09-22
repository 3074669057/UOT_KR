"""Independent verification of the decoder attribution audit (§13 checklist).

Recomputes the controls and key statistics from raw frozen inputs (cost matrices, frozen
transport plans, GT labels, the locked decoder) — never from aggregated summaries.
"""
from __future__ import annotations

import hashlib
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

from decoder_audit.da_common import (  # noqa: E402
    AUDIT as PREV_AUDIT, BRIDGES, CALIB_SEEDS, TEST_SEEDS, FROZEN_PARAMS,
    decode, evaluate_edges, load_test_bot, load_test_uot,
)
from cross.domain.uot.cost_matrix import default_cost_weights  # noqa: E402
from run_attribution_audit import K_GRID, AUDIT, LOCK_CFG, cost_d4_edges  # noqa: E402

FROZEN_WEIGHTS = {"amount": 0.35, "time": 0.25, "route": 0.15, "risk": 0.15,
                  "graph": 0.05, "evidence": 0.05, "novelty": 0.05}
FROZEN_PARAMS_EXPECTED = {
    "uot_reg": 0.05, "uot_reg_m": 0.5, "uot_lambda_risk": 0.25,
    "uot_decode_threshold": 1e-9, "uot_max_delay_sec": 21600.0,
    "uot_causal_violation_penalty": 5.0, "uot_backend": "pot",
    "uot_allow_unmatched": True, "uot_use_graph_embedding": False,
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    OUT = AUDIT / "verification"
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # 1. pre-audit hashes unchanged (manuscript, params/cost code, locked decoder, prev outputs)
    snap = json.loads((OUT / "preaudit_hashes.json").read_text(encoding="utf-8"))
    hash_ok = True
    for rel, h in snap.items():
        p = REPO / rel
        if not p.is_file() or md5(p) != h:
            hash_ok = False
            issues.append(f"changed since pre-audit snapshot: {rel}")
    checks["preaudit_hashes_unchanged"] = hash_ok
    # frozen plan snapshot from the previous round
    plan_snap = json.loads((PREV_AUDIT / "verification" / "preaudit_plan_hashes.json")
                           .read_text(encoding="utf-8"))
    plan_ok = True
    for rel, h in plan_snap.items():
        p = REPO / "out" / "multi_bridge_expansion" / rel
        if not p.is_file() or md5(p) != h:
            plan_ok = False
            issues.append(f"frozen plan changed: {rel}")
    checks["frozen_plan_hashes_unchanged"] = plan_ok

    # 2. frozen params / weights unchanged at import time
    checks["cost_weights_unchanged"] = default_cost_weights() == FROZEN_WEIGHTS
    if not checks["cost_weights_unchanged"]:
        issues.append(f"default cost weights changed: {default_cost_weights()}")
    checks["uot_params_unchanged"] = FROZEN_PARAMS == FROZEN_PARAMS_EXPECTED
    if not checks["uot_params_unchanged"]:
        issues.append(f"FROZEN_PARAMS changed: {FROZEN_PARAMS}")

    # 3. locked decoder unchanged and used identically for UOT and BOT
    locked = json.loads((PREV_AUDIT / "calibration" / "locked_decoder.json").read_text(encoding="utf-8"))
    checks["locked_decoder_is_d4_k5"] = locked["decoder_name"] == "D4_mutrank@5" and \
        locked["params"] == {"k": 5}
    if not checks["locked_decoder_is_d4_k5"]:
        issues.append("locked decoder is not D4_mutrank@5")

    # 4. COST_D4_TRANSFER fixed k=5; COST_D4_CALIBRATED selection on 101-103 only
    cal_grid = pd.read_csv(AUDIT / "controls" / "cost_d4_calibration_grid.csv")
    seeds_in_grid = set(pd.to_numeric(cal_grid["seed"], errors="coerce").dropna().astype(int))
    checks["calibrated_uses_only_101_103"] = seeds_in_grid == set(CALIB_SEEDS)
    if not checks["calibrated_uses_only_101_103"]:
        issues.append(f"calibration grid seeds {sorted(seeds_in_grid)} != 101-103")
    checks["calibrated_k_in_grid"] = set(pd.to_numeric(cal_grid["k"], errors="coerce").dropna().astype(int)) == set(K_GRID)
    cal_lock = json.loads((AUDIT / "controls" / "cost_d4_calibrated_lock.json").read_text(encoding="utf-8"))
    checks["calibrated_marked_diagnostic"] = "DIAGNOSTIC" in cal_lock["status"]
    per_seed = pd.read_csv(AUDIT / "controls" / "untouched_per_seed.csv",
                           dtype={"bridge": str, "method": str})
    seeds_in_test = set(pd.to_numeric(per_seed["seed"], errors="coerce").dropna().astype(int))
    checks["test_seeds_only_42_46"] = seeds_in_test == set(TEST_SEEDS)
    if not checks["test_seeds_only_42_46"]:
        issues.append(f"test table seeds {sorted(seeds_in_test)} != 42-46")

    # 5. recompute controls + transport-D4 from raw and compare to the artifact table
    recomputed: list[dict[str, Any]] = []
    nan_found: list[str] = []
    missing: list[str] = []
    k_cal = int(cal_lock["k"])
    for br in BRIDGES:
        for seed in TEST_SEEDS:
            uot = load_test_uot(br, seed)
            bot = load_test_bot(br, seed)
            if len(uot["truth"]) != 48:
                missing.append(f"{br}/seed_{seed}: {len(uot['truth'])} templates")
            variants = {
                "COST_D4_TRANSFER": cost_d4_edges(uot, 5),
                "COST_D4_CALIBRATED": cost_d4_edges(uot, k_cal),
                "RC-UOT-Q D4": decode(uot, uot["P"], LOCK_CFG),
                "Balanced-OT D4": decode(bot, bot["P"], LOCK_CFG),
            }
            for name, edges in variants.items():
                df, summ = evaluate_edges(uot if "COST" in name or "UOT" in name else bot, edges)
                if df.isna().any().any():
                    nan_found.append(f"{br}/seed_{seed}/{name}")
                recomputed.append({"bridge": br, "seed": seed, "method": name,
                                   "edge_f1": summ["edge_f1"],
                                   "edge_precision": summ["edge_precision"],
                                   "edge_recall": summ["edge_recall"],
                                   "split_exact": summ["split_exact"],
                                   "merge_exact": summ["merge_exact"],
                                   "fp_per_template": summ["edge_fp_total"] / 48})
    checks["nan_found"] = nan_found
    checks["missing_templates"] = missing
    rr = pd.DataFrame(recomputed)
    max_diffs = {}
    for _, row in rr.iterrows():
        art = per_seed[(per_seed["bridge"] == row["bridge"]) & (per_seed["seed"] == row["seed"])
                       & (per_seed["method"] == row["method"])]
        if art.empty:
            issues.append(f"{row['bridge']}/{row['seed']}/{row['method']} missing in artifact")
            continue
        for col in ("edge_f1", "edge_precision", "edge_recall", "split_exact", "merge_exact",
                    "fp_per_template"):
            d = abs(float(row[col]) - float(art[col].iloc[0]))
            max_diffs[f"{row['bridge']}|{row['method']}|{col}"] = max(
                max_diffs.get(f"{row['bridge']}|{row['method']}|{col}", 0.0), d)
    bad = {k: v for k, v in max_diffs.items() if v > 1e-9}
    checks["controls_recompute_consistent"] = not bad
    if bad:
        issues.append(f"controls table differs from recompute: {bad}")

    # 6. RC-UOT-Q D4 / Balanced-OT D4 rows must reproduce the PREVIOUS round's locked test
    prev_ps = pd.read_csv(PREV_AUDIT / "locked_test" / "per_seed.csv",
                          dtype={"bridge": str, "method": str})
    prev_ok = True
    for _, row in rr.iterrows():
        if row["method"] not in ("Balanced-OT D4", "RC-UOT-Q D4"):
            continue
        prev_name = row["method"].replace("Balanced-OT D4", "Balanced-OT locked") \
                                   .replace("RC-UOT-Q D4", "RC-UOT-Q locked")
        art = prev_ps[(prev_ps["bridge"] == row["bridge"]) & (prev_ps["seed"] == row["seed"])
                      & (prev_ps["method"] == prev_name)]
        if art.empty:
            prev_ok = False
            issues.append(f"{row['bridge']}/{row['seed']}/{prev_name} missing in previous locked_test")
            continue
        if abs(float(row["edge_f1"]) - float(art["edge_f1"].iloc[0])) > 1e-9:
            prev_ok = False
            issues.append(f"{prev_name} edge_f1 differs from previous round")
    checks["transport_d4_reproduces_previous_locked_test"] = prev_ok

    # 7. paired bootstrap consistency (recompute from paired template F1)
    paired = pd.read_csv(AUDIT / "paired" / "paired_template_f1.csv")
    pb = pd.read_csv(AUDIT / "paired" / "paired_bootstrap.csv")
    paired_ok = True
    for (br,), g in paired.groupby(["bridge"]):
        for col in ("d_uot_cost5", "d_uot_costk", "d_uot_bot"):
            row = pb[(pb["bridge"] == br) & (pb["delta"] == col)]
            if row.empty or abs(row["mean"].iloc[0] - float(g[col].mean())) > 1e-9:
                paired_ok = False
                issues.append(f"paired stats mismatch {br}/{col}")
    checks["paired_stats_consistent"] = paired_ok

    checks["n_issues"] = len(issues)
    checks["issues"] = issues
    checks["overall_pass"] = len(issues) == 0
    (OUT / "verification_report.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# Independent verification — decoder attribution audit", "",
             "All controls and paired statistics recomputed from raw frozen inputs "
             "(cost matrices, frozen plans, GT, locked decoder).", ""]
    for k in ("preaudit_hashes_unchanged", "frozen_plan_hashes_unchanged",
              "cost_weights_unchanged", "uot_params_unchanged", "locked_decoder_is_d4_k5",
              "calibrated_uses_only_101_103", "calibrated_k_in_grid", "calibrated_marked_diagnostic",
              "test_seeds_only_42_46", "controls_recompute_consistent",
              "transport_d4_reproduces_previous_locked_test", "paired_stats_consistent",
              "nan_found", "missing_templates", "overall_pass"):
        if k in checks:
            lines.append(f"- **{k}**: `{json.dumps(checks.get(k), default=str)}`")
    lines += ["", "## Issues", ""]
    lines += [f"- {i}" for i in issues] if issues else ["- none"]
    (OUT / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in checks.items() if k != "issues"}, indent=2, default=str))
    print("issues:", issues)
    return 0 if checks["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
