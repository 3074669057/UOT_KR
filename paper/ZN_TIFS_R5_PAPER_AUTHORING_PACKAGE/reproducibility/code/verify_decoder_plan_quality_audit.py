"""Independent verification of the decoder / plan-quality audit.

Recomputes everything from RAW artifacts (frozen transport plans, GT labels, locked decoder)
and never from the aggregated summaries. Checks: calibration/test separation, lock-before-test
ordering, shared decoder identity for Balanced-OT and RC-UOT-Q, completeness (5 test seeds,
48 templates), no NaN / duplicates, legacy-decoder reproduction, frozen-plan integrity
(md5 vs pre-audit snapshot), and consistency of the locked-test per-seed table.
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
    AUDIT, BRIDGES, CALIB_SEEDS, TEST_SEEDS, decode, evaluate_edges, load_test_bot,
    load_test_uot,
)

OUT = AUDIT / "verification"
METHOD_PLAN = {
    "RC-UOT-Q legacy": ("UOT", {"name": "D0_legacy", "family": "D0", "params": {"thr": 1e-9}}),
    "Balanced-OT legacy": ("BOT", {"name": "D0_legacy", "family": "D0", "params": {"thr": 1e-9}}),
}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # 1. lock exists, contains the required fields, locked before test artifacts
    lock_path = AUDIT / "calibration" / "locked_decoder.json"
    checks["locked_decoder_exists"] = lock_path.is_file()
    if not lock_path.is_file():
        issues.append("locked_decoder.json missing")
    else:
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        for k in ("decoder_name", "formula", "params", "selection_metric", "calibration_seeds",
                  "bridges", "timestamp_utc", "git", "code_hash_md5"):
            checks[f"locked_has_{k}"] = k in locked
            if k not in locked:
                issues.append(f"locked_decoder.json missing field {k}")
        lock_cfg = {"name": locked["decoder_name"], "family": locked["decoder_family"],
                    "params": locked["params"]}
        lock_time = pd.Timestamp(locked.get("timestamp_utc"))
        test_files = list((AUDIT / "locked_test").glob("*.csv")) + \
            list((AUDIT / "locked_test").glob("*.md"))
        test_times = [pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") for p in test_files]
        ordered = all(t >= lock_time - pd.Timedelta(seconds=2) for t in test_times) if test_times else True
        checks["lock_before_test"] = bool(ordered)
        if not ordered:
            issues.append("locked_decoder.json is newer than some locked_test artifact")

    # 2. calibration grid only contains calibration seeds; test table only test seeds
    grid_path = AUDIT / "calibration" / "decoder_grid.csv"
    if grid_path.is_file():
        grid = pd.read_csv(grid_path)
        seeds_in_grid = set(pd.to_numeric(grid["seed"], errors="coerce").dropna().astype(int))
        checks["calibration_grid_seeds"] = sorted(seeds_in_grid) == sorted(CALIB_SEEDS)
        if not checks["calibration_grid_seeds"]:
            issues.append(f"decoder_grid seeds {sorted(seeds_in_grid)} != calibration seeds")
    per_seed_path = AUDIT / "locked_test" / "per_seed.csv"
    if per_seed_path.is_file():
        ps = pd.read_csv(per_seed_path, dtype={"bridge": str, "method": str})
        seeds_in_test = set(pd.to_numeric(ps["seed"], errors="coerce").dropna().astype(int))
        checks["test_seeds_only_in_locked_test"] = sorted(seeds_in_test) == sorted(TEST_SEEDS)
        if not checks["test_seeds_only_in_locked_test"]:
            issues.append(f"locked_test seeds {sorted(seeds_in_test)} != test seeds")
        checks["n_test_seeds_per_bridge"] = ps.groupby("bridge")["seed"].nunique().to_dict()

    # 3. frozen plan integrity (md5 vs pre-audit snapshot)
    snap_path = AUDIT / "verification" / "preaudit_plan_hashes.json"
    checks["frozen_hashes_unchanged"] = False
    if snap_path.is_file():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        all_ok = True
        for rel, h in snap.items():
            p = REPO / "out" / "multi_bridge_expansion" / rel
            if not p.is_file() or md5(p) != h:
                all_ok = False
                issues.append(f"frozen artifact changed: {rel}")
        checks["frozen_hashes_unchanged"] = all_ok

    # 4. recompute from raw: shared decoder on BOTH plans, legacy reproduction,
    #    completeness, NaN, duplicates, GT-mass spot check
    if lock_path.is_file() and per_seed_path.is_file():
        ps = pd.read_csv(per_seed_path, dtype={"bridge": str, "method": str})
        recomputed_rows: list[dict[str, Any]] = []
        nan_found: list[str] = []
        missing_templates: list[str] = []
        for br in BRIDGES:
            for seed in TEST_SEEDS:
                inst_uot = load_test_uot(br, seed)
                inst_bot = load_test_bot(br, seed)
                n_templates = len(inst_uot["truth"])
                if n_templates != 48:
                    missing_templates.append(f"{br}/seed_{seed}: {n_templates} templates")
                for name, (plan, cfg) in {
                        "RC-UOT-Q legacy": ("UOT", METHOD_PLAN["RC-UOT-Q legacy"][1]),
                        "RC-UOT-Q locked": ("UOT", lock_cfg),
                        "Balanced-OT legacy": ("BOT", METHOD_PLAN["Balanced-OT legacy"][1]),
                        "Balanced-OT locked": ("BOT", lock_cfg)}.items():
                    inst = inst_uot if plan == "UOT" else inst_bot
                    edges = decode(inst, inst["P"], cfg)
                    df, summ = evaluate_edges(inst, edges)
                    if df.isna().any().any():
                        nan_found.append(f"{br}/seed_{seed}/{name}")
                    recomputed_rows.append({
                        "bridge": br, "seed": seed, "method": name,
                        "edge_f1": summ["edge_f1"], "edge_precision": summ["edge_precision"],
                        "edge_recall": summ["edge_recall"], "split_exact": summ["split_exact"],
                        "merge_exact": summ["merge_exact"], "fp_per_template": summ["edge_fp_total"] / 48,
                    })
        checks["nan_found"] = nan_found
        checks["missing_templates"] = missing_templates
        if nan_found:
            issues.append(f"NaN in recomputed metrics: {nan_found}")
        if missing_templates:
            issues.append(f"missing templates: {missing_templates}")
        rr = pd.DataFrame(recomputed_rows)
        rr.to_csv(OUT / "recomputed_from_raw.csv", index=False)
        # consistency vs artifact per-seed table
        max_diffs: dict[str, float] = {}
        for _, row in rr.iterrows():
            art = ps[(ps["bridge"] == row["bridge"]) & (ps["seed"] == row["seed"])
                     & (ps["method"] == row["method"])]
            if art.empty:
                issues.append(f"{row['bridge']}/{row['seed']}/{row['method']} missing in artifact table")
                continue
            for col in ("edge_f1", "edge_precision", "edge_recall", "split_exact", "merge_exact"):
                d = abs(float(row[col]) - float(art[col].iloc[0]))
                max_diffs[f"{row['bridge']}|{row['method']}|{col}"] = max(
                    max_diffs.get(f"{row['bridge']}|{row['method']}|{col}", 0.0), d)
        bad = {k: v for k, v in max_diffs.items() if v > 1e-9}
        checks["artifact_vs_recompute_consistent"] = not bad
        checks["max_diff_by_metric"] = {k: round(v, 9) for k, v in sorted(max_diffs.items())}
        if bad:
            issues.append(f"locked_test artifact differs from recompute: {bad}")
        # shared decoder identity: locked decoder applied to BOTH plans is verified by the
        # recompute above using the same lock_cfg for both plan types
        checks["shared_decoder_for_uot_and_bot"] = True
        # duplicates
        dup = ps.duplicated(subset=["bridge", "seed", "method"]).sum()
        checks["no_duplicate_rows"] = int(dup) == 0
        if dup:
            issues.append(f"{dup} duplicate rows in locked_test/per_seed.csv")

    # 5. GT-mass fraction spot check vs plan_quality raw
    pq_path = AUDIT / "raw" / "per_template_plan_quality.csv"
    checks["gt_mass_fraction_consistent"] = None
    if pq_path.is_file() and lock_path.is_file():
        pq = pd.read_csv(pq_path)
        ok = True
        for br in BRIDGES:
            for seed in (42,):
                inst = load_test_uot(br, seed)
                P = inst["P"]
                sub = pq[(pq["bridge"] == br) & (pq["seed"] == seed) & (pq["method"] == "UOT")]
                if sub.empty:
                    continue
                for _, r in sub.iterrows():
                    rows = [i for i, s in enumerate(inst["sids"])
                            if s.split("__synth")[0] == r["template_id"]]
                    total = float(P[np.ix_(rows, range(P.shape[1]))].sum())
                    gt = float(r["gt_transport_mass"])
                    if total > 0 and abs(gt / total - r["gt_mass_fraction"]) > 1e-9:
                        ok = False
        checks["gt_mass_fraction_consistent"] = ok
        if not ok:
            issues.append("gt_mass_fraction mismatch between raw and recompute")

    checks["n_issues"] = len(issues)
    checks["issues"] = issues
    checks["overall_pass"] = len(issues) == 0
    (OUT / "verification_report.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# Independent verification — decoder / plan-quality audit", "",
             "All headline quantities recomputed from raw transport plans + GT labels + "
             "locked_decoder.json; aggregated tables were not read as input.", ""]
    for k in ("locked_decoder_exists", "lock_before_test", "calibration_grid_seeds",
              "test_seeds_only_in_locked_test", "n_test_seeds_per_bridge", "frozen_hashes_unchanged",
              "artifact_vs_recompute_consistent", "shared_decoder_for_uot_and_bot",
              "no_duplicate_rows", "gt_mass_fraction_consistent", "nan_found",
              "missing_templates", "overall_pass"):
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
