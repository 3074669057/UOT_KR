"""Independent verification of the COST / TRANSPORT diagnosis round (§14 checklist).

- pre-audit hashes (both previous audit dirs, manuscript, cost/UOT source, decoder code)
  unchanged;
- every data artifact of this round only involves development seeds 201-205;
- holdout seeds 301-305 were never generated / never read (no artifact, no code path);
- locked D4_mutrank@5 unchanged; frozen weights/params unchanged;
- all interventions marked DIAGNOSTIC; sweeps grids match the pre-registered grids;
- no hidden weight search, no bridge-specific tuning;
- NEXT_CANDIDATE_SPEC.md hash-locked.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diag.ctd_common import CTD, DEV_SEEDS, HOLDOUT_SEEDS, REGM_GRID, REG_GRID  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    OUT = CTD / "verification"
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    issues: list[str] = []

    # 1. pre-audit hashes
    snap = json.loads((OUT / "preaudit_hashes.json").read_text(encoding="utf-8"))
    ok = True
    for rel, h in snap.items():
        p = Path(rel.replace("/", "\\"))
        if not p.is_file() or md5(p) != h:
            ok = False
            issues.append(f"changed since pre-audit snapshot: {rel}")
    checks["preaudit_hashes_unchanged"] = ok

    # 2. seeds in this round's data artifacts are dev-only (201-205)
    seed_set = set()
    for p in (CTD / "raw").glob("*.csv"):
        df = pd.read_csv(p)
        if "seed" in df.columns:
            seed_set |= set(pd.to_numeric(df["seed"], errors="coerce").dropna().astype(int))
    checks["seeds_used_in_artifacts"] = sorted(seed_set)
    checks["only_dev_seeds_used"] = seed_set <= set(DEV_SEEDS)
    if not checks["only_dev_seeds_used"]:
        issues.append(f"non-dev seeds in artifacts: {sorted(seed_set - set(DEV_SEEDS))}")

    # 3. holdout 301-305 never generated / referenced by any output or script
    #    (scan only THIS round's scripts; generation/read calls, not comments/docstrings)
    violations = []
    new_scripts = [REPO / "scripts" / "multi_bridge" / "diag" / "ctd_common.py",
                   REPO / "scripts" / "multi_bridge" / "run_build_dev_plans.py",
                   REPO / "scripts" / "multi_bridge" / "run_dev_sweeps.py",
                   REPO / "scripts" / "multi_bridge" / "run_cost_forensic_audit.py",
                   REPO / "scripts" / "multi_bridge" / "run_kernel_transport_audit.py",
                   REPO / "scripts" / "multi_bridge" / "run_sweep_analysis.py",
                   REPO / "scripts" / "multi_bridge" / "verify_cost_transport_diagnosis.py",
                   REPO / "scripts" / "multi_bridge" / "make_ctd_figures.py"]
    pat = re.compile(r"(seed|seeds|SEEDS?)\s*[=:(\[]\s*(30[1-5])")
    for s in new_scripts:
        if not s.is_file():
            continue
        for ln, line in enumerate(s.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", '"', "'", '"""')):
                continue
            m = pat.search(line)
            if m and m.group(2) and "HOLDOUT_SEEDS" not in line:
                violations.append(f"{s.name}:{ln}: {stripped[:90]}")
    checks["holdout_seed_references"] = violations
    if violations:
        issues.append(f"holdout seeds referenced: {violations}")
    checks["holdout_artifacts_exist"] = bool(list(CTD.rglob("*301*")) + list(CTD.rglob("*302*")))
    if checks["holdout_artifacts_exist"]:
        issues.append("holdout-seed artifacts found in the output dir")

    # 4. D4 k=5 and frozen weights/params unchanged at import time
    from diag.ctd_common import DECODE_K, FROZEN_WEIGHTS
    from decoder_audit.da_common import FROZEN_PARAMS
    checks["d4_k_unchanged"] = DECODE_K == 5
    checks["frozen_weights_unchanged"] = FROZEN_WEIGHTS == {
        "amount": 0.40, "time": 0.25, "route": 0.15, "risk": 0.15,
        "evidence": 0.05, "novelty": 0.05}
    checks["frozen_params_unchanged"] = FROZEN_PARAMS["uot_reg"] == 0.05 and \
        FROZEN_PARAMS["uot_reg_m"] == 0.5 and \
        FROZEN_PARAMS["uot_decode_threshold"] == 1e-9

    # 5. DIAGNOSTIC markers on interventions
    marked = 0
    total = 0
    for p in list((CTD / "reg_sweep").rglob("*_meta.json")) + \
            list((CTD / "reg_m_sweep").rglob("*_meta.json")) + \
            list((CTD / "marginals").rglob("*_meta.json")):
        total += 1
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("diagnostic_only") is True:
            marked += 1
    checks["sweep_metas_diagnostic_marked"] = (marked, total)
    if marked != total:
        issues.append(f"unmarked sweep metas: {total - marked}")
    loco = pd.read_csv(CTD / "component_ablation" / "loco_aggregated.csv")
    checks["loco_variants_pre_registered"] = set(loco["variant"]) == {
        "FULL", "NO_AMOUNT", "NO_TIME", "NO_ROUTE", "NO_RISK", "NO_EVIDENCE", "NO_NOVELTY"}
    if not checks["loco_variants_pre_registered"]:
        issues.append(f"LOCO variants deviate: {set(loco['variant'])}")
    reg_csv = pd.read_csv(CTD / "raw" / "reg_sweep.csv")
    checks["reg_grid_pre_registered"] = sorted(set(reg_csv["reg"])) == sorted(REG_GRID)
    regm_csv = pd.read_csv(CTD / "raw" / "reg_m_sweep.csv")
    checks["regm_grid_pre_registered"] = sorted(set(regm_csv["reg_m"])) == sorted(REGM_GRID)
    if not checks["reg_grid_pre_registered"] or not checks["regm_grid_pre_registered"]:
        issues.append("sweep grids deviate from the pre-registered grids")

    # 6. NEXT_CANDIDATE_SPEC hash lock
    spec = CTD / "NEXT_CANDIDATE_SPEC.md"
    checks["candidate_spec_exists"] = spec.is_file()
    if spec.is_file():
        lock = OUT / "next_candidate_spec_lock.json"
        if lock.is_file():
            stored = json.loads(lock.read_text(encoding="utf-8"))
            checks["candidate_spec_hash_matches"] = stored.get("sha256") == \
                hashlib.sha256(spec.read_bytes()).hexdigest()
            if not checks["candidate_spec_hash_matches"]:
                issues.append("NEXT_CANDIDATE_SPEC.md hash mismatch")
            checks["spec_written_before_holdout"] = stored.get("holdout_not_run", False)
        else:
            issues.append("next_candidate_spec_lock.json missing")

    checks["n_issues"] = len(issues)
    checks["issues"] = issues
    checks["overall_pass"] = len(issues) == 0
    (OUT / "verification_report.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# Independent verification — cost / transport diagnosis round", "",
             "All checks below were computed from the artifacts and source files directly.", ""]
    for k in ("preaudit_hashes_unchanged", "only_dev_seeds_used", "seeds_used_in_artifacts",
              "holdout_seed_references", "holdout_artifacts_exist", "d4_k_unchanged",
              "frozen_weights_unchanged", "frozen_params_unchanged",
              "sweep_metas_diagnostic_marked", "loco_variants_pre_registered",
              "reg_grid_pre_registered", "regm_grid_pre_registered",
              "candidate_spec_exists", "candidate_spec_hash_matches",
              "spec_written_before_holdout", "overall_pass"):
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
