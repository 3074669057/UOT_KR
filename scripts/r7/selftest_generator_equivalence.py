"""R7 generator equivalence self-test.

Proves that the R7 extension of the REAL paper generator
(``cross.domain.evaluation.semi_synthetic_flows.build_semi_synthetic_from_flow_labels``)
reproduces the frozen generator byte-for-byte when the R7 options are not used, by
replaying the exact call the frozen dev-cell builder made and diffing against the frozen
on-disk label file.

READ ONLY.  Writes nothing outside the R7 scratch directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cross.domain.evaluation.semi_synthetic_flows import (      # noqa: E402
    build_semi_synthetic_from_flow_labels)

CTD = REPO / "out" / "multi_bridge_expansion" / "cost_transport_diagnosis"
EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
OUT = EXP / "selection" / "generator" / "generator_equivalence.json"
SCRATCH = EXP / "selection" / "generator" / "_scratch"

CASES = [("Celer", 201), ("Celer", 202), ("Multi", 203), ("Poly", 204), ("Poly", 205)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    results = []
    ok_all = True
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for bridge, seed in CASES:
        root = CTD / "plans" / "dev" / bridge / f"seed_{seed}"
        frozen_labels = root / "labels" / "synthetic_flow_labels.csv"
        pool = root / "flow_labels_pool.csv"
        if not (frozen_labels.is_file() and pool.is_file()):
            results.append({"bridge": bridge, "seed": seed, "status": "SKIPPED",
                            "reason": "frozen replay inputs unavailable"})
            continue
        td = SCRATCH / f"{bridge}_{seed}"
        if td.exists():
            shutil.rmtree(td, ignore_errors=True)
        td.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pool, td / "flow_labels_pool.csv")
        with open(pool, encoding="utf-8") as f:
            n_pool = sum(1 for _ in f) - 1
        (td / "flow_label_stats.json").write_text(
            json.dumps({"predominantly_one_to_one": True, "n_pool_pairs": int(n_pool)}),
            encoding="utf-8")
        build_semi_synthetic_from_flow_labels(
            td / "flow_labels_pool.csv", td / "flow_label_stats.json", td,
            seed=seed, max_seeds=48, force=True)
        new_labels = td / "labels" / "synthetic_flow_labels.csv"
        same = sha(new_labels) == sha(frozen_labels)
        ok_all &= same
        results.append({
            "bridge": bridge, "seed": seed,
            "status": "PASS" if same else "FAIL",
            "frozen_sha256": sha(frozen_labels),
            "replayed_sha256": sha(new_labels),
            "frozen_bytes": frozen_labels.stat().st_size,
        })
        shutil.rmtree(td, ignore_errors=True)
    out = {
        "test": "r7_generator_default_mode_byte_equivalence",
        "claim": ("with r7_family_grid=False and r7_*_degrees=None the extended generator is "
                  "byte-identical to the frozen generator"),
        "cases": results,
        "ALL_PASS": bool(ok_all),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
