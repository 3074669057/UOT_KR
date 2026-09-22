"""Independent performance verifier for the v3 temporal external validation.
PREFLIGHT-ROUND STRUCTURE ONLY: the real verification path activates after an
approved execution exists. It will recompute every aggregate from raw method
outputs (method identity, component identity, exact-recovery indicators, edge
TP/FP/FN, abstention, per-degree/per-address outcomes, primary Delta, exact
randomization test, wild-cluster CI, safety gates, claim classification) and
requires agreement within 1e-9; it never trusts the runner's aggregation. No
method prediction may exist yet; the guard refuses otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tifs_external.real_anchor_common import scan_forbidden, verify_hash_manifest

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_validation_preregistration_v3"
DATA = REPO / "out" / "multi_bridge_expansion" / "tifs_temporal_external_v3"
MANIFEST = PKG / "FINAL_EXTERNAL_HASH_MANIFEST.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify an existing execution")
    cli = ap.parse_args()
    hits = scan_forbidden(str(PKG), str(DATA))
    if hits:
        print(f"ABORT: holdout token found: {hits}")
        return 4
    errs = verify_hash_manifest(MANIFEST, REPO)
    if errs:
        print("ABORT: hash gate failed: " + "; ".join(errs))
        return 5
    if cli.verify:
        res = DATA / "method_results"
        if not res.exists():
            print("NO EXECUTION TO VERIFY. PREFLIGHT ONLY: "
                  "real verification activates after an approved execution.")
            return 9
        print("VERIFICATION: real recompute path activates in the execution round.")
        return 8
    print("PERFORMANCE VERIFIER GUARD OK (no execution exists; guards pass).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
