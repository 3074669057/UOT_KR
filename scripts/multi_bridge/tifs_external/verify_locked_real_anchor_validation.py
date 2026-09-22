"""Independent verifier for the real-anchor external validation (DESIGN-ONLY round).

Re-checks the same hash/isolation gates as the runner, then recomputes every
aggregate from the raw per-component artifacts and requires agreement within 1e-9.
Never re-solves. Refuses any 301-305 dependency. In the current design-only round it
can only verify the GUARD behavior and toy preflight; the real verification path
activates after an approved execution exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tifs_external.real_anchor_common import scan_forbidden, verify_hash_manifest

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "out" / "multi_bridge_expansion" / "tifs_real_anchor_external_validation_preregistration_v2"
MANIFEST = PKG / "HASH_MANIFEST_v2.json"
RESULT_ROOT = REPO / "out" / "multi_bridge_expansion" / "tifs_real_anchor_external_validation_results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify an existing execution")
    cli = ap.parse_args()

    hits = scan_forbidden(str(PKG), str(RESULT_ROOT))
    if hits:
        print(f"ABORT: forbidden holdout dependency tokens found: {hits}")
        return 4
    errs = verify_hash_manifest(MANIFEST, REPO)
    if errs:
        print("ABORT: hash gate failed: " + "; ".join(errs))
        return 5
    if cli.verify:
        if not RESULT_ROOT.exists():
            print("NO EXECUTION TO VERIFY (result directory absent). Design-only round.")
            return 9
        print("VERIFICATION: real verification logic activates in the approval round.")
        return 8
    print("VERIFIER GUARD OK (isolation scan + hash gate pass; nothing to verify).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
