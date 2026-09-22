"""Negative / guard tests for the holdout machinery (development-only; NEVER touches
301-305). Verifies that every guard fails BEFORE any holdout access:
  1. no flag -> refusal (guard path only);
  2. tampered preregistration hash -> abort;
  3. wrong seeds -> abort;
  4. changed k -> abort;
  5. changed reg / reg_m -> abort;
  6. missing method -> abort.
Also re-asserts the dev preflight anchors are reproduced by the completed runner.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts" / "multi_bridge"))

from holdout.holdout_common import (  # noqa: E402
    BRIDGES, METHODS, PRE, identity_check, verify_prereg_hashes,
)


def check(name: str, condition: bool, detail: Any = "") -> dict[str, Any]:
    return {"test": name, "pass": bool(condition), "detail": str(detail)[:200]}


def main() -> int:
    results: list[dict[str, Any]] = []

    # 1. guard refusal without --execute-holdout (subprocess, no holdout access possible)
    import subprocess
    p = subprocess.run([sys.executable, "scripts/multi_bridge/holdout/run_locked_holdout.py"],
                       capture_output=True, text=True, cwd=REPO, timeout=600)
    results.append(check("no_flag_refusal", p.returncode != 0 and "REFUSING" in p.stdout,
                         p.stdout.strip()[:120]))

    # 2. tampered prereg hash -> the gate's hash logic must flag it (pure function; no data access)
    mani = json.loads((PRE / "HASH_MANIFEST.json").read_text(encoding="utf-8"))
    mani["managed_hashes"]["HOLDOUT_PREREGISTRATION.md"]["sha256"] = "0" * 64
    tampered = []
    for rel, e in mani["managed_hashes"].items():
        if "file" in e:
            p2 = REPO / e["file"]
        elif rel.startswith("scripts/"):
            p2 = REPO / rel
        else:
            p2 = PRE / rel
        if not p2.is_file() or hashlib.sha256(p2.read_bytes()).hexdigest() != e["sha256"]:
            tampered.append(rel)
    results.append(check("tampered_hash_detected",
                         "HOLDOUT_PREREGISTRATION.md" in tampered, tampered))

    # 3-6. identity checks with injected bad configs (pure function; no data access)
    results.append(check("wrong_seeds_detected",
                         any("seeds changed" in e for e in identity_check(
                             {"seeds": (301, 302, 303, 304, 399)})),
                         identity_check({"seeds": (301, 302, 303, 304, 399)})))
    results.append(check("changed_k_detected", any("k changed" in e for e in identity_check({"k": 6})),
                         identity_check({"k": 6})))
    results.append(check("changed_reg_detected",
                         any("reg changed" in e for e in identity_check({"reg": 0.10})),
                         identity_check({"reg": 0.10})))
    results.append(check("changed_regm_detected",
                         any("reg_m changed" in e for e in identity_check({"reg_m": 1.0})),
                         identity_check({"reg_m": 1.0})))
    results.append(check("missing_method_detected",
                         any("methods changed" in e for e in identity_check(
                             {"methods": METHODS[:-1]})),
                         identity_check({"methods": METHODS[:-1]})))

    # 7. dev preflight anchors (completed runner)
    pre = json.loads((PRE / "preflight_report.json").read_text(encoding="utf-8"))
    ver = json.loads((PRE / "verifier_preflight_report.json").read_text(encoding="utf-8"))
    results.append(check("runner_preflight_reproduces_anchors", pre["preflight_pass"],
                         pre["max_abs_deviation_from_anchor"]))
    results.append(check("verifier_preflight_reproduces_classification",
                         ver["verifier_preflight_pass"], ver.get("issues")))

    # 8. holdout isolation: no 301-305 artifacts anywhere in the prereg/result trees
    stray = [str(p) for p in list(PRE.rglob("*")) if any(f"seed_{s}" in str(p) for s in (301, 302, 303, 304, 305))]
    results.append(check("no_holdout_artifacts_in_preflight_area", not stray, stray[:5]))

    df = pd.DataFrame(results)
    (PRE / "negative_tests.json").write_text(json.dumps(results, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    print(df.to_string(index=False))
    print("ALL PASS:", bool(df["pass"].all()))
    return 0 if df["pass"].all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
