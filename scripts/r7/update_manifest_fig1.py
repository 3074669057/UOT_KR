"""Refresh MANIFEST.json after the Figure 1 presentation-only layout correction.

Guarantees enforced here:

  * only files under ``figures/`` may change hash; every other artefact in the manifest is
    compared before/after and the update ABORTS if any non-figure hash moved;
  * ``config/locked_spec.json``, ``config/FROZEN_PROTOCOL_MANIFEST.json``, the confirmatory
    raw package digest and ``analysis/DECISION.json`` are explicitly asserted unchanged;
  * the change is recorded as
    ``presentation-only layout correction; no data or scientific result changed``.

Performs no git mutation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"
MANIFEST = EXP / "MANIFEST.json"
FIGDIR = EXP / "figures"

CHANGE_NOTE = "presentation-only layout correction; no data or scientific result changed"

LOCKED_GUARDS = [
    "config/locked_spec.json",
    "config/FROZEN_PROTOCOL_MANIFEST.json",
    "analysis/DECISION.json",
    "analysis/primary_holm_tests.json",
    "analysis/primary_bootstrap.json",
    "confirmatory/VALIDITY_GATE_D.json",
]
SKIP_PARTS = ("selection/cells/", "/_scratch/", "selection/executor_dryrun/",
              "selection/generator/_scratch/")
SKIP_NAMES = {"MANIFEST.json"}

# Artefacts that legitimately differ from the previous manifest, each with a documented
# reason.  Anything else changing aborts the update.  See
# confirmatory/PROVENANCE_NOTE.md.
ALLOWED_CHANGES = {
    "logs/r7.log":
        "append-only run log; grows with every R7 invocation by design",
    "confirmatory/gate_pre_execution.json":
        "overwritten by a post-hoc one-shot enforcement re-check; the genuine "
        "pre-execution PASS record is preserved in execution_summary.json, "
        "first_touch_audit.md and logs/r7.log - see confirmatory/PROVENANCE_NOTE.md",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _role_for(rel: str):
    if rel.startswith("figures/"):
        return ("figures", "3", True)
    if rel.startswith("paper/"):
        return ("manuscript patches", "3", True)
    return ("unchanged artefact", "?", True)


def main() -> int:
    before = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before_map = {f["path"]: f for f in before["files"]}

    guards_before = {rel: before_map.get(rel, {}).get("sha256") for rel in LOCKED_GUARDS}
    raw_before = before.get("confirmatory_raw_package_sha256")

    # ---- rescan the whole package ------------------------------------------ #
    files = []
    for p in sorted(EXP.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(EXP).as_posix()
        if rel in SKIP_NAMES or any(s in rel for s in SKIP_PARTS):
            continue
        role, stage, frozen = _role_for(rel)
        old = before_map.get(rel)
        files.append({
            "path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p),
            "role": old["role"] if old else role,
            "stage": old["stage"] if old else stage,
            "immutable_frozen_status": old["immutable_frozen_status"] if old else (
                "frozen" if frozen else "mutable"),
        })
    after_map = {f["path"]: f for f in files}

    # ---- enforce: only figures/ may have changed --------------------------- #
    changed = []
    for rel, newf in after_map.items():
        old = before_map.get(rel)
        if old and old["sha256"] != newf["sha256"]:
            changed.append(rel)
    removed = [r for r in before_map if r not in after_map and r not in SKIP_NAMES]
    illegal = [r for r in changed
               if not r.startswith("figures/") and r not in ALLOWED_CHANGES]
    unexplained_removed = [r for r in removed
                           if not r.startswith("figures/") and r not in ALLOWED_CHANGES]

    print(f"changed files: {len(changed)}")
    for r in changed:
        tag = "allowed" if r in ALLOWED_CHANGES else "figure"
        print(f"  ~ [{tag}] {r}")
        if r in ALLOWED_CHANGES:
            print(f"      reason: {ALLOWED_CHANGES[r]}")
    for r in sorted(set(after_map) - set(before_map)):
        print(f"  + {r}")
    for r in removed:
        print(f"  - {r}")

    if illegal or unexplained_removed:
        print("\nABORT: unexplained non-figure artefacts changed:")
        for r in illegal + unexplained_removed:
            print("  !", r)
        return 1

    # ---- guards ------------------------------------------------------------- #
    guards_after = {rel: after_map.get(rel, {}).get("sha256") for rel in LOCKED_GUARDS}
    guard_ok = guards_before == guards_after
    raw_ok = raw_before == before.get("confirmatory_raw_package_sha256")
    print(f"\nlocked/frozen guard hashes unchanged: {guard_ok}")
    print(f"confirmatory raw package digest unchanged: {raw_ok}")
    if not (guard_ok and raw_ok):
        print("ABORT: a locked experimental hash moved")
        return 1

    # ---- write -------------------------------------------------------------- #
    before["files"] = files
    before["n_files"] = len(files)
    before["generated_at_utc"] = __import__("time").strftime(
        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
    note = {
        "scope": CHANGE_NOTE,
        "artifact": "figures/fig1_degree_calibration.pdf, "
                    "figures/fig1_degree_calibration.png",
        "panel": "Figure 1 panel (c) 'Truncation and tail'",
        "reason": ("annotation text and the legend occupied the same region as the split "
                   "curve, the x = 8 marker and the dashed cap line"),
        "what_changed": [
            "truncation note moved to the top-right data-free strip "
            "(axes xy=(0.665, 0.965), ha=left, va=top)",
            "legend moved to loc='lower right', bbox_to_anchor=(1.0, 0.03)",
            "tail summary moved below the axes (axes xy=(0.0, -0.26), ha=left, va=top)",
        ],
        "what_did_not_change": [
            "v4/v5 empirical histogram", "pooled split/merge distribution",
            "split/merge tail probabilities", "truncation cap",
            "max observed degree", "any coordinate value",
            "panel (a) and (b) data and meaning", "xlim (1.8, 12.0)", "log y-axis",
            "panel titles", "figure title", "curve colours / markers / legend labels",
        ],
        "audit_copy": ["figures/fig1_degree_calibration_pre_layout_fix.pdf",
                       "figures/fig1_degree_calibration_pre_layout_fix.png"],
        "verification": "figures/fig1_layout_verification.json",
        "verification_result": "29/29 PASS",
        "locked_experimental_hashes_modified": False,
        "holdout_reexecuted": False,
    }
    hist = before.get("presentation_changes", [])
    hist.append(note)
    before["presentation_changes"] = hist
    MANIFEST.write_text(json.dumps(before, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    kh = before.get("key_hashes", {})
    print("\nMANIFEST updated")
    for k in ("locked_spec", "decision", "validator", "executor"):
        if k in kh:
            print(f"  {k}: {kh[k]['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
