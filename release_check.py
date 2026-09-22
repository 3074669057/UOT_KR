#!/usr/bin/env python3
"""release_check.py -- verify this artifact release from a fresh clone, with no external data.

This script is the entry point an artifact reviewer should run first.  It performs the
checks that are possible using ONLY the files shipped in this repository, and reports
explicitly which checks are not possible here and why.

Usage
-----
    python release_check.py                     # run every offline check
    python release_check.py --json out.json      # also write a machine-readable report

Exit status
-----------
    0  every offline check passed
    1  at least one offline check failed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
R7 = REPO / "out" / "r7_confirmatory_kernel_ranking_20260917"

# The validator stamps generated_at_utc / runtime_sec, so its report is written to a
# scratch file and deleted again -- never to the hash-locked confirmatory/ location.
_TMP_REPORT = REPO / ".release_check_validator.json"

RESULTS: list[dict] = []

# Windows consoles default to a legacy codepage that cannot encode every path in this
# release; make reporting robust so an encoding error cannot look like a check failure.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"{'PASS' if ok else 'FAIL':4}  {name}: {detail}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
def check_layout() -> None:
    """The files the README promises must exist."""
    required = [
        "src/cross",
        "scripts",
        "tests",
        "config",
        "schemas",
        "docs",
        "paper/ZN_TIFS_CN_R11_SUBMISSION_READY.docx",
        "out/r5_posthoc_hparam_sensitivity_20260917/FINAL_EXPERIMENT_REPORT.md",
        "out/r6_posthoc_kernel_k_control_20260917/FINAL_EXPERIMENT_REPORT.md",
        "out/r7_confirmatory_kernel_ranking_20260917/FINAL_EXPERIMENT_REPORT.md",
        "out/r7_confirmatory_kernel_ranking_20260917/BLOCKER_REPORT.md",
        "audit/risk_field/FINAL_RISK_AUDIT_REPORT.md",
    ]
    missing = [r for r in required if not (REPO / r).exists()]
    record("layout_required_paths", not missing,
           "all present" if not missing else f"missing: {missing}")


def check_manuscript_hash() -> None:
    """The submission-ready manuscript must match the hash pinned in RELEASE_SHA256SUMS.txt."""
    sums = REPO / "RELEASE_SHA256SUMS.txt"
    if not sums.exists():
        record("manuscript_hash", False, "RELEASE_SHA256SUMS.txt not found")
        return
    target = "paper/ZN_TIFS_CN_R11_SUBMISSION_READY.docx"
    want = None
    for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip().lstrip("*") == target:
            want = parts[0]
            break
    if want is None:
        record("manuscript_hash", False, f"{target} not listed in RELEASE_SHA256SUMS.txt")
        return
    got = sha256(REPO / target)
    record("manuscript_hash", got == want, f"sha256 {got[:16]}... (expected {want[:16]}...)")


def check_locked_protocol() -> None:
    """The frozen R7 protocol hash must be the one the report and validator cite."""
    spec = R7 / "config" / "locked_spec.json"
    if not spec.exists():
        record("r7_locked_protocol", False, "config/locked_spec.json missing")
        return
    got = sha256(spec)
    expect = "9775479b742833d02beeffb469569b0d885002749b84e996844dcce9b845e11d"
    record("r7_locked_protocol", got == expect,
           f"{got[:16]}... (expected {expect[:16]}...)")


def check_confirmatory_present() -> None:
    raw = R7 / "confirmatory" / "raw" / "units"
    ledger = R7 / "confirmatory" / "CONFIRMATORY_TOUCH_ONCE__411_420.json"
    units = sorted(raw.glob("unit__*.json")) if raw.is_dir() else []
    record("r7_confirmatory_units", len(units) == 30,
           f"{len(units)} unit files (expected 30)")
    record("r7_one_shot_ledger", ledger.exists(),
           "ledger present -> the one-shot holdout has been spent; re-execution is refused")


def check_manifest_frozen_files() -> None:
    """Verify every file listed in the frozen R7 MANIFEST against its recorded SHA256.

    Entries that live inside one of the packaged evidence archives are reported by
    archive name; entries that are genuinely absent are reported (the manifest also
    covers .npz intermediates that this release excludes by design).
    """
    import zipfile

    manifest = R7 / "MANIFEST.json"
    if not manifest.exists():
        record("r7_manifest_hashes", False, "MANIFEST.json missing")
        return
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8")).get("files", [])
    except Exception as exc:  # noqa: BLE001
        record("r7_manifest_hashes", False, f"unreadable: {exc}")
        return

    # map archive -> set of member paths, so packaged files count as present.
    #
    # The archives store members relative to the packaged directory: the R7 selection
    # pack holds `data_manifest.json` and `cells/...`, while MANIFEST.json lists those
    # same files as `selection/data_manifest.json` and `selection/cells/...`.  Each
    # member is therefore registered under its full path, its first component stripped,
    # and its last two components, so a manifest path of `selection/x/y` matches the
    # archive member `x/y` as well as `y`.
    packed: dict[str, set[str]] = {}
    for zp in (REPO / "out").rglob("*_evidence.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                members: set[str] = set()
                for n in zf.namelist():
                    n = n.replace("\\", "/").lstrip("./")
                    members.add(n)
                    parts = n.split("/")
                    if len(parts) > 1:
                        members.add("/".join(parts[1:]))
                    if len(parts) > 2:
                        members.add("/".join(parts[2:]))
                packed[zp.name] = members
        except Exception:  # noqa: BLE001
            packed[zp.name] = set()

    def _in_archive(rel: str) -> str | None:
        """Return the archive holding `rel`, allowing for one packaging-prefix shift."""
        candidates = [rel]
        parts = rel.split("/")
        if len(parts) > 1:
            candidates.append("/".join(parts[1:]))
        for name, members in packed.items():
            for cand in candidates:
                if cand in members:
                    return name
        return None

    ok = mismatch = packaged = npz_skipped = 0
    problems: list[str] = []
    for it in entries:
        rel = str(it.get("path", ""))
        want = str(it.get("sha256", "")).lower()
        target = R7 / rel
        if not target.is_file():
            if rel.endswith(".npz"):
                npz_skipped += 1
                continue
            holder = _in_archive(rel)
            if holder:
                packaged += 1
                continue
            problems.append(f"absent: {rel}")
            continue
        if sha256(target) == want:
            ok += 1
        else:
            mismatch += 1
            problems.append(f"MISMATCH: {rel}")

    detail = (f"{ok} verified, {packaged} inside archives, {npz_skipped} excluded .npz, "
              f"{mismatch} mismatch, {len(problems) - mismatch} absent")
    record("r7_manifest_hashes", mismatch == 0 and (len(problems) - mismatch) == 0, detail)
    for p in problems[:10]:
        print(f"      {p}")


def _snapshot_manifest_files() -> dict[str, str]:
    manifest = R7 / "MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8")).get("files", [])
    except Exception:  # noqa: BLE001
        return {}
    snap: dict[str, str] = {}
    for it in entries:
        rel = str(it.get("path", ""))
        p = R7 / rel
        if p.is_file():
            snap[rel] = sha256(p)
    return snap


def check_validator() -> None:
    """Run the frozen independent validator; it recomputes every metric from primitives.

    `--out` is redirected to a scratch path on purpose: the validator stamps
    `generated_at_utc` and `runtime_sec`, so writing to its default location would
    overwrite the hash-locked `confirmatory/VALIDITY_GATE_E.json` and break the manifest.
    """
    script = REPO / "scripts" / "validate_r7_confirmatory_results.py"
    if not script.exists():
        record("r7_independent_validator", False, "scripts/validate_r7_confirmatory_results.py missing")
        return
    proc = subprocess.run([sys.executable, str(script), "--out", str(_TMP_REPORT)],
                          cwd=REPO, capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:] or [""]
    record("r7_independent_validator", proc.returncode == 0,
           f"exit={proc.returncode}; {tail[0]}")
    if _TMP_REPORT.exists():
        try:
            payload = json.loads(_TMP_REPORT.read_text(encoding="utf-8"))
            verdict = payload.get("GATE_E") or payload.get("gate_e") or payload.get("status")
            record("r7_gate_e_refresh", str(verdict).upper().find("PASS") >= 0,
                   f"freshly recomputed verdict: {verdict}")
        except Exception as exc:  # noqa: BLE001
            record("r7_gate_e_refresh", False, f"unreadable: {exc}")
        finally:
            _TMP_REPORT.unlink(missing_ok=True)

    # and the committed, hash-locked gate record must itself say PASS
    gate = R7 / "confirmatory" / "VALIDITY_GATE_E.json"
    if gate.exists():
        try:
            payload = json.loads(gate.read_text(encoding="utf-8"))
            verdict = payload.get("GATE_E") or payload.get("gate_e") or payload.get("status")
            record("r7_gate_e_frozen_artifact", str(verdict).upper().find("PASS") >= 0,
                   f"committed frozen verdict: {verdict}")
        except Exception as exc:  # noqa: BLE001
            record("r7_gate_e_frozen_artifact", False, f"unreadable: {exc}")


def check_reported_metrics() -> None:
    """Cross-check the published headline numbers against the shipped result table.

    FINAL_EXPERIMENT_REPORT.md section 7 publishes, bridge-balanced:
        UOT_KR         = 0.429210
        HUNGARIAN_1TO1 = 0.403679
        SUPPORT_PLUS_K = 0.417081
    These are recomputed here from the raw per-bridge table, independently of the
    aggregate table, so a corrupted aggregate cannot pass silently.
    """
    csv = R7 / "analysis" / "confirmatory_bridge_summary.csv"
    if not csv.exists():
        record("r7_headline_metrics", False, "analysis/confirmatory_bridge_summary.csv missing")
        return
    try:
        import csv as _csv
        per_bridge: dict[str, list[float]] = {}
        with csv.open(encoding="utf-8", newline="") as fh:
            for row in _csv.DictReader(fh):
                per_bridge.setdefault(row["method"], []).append(float(row["macro_edge_f1_mean"]))
        balanced = {m: sum(v) / len(v) for m, v in per_bridge.items()}
    except Exception as exc:  # noqa: BLE001
        record("r7_headline_metrics", False, f"could not parse {csv.name}: {exc}")
        return

    published = {"UOT_KR": 0.429210, "HUNGARIAN_1TO1": 0.403679, "SUPPORT_PLUS_K": 0.417081}
    worst = 0.0
    for method, want in published.items():
        got = balanced.get(method)
        if got is None:
            record("r7_headline_metrics", False, f"{method} absent from {csv.name}")
            return
        worst = max(worst, abs(got - want))
    # Tolerance 5e-7: the published values are the same numbers rounded to 6 decimals.
    record("r7_headline_metrics", worst <= 5e-7,
           f"3 published bridge-balanced macro edge F1 values reproduced, max |diff| = {worst:.3e}")


def report_not_offline() -> None:
    print()
    print("NOT CHECKABLE OFFLINE (reported, not failed):")
    for line in [
        "  r5/r6 development cells  -> needs out/multi_bridge_expansion/cost_transport_diagnosis/plans/dev/",
        "                             (≈133 MB of cost/transport .npz, excluded; see EXTERNAL_DATA_MANIFEST.md)",
        "  r7 one-shot re-execution -> permanently refused by design (the holdout ledger exists)",
        "  raw on-chain corpora     -> never released (address-level labels withheld; see README 'Data availability')",
    ]:
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None, help="write a machine-readable report")
    args = ap.parse_args()

    print(f"release_check.py  repo={REPO}")
    print("-" * 78)
    check_layout()
    check_manuscript_hash()
    check_locked_protocol()
    check_confirmatory_present()
    check_manifest_frozen_files()
    before = _snapshot_manifest_files()
    check_validator()
    after = _snapshot_manifest_files()
    changed = sorted(k for k in before if before.get(k) != after.get(k))
    record("release_tree_unmodified", not changed,
           "no manifest-listed file changed"
           if not changed else f"MODIFIED BY VERIFICATION: {changed}")
    check_reported_metrics()
    report_not_offline()

    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    print("-" * 78)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} offline checks passed")

    if args.json:
        args.json.write_text(json.dumps(
            {"repo": str(REPO), "checks": RESULTS,
             "offline_failures": len(failed)}, indent=2), encoding="utf-8")
        print(f"report written to {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
