#!/usr/bin/env python
"""Phase 2: GT traceability verification for flow aggregation.

Validates that every mapped anchor pair can be traced through its entity assignments.
Output: gt_traceability_check.json
"""
from __future__ import annotations

import json, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(r"<REPO>")
OUT_DIR = REPO / "out" / "bsc_open_independent_v1" / "stage5_6_flow_aggregation"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def verify():
    mode_csv = OUT_DIR / "mode_assignments.csv"
    src_json = OUT_DIR / "entity_to_tx_mapping_src.json"
    dst_json = OUT_DIR / "entity_to_tx_mapping_dst.json"

    errors = []
    warnings = []
    checks = {}

    # --- Check 1: Files exist ---
    for f in [mode_csv, src_json, dst_json]:
        if not f.exists():
            errors.append(f"Missing file: {f}")
    if errors:
        result = {"status": "FAIL", "errors": errors, "generated_at": _utc()}
        with open(OUT_DIR / "gt_traceability_check.json", "w") as f:
            json.dump(result, f, indent=2)
        print("FAIL: missing files")
        return result

    # --- Check 2: Load data ---
    mode = pd.read_csv(mode_csv)
    with open(src_json) as f:
        src_map = json.load(f)
    with open(dst_json) as f:
        dst_map = json.load(f)

    checks["n_mode_rows"] = int(len(mode))
    checks["n_src_entities"] = len(src_map)
    checks["n_dst_entities"] = len(dst_map)

    # Build reverse lookups
    src_tx_to_entity = {}
    for eid, txs in src_map.items():
        for txh in txs:
            src_tx_to_entity[txh] = eid

    dst_tx_to_entity = {}
    for eid, txs in dst_map.items():
        for txh in txs:
            dst_tx_to_entity[txh] = eid

    # --- Check 3: Verify every mode row ---
    mismatches = 0
    mode_errors_list = []
    for idx, row in mode.iterrows():
        stx = row["src_tx_hash"]
        dtx = row["dst_tx_hash"]
        se = row["src_entity_id"]
        de = row["dst_entity_id"]
        mode_label = row["mode"]

        issues = []

        # Verify src entity membership
        actual_se = src_tx_to_entity.get(stx)
        if actual_se != se:
            issues.append(f"src_entity mismatch: expected {se}, found {actual_se}")

        # Verify dst entity membership
        actual_de = dst_tx_to_entity.get(dtx)
        if actual_de != de:
            issues.append(f"dst_entity mismatch: expected {de}, found {actual_de}")

        # Verify mode consistency
        if se in src_map and de in dst_map:
            actual_s_size = len(src_map[se])
            actual_d_size = len(dst_map[de])
            expected_s = row["src_size"]
            expected_d = row["dst_size"]
            if actual_s_size != expected_s:
                issues.append(f"src_size mismatch: mode says {expected_s}, entity has {actual_s_size}")
            if actual_d_size != expected_d:
                issues.append(f"dst_size mismatch: mode says {expected_d}, entity has {actual_d_size}")

            # Verify mode label
            if actual_s_size == 1 and actual_d_size == 1:
                expected_mode = "1:1"
            elif actual_s_size >= 2 and actual_d_size == 1:
                expected_mode = "M:1"
            elif actual_s_size == 1 and actual_d_size >= 2:
                expected_mode = "1:N"
            else:
                expected_mode = "M:N"

            if mode_label != expected_mode:
                issues.append(f"mode label mismatch: row says {mode_label}, computed {expected_mode}")

        if issues:
            mismatches += 1
            mode_errors_list.append({
                "row_index": int(idx),
                "src_tx_hash": stx,
                "dst_tx_hash": dtx,
                "issues": issues,
            })

    checks["mode_rows_verified"] = len(mode)
    checks["mode_rows_with_issues"] = mismatches
    if mismatches > 0:
        errors.append(f"{mismatches} mode rows have issues")

    # --- Check 4: Entity ID format ---
    format_issues = 0
    for eid in list(src_map.keys())[:10] + list(dst_map.keys())[:10]:
        if not (eid.startswith("S_") or eid.startswith("R_")):
            format_issues += 1
    checks["entity_id_format_ok"] = format_issues == 0
    if format_issues > 0:
        errors.append(f"{format_issues} entities have invalid ID format")

    # --- Check 5: No orphan entity links in mode rows ---
    orphan_src = sum(1 for _, r in mode.iterrows() if r["src_entity_id"] not in src_map)
    orphan_dst = sum(1 for _, r in mode.iterrows() if r["dst_entity_id"] not in dst_map)
    checks["orphan_src_entities_in_mode"] = int(orphan_src)
    checks["orphan_dst_entities_in_mode"] = int(orphan_dst)
    if orphan_src > 0:
        errors.append(f"{orphan_src} mode rows reference non-existent src entities")
    if orphan_dst > 0:
        errors.append(f"{orphan_dst} mode rows reference non-existent dst entities")

    # --- Check 6: Entity size distribution ---
    src_sizes = [len(txs) for txs in src_map.values()]
    dst_sizes = [len(txs) for txs in dst_map.values()]
    checks["src_entity_size_stats"] = {
        "min": min(src_sizes) if src_sizes else 0,
        "max": max(src_sizes) if src_sizes else 0,
        "mean": round(sum(src_sizes) / len(src_sizes), 2) if src_sizes else 0,
        "median": sorted(src_sizes)[len(src_sizes) // 2] if src_sizes else 0,
        "gt_1_tx": sum(1 for s in src_sizes if s > 1),
        "eq_1_tx": sum(1 for s in src_sizes if s == 1),
    }
    checks["dst_entity_size_stats"] = {
        "min": min(dst_sizes) if dst_sizes else 0,
        "max": max(dst_sizes) if dst_sizes else 0,
        "mean": round(sum(dst_sizes) / len(dst_sizes), 2) if dst_sizes else 0,
        "median": sorted(dst_sizes)[len(dst_sizes) // 2] if dst_sizes else 0,
        "gt_1_tx": sum(1 for s in dst_sizes if s > 1),
        "eq_1_tx": sum(1 for s in dst_sizes if s == 1),
    }

    # --- Check 7: All mapped src_tx belong to exactly one entity ---
    src_tx_entity_counts = {}
    for eid, txs in src_map.items():
        for txh in txs:
            src_tx_entity_counts[txh] = src_tx_entity_counts.get(txh, 0) + 1
    multi_src = sum(1 for c in src_tx_entity_counts.values() if c > 1)
    checks["src_tx_in_multiple_entities"] = multi_src
    if multi_src > 0:
        errors.append(f"{multi_src} src txs assigned to multiple entities")

    dst_tx_entity_counts = {}
    for eid, txs in dst_map.items():
        for txh in txs:
            dst_tx_entity_counts[txh] = dst_tx_entity_counts.get(txh, 0) + 1
    multi_dst = sum(1 for c in dst_tx_entity_counts.values() if c > 1)
    checks["dst_tx_in_multiple_entities"] = multi_dst
    if multi_dst > 0:
        errors.append(f"{multi_dst} dst txs assigned to multiple entities")

    # --- Verdict ---
    passed = len(errors) == 0
    result = {
        "status": "PASS" if passed else "FAIL",
        "generated_at": _utc(),
        "checks": checks,
        "errors": errors if errors else [],
        "warnings": warnings,
        "sample_errors": mode_errors_list[:5] if mode_errors_list else [],
    }

    with open(OUT_DIR / "gt_traceability_check.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'PASS' if passed else 'FAIL'}: {len(errors)} errors, {len(warnings)} warnings")
    for k, v in checks.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    return result


if __name__ == "__main__":
    verify()
