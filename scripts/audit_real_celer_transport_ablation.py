#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit Real Celer Transport Solver Ablation results for consistency.

Checks that all runs have complete output, shared input_hash,
identical cost matrix config, consistent candidate pool, valid time filter,
and that summary.csv matches metrics.json.
Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit(run_root: Path) -> int:
    runs_dir = run_root / "runs"
    if not runs_dir.is_dir():
        print(f"FAIL: runs directory not found: {runs_dir}")
        return 1

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        print("FAIL: No run directories found")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    input_hashes: set[str] = set()
    cost_config_sigs: set[str] = set()
    time_filters: set[bool] = set()
    candidate_pool_hashes: set[str] = set()
    all_metrics: list[dict[str, Any]] = []

    required_files = [
        "config.json",
        "meta.json",
        "metrics.json",
        "solver_meta.json",
        "transport_plan.csv",
        "topk_candidates.csv",
    ]

    # 1. Per-run checks
    for run_dir in run_dirs:
        rid = run_dir.name
        for fname in required_files:
            if not (run_dir / fname).is_file():
                errors.append(f"Missing {fname} in {rid}")

        mp = run_dir / "metrics.json"
        if mp.is_file():
            m = _load_json(mp)
            all_metrics.append(m)
            ih = m.get("input_hash", "")
            if ih:
                input_hashes.add(ih)

        meta_p = run_dir / "meta.json"
        if meta_p.is_file():
            meta = _load_json(meta_p)
            cc = meta.get("cost_context", {})
            apply_tf = meta.get("apply_joint_time_filter", None)
            if apply_tf is not None:
                time_filters.add(bool(apply_tf))

            # Build cost_config signature
            cost_sig_parts = [
                json.dumps(cc.get("cost_weights", {}), sort_keys=True),
                str(cc.get("time_delay_policy", "")),
                str(cc.get("max_delay_sec", "")),
                str(cc.get("causal_violation_penalty", "")),
                str(cc.get("candidate_pool_source", "")),
            ]
            cost_config_sigs.add("|".join(cost_sig_parts))

            # Candidate pool consistency
            n_cand = cc.get("n_candidate_edges")
            cp_source = cc.get("candidate_pool_source", "unknown")
            if n_cand is not None and cp_source is not None:
                candidate_pool_hashes.add(f"{cp_source}:{n_cand}")

            # Check apply_joint_time_filter
            if apply_tf is not True:
                errors.append(f"apply_joint_time_filter is not True in {rid}: {apply_tf}")

    # 2. Input hash consistency
    if len(input_hashes) == 1 and "" not in input_hashes:
        print(f"[OK] All runs share input_hash: {next(iter(input_hashes))}")
    elif len(input_hashes) == 1 and "" in input_hashes:
        # All empty: paper-aligned, acceptable but note it
        print("[OK] input_hash: all paper-aligned (no hash tracking)")
    elif len(input_hashes) == 0:
        errors.append("input_hash: no hashes found in any run (provenance missing)")
    else:
        errors.append(f"input_hash not identical: found {len(input_hashes)} distinct hashes: {sorted(input_hashes)}")

    # 3. Cost config consistency
    if len(cost_config_sigs) != 1:
        errors.append(f"cost_config not identical: {len(cost_config_sigs)} variants")
    else:
        print("[OK] All runs share identical cost config")

    # 4. Time filter consistency
    time_filter_consistent = len(time_filters) == 1 and True in time_filters
    if not time_filter_consistent:
        errors.append(f"apply_joint_time_filter not consistently True: {time_filters}")
    else:
        print("[OK] All runs have apply_joint_time_filter=True")

    # 5. Candidate pool consistency
    if len(candidate_pool_hashes) != 1:
        errors.append(f"candidate pool not identical: {candidate_pool_hashes}")
    else:
        print(f"[OK] All runs share candidate pool: {next(iter(candidate_pool_hashes))}")

    # 6. Required solvers
    solver_names = set()
    for m in all_metrics:
        solver_names.add(m.get("solver", ""))
    for s in ["rc_uot_full", "cost_ranking"]:
        if s not in solver_names:
            errors.append(f"Required solver '{s}' not found")
        else:
            print(f"[OK] Required solver '{s}' present")

    # 7. Per-metric checks for non-skipped runs
    for m in all_metrics:
        rid = m.get("run_id", "?")
        solver = m.get("solver", "?")
        if m.get("skipped"):
            continue

        # Before/after filter metrics must exist
        for key in ["pair_precision_before_filter", "pair_precision_after_filter",
                     "pair_recall_before_filter", "pair_recall_after_filter"]:
            if m.get(key) is None:
                errors.append(f"{rid}: {key} is None for non-skipped run")

        # Causal violation after filter must be 0
        cvr_after = m.get("causal_violation_rate_after_filter")
        if cvr_after is not None and cvr_after > 1e-12:
            errors.append(f"{rid}: causal_violation_rate_after_filter = {cvr_after} > 0")

        # Filter diagnostics: no negative flow_* after filter
        fd = m.get("filter_diagnostics", {})
        if fd:
            fi_neg_after = fd.get("n_flow_i_negative_after_filter", 0)
            fj_neg_after = fd.get("n_flow_j_negative_after_filter", 0)
            if fi_neg_after is not None and fi_neg_after > 0:
                errors.append(f"{rid}: n_flow_i_negative_after_filter = {fi_neg_after} > 0")
            if fj_neg_after is not None and fj_neg_after > 0:
                errors.append(f"{rid}: n_flow_j_negative_after_filter = {fj_neg_after} > 0")
        else:
            errors.append(f"{rid}: missing filter_diagnostics in metrics")

        # Candidate pool must not be dense_debug (unless marked)
        cp_src = m.get("candidate_pool_source", "")
        if cp_src == "dense_debug":
            errors.append(f"{rid}: candidate_pool_source is dense_debug (not paper-aligned)")

    # 7.5. Cross-source hash consistency (metrics vs meta vs provenance)
    for run_dir in run_dirs:
        rid = run_dir.name
        mp = run_dir / "metrics.json"
        meta_p = run_dir / "meta.json"
        if mp.is_file() and meta_p.is_file():
            m = _load_json(mp)
            meta = _load_json(meta_p)
            # Check input_hash consistency
            m_ih = m.get("input_hash", "")
            meta_ih = meta.get("input_hash", "")
            if m_ih and meta_ih and m_ih != meta_ih:
                errors.append(f"{rid}: input_hash mismatch: metrics={m_ih} vs meta={meta_ih}")
            # Check config_hash consistency
            m_ch = m.get("config_hash", "")
            meta_ch = meta.get("config_hash", "")
            if m_ch and meta_ch and m_ch != meta_ch:
                errors.append(f"{rid}: config_hash mismatch: metrics={m_ch} vs meta={meta_ch}")
            # If metrics has empty hash but meta has non-empty, that is a failure
            if (not m_ih or m_ih == "") and meta_ih and meta_ih != "":
                errors.append(f"{rid}: input_hash empty in metrics but non-empty in meta ({meta_ih})")
            if (not m_ch or m_ch == "") and meta_ch and meta_ch != "":
                errors.append(f"{rid}: config_hash empty in metrics but non-empty in meta ({meta_ch})")

    print("[OK] Hash consistency checked")

    # 8. Summary vs metrics consistency
    summary_path = run_root / "summary.csv"
    summary_ok = True
    if summary_path.is_file():
        summary_df = pd.read_csv(summary_path)
        for _, srow in summary_df.iterrows():
            rid = srow.get("run_id", "")
            matching = [m for m in all_metrics if m.get("run_id") == rid]
            if not matching:
                warnings.append(f"run_id {rid} in summary but no metrics found")
                continue
            m = matching[0]
            for col in ["pair_precision_after_filter", "pair_recall_after_filter", "pair_f1_after_filter"]:
                sv = srow.get(col)
                mv = m.get(col)
                if sv is not None and mv is not None and not pd.isna(sv) and not pd.isna(mv):
                    if abs(float(sv) - float(mv)) > 0.001:
                        errors.append(f"Mismatch {col} for {rid}: summary={sv:.6f} vs metrics={mv:.6f}")
                        summary_ok = False
        if summary_ok and not any("Mismatch" in e for e in errors):
            print("[OK] summary.csv consistent with metrics.json")
    else:
        warnings.append("summary.csv not found")

    # 8.5. Paper-aligned parity checks
    parity_path = run_root / "table5_parity_report.json"
    if parity_path.is_file():
        parity = _load_json(parity_path)
        parity_pass = parity.get("parity_pass", False)

        if not parity_pass:
            # Only fail if rc_uot_full specifically failed parity (other solvers may show false for non-parity reasons)
            rc_metrics = [m for m in all_metrics if m.get("solver") == "rc_uot_full" and not m.get("skipped")]
            if rc_metrics and not rc_metrics[0].get("table5_parity_pass", True):
                errors.append(
                    f"Table 5 parity not passed for rc_uot_full: {parity.get('failure_reasons', [])}"
                )
            else:
                warnings.append(
                    f"table5_parity_report.json parity_pass=False but rc_uot_full metrics OK"
                )
        else:
            print("[OK] Table 5 parity passed")

        # Check all non-skipped solvers have table5_parity_pass
        for m in all_metrics:
            rid = m.get("run_id", "?")
            if m.get("skipped"):
                continue
            t5_pass = m.get("table5_parity_pass")
            if t5_pass is False:
                errors.append(f"{rid}: table5_parity_pass is False")
            elif t5_pass is True:
                print(f"[OK] {rid}: table5_parity_pass=True")

        # Check provenance hashes consistency across solvers
        provenance_hashes: dict[str, set[str]] = {}
        for m in all_metrics:
            rid = m.get("run_id", "?")
            if m.get("skipped"):
                continue
            for key in ("candidate_pool_mode",):
                val = m.get(key)
                if val is not None:
                    provenance_hashes.setdefault(key, set()).add(str(val))

        for key, vals in provenance_hashes.items():
            if len(vals) != 1:
                errors.append(f"{key} not identical across solvers: {vals}")
            else:
                print(f"[OK] All solvers share {key}: {next(iter(vals))}")

        # Check per_asset_group mode
        cp_mode = next(iter(provenance_hashes.get("candidate_pool_mode", set())), "")
        if cp_mode and cp_mode != "per_asset_group" and cp_mode != "per_tx_pair":
            errors.append(f"candidate_pool_mode is '{cp_mode}', expected 'per_asset_group' or 'per_tx_pair'")
    else:
        errors.append("table5_parity_report.json not found")

    # 9. Generate report
    missing_files_list = []
    for rd in run_dirs:
        for fname in required_files:
            if not (rd / fname).is_file():
                missing_files_list.append(f"{rd.name}/{fname}")

    report = {
        "status": "pass" if len(errors) == 0 else "fail",
        "checked_runs": len(run_dirs),
        "missing_files": missing_files_list,
        "input_hash_consistent": len(input_hashes) == 1,
        "cost_config_consistent": len(cost_config_sigs) == 1,
        "time_filter_consistent": time_filter_consistent,
        "candidate_pool_consistent": len(candidate_pool_hashes) == 1,
        "has_rc_uot_full": "rc_uot_full" in solver_names,
        "has_cost_ranking": "cost_ranking" in solver_names,
        "summary_matches_metrics": summary_ok,
        "table5_parity_exists": parity_path.is_file() if parity_path else False,
        "table5_parity_pass": parity.get("parity_pass") if (parity_path and parity_path.is_file()) else False,
        "failures": errors,
        "warnings": warnings,
        "n_runs": len(run_dirs),
        "solver_names": sorted(solver_names),
    }

    report_path = run_root / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudit report: {report_path}")

    if errors:
        print(f"\nFAIL: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    print("\nOK: All checks passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit Real Celer Transport Solver Ablation")
    p.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "real_celer_transport_ablation",
        help="Run root directory",
    )
    args = p.parse_args(argv)
    return audit(Path(args.run_root))


if __name__ == "__main__":
    raise SystemExit(main())
