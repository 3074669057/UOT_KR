#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package v3 + v3b ablation results for paper/archive (read-only copy + index)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
V3_SRC = REPO / "out/baseline_compare/bridge_semantic_ablation_v3"
V3B_SRC = REPO / "out/baseline_compare/bridge_semantic_ablation_v3b_high_f1"
PKG = REPO / "out/baseline_compare/bridge_semantic_ablation_v3_v3b_package"
ZIP_OUT = REPO / "out/baseline_compare/bridge_semantic_ablation_v3_v3b_package.zip"

V3_FILES = [
    "implementation_audit.md",
    "dry_run_probe_report.json",
    "dry_run_probe_report.md",
    "degradation_curve_v3.json",
    "degradation_curve_v3.md",
    "acceptance_report.json",
    "claim_support.md",
    "manifest.json",
    "production_base_integrity_audit.json",
    "connector_status_frozen.json",
    "full_run_report.json",
    "phase1_prediction_equality_audit.json",
    "id_anchor_prediction_equality_audit.json",
    "routeA_v2_equivalence_audit.json",
    "rc_uot_q_v2_equivalence_audit.json",
    "paper_rq5_v3_main_table.md",
    "paper_rq5_v3b_tuned_table.md",
    "paper_rq5_paragraph_en.md",
    "paper_rq5_paragraph_cn.md",
    "paper_rq5_section_4_7_final_en.md",
    "paper_rq5_section_4_7_final_cn.md",
]

V3B_FILES = [
    "acceptance_report.json",
    "manifest.json",
    "splits/dev_pairs.csv",
    "splits/test_pairs.csv",
    "splits/split_manifest.json",
    "summary/high_f1_test_table.md",
    "summary/high_f1_test_table.csv",
    "summary/high_f1_dev_search.md",
    "summary/subset_scope_table.md",
    "summary/no_all_diagnostic.md",
    "summary/paper_safe_claims.md",
    "summary/manifest.json",
]

V3B_DIRS = ["dev_search", "dev_tune", "test_eval", "summary", "splits", "subset_analysis", "transport_cache", "integrity"]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _fmt_f1(v: Any) -> str:
    if v is None or v == "" or v == "N/A":
        return "N/A"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _connector_pair_f1_display(status: str | None, f1: Any) -> str:
    if status == "BLOCKED":
        return "N/A"
    if f1 is None:
        return "N/A"
    return _fmt_f1(f1)


def _connector_operational_score(status: str | None, f1: Any) -> str:
    if status == "BLOCKED":
        return "0.0000"
    if status == "ZERO_PREDICTIONS":
        return "0.0000"
    if f1 is None:
        return "N/A"
    return _fmt_f1(f1)


def _copy_file(src: Path, dst: Path, missing: list[dict[str, str]]) -> bool:
    if not src.is_file():
        missing.append({"expected": str(src.relative_to(REPO)), "destination": str(dst.relative_to(REPO))})
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_tree(src: Path, dst: Path, missing: list[dict[str, str]]) -> bool:
    if not src.is_dir():
        missing.append({"expected_dir": str(src.relative_to(REPO)), "destination": str(dst.relative_to(REPO))})
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return True


def _build_unified_rows(v3_rows: dict[str, dict], v3b_rows: dict[str, dict]) -> list[dict[str, Any]]:
    all_masks = sorted(set(v3_rows) | set(v3b_rows))
    unified: list[dict[str, Any]] = []
    for mask_id in all_masks:
        v3 = v3_rows.get(mask_id, {})
        v3b = v3b_rows.get(mask_id, {})
        notes: list[str] = []

        conn_status = v3.get("connector_status")
        conn_f1_raw = v3.get("connector_pair_f1")
        conn_f1 = _connector_pair_f1_display(conn_status, conn_f1_raw)
        conn_op = _connector_operational_score(conn_status, conn_f1_raw)

        if mask_id in v3 and mask_id not in v3b:
            notes.append("v3-only mask (not in v3b eval set)")
        if mask_id in v3b and mask_id not in v3:
            notes.append("v3b-only mask (chain-observables diagnostic or supplementary)")
        if mask_id in v3 and mask_id in v3b:
            notes.append("v3 metrics are full-set; v3b metrics are held-out test split (5075 pairs)")
        if conn_status == "BLOCKED":
            notes.append("Connector BLOCKED: pair-F1 N/A; operational_score=0 is deployment inapplicability, not pair-F1")
        if mask_id == "no_all_bridge_semantics":
            notes.append("strict all-bridge-semantics missing (B ∪ I)")
        if mask_id == "no_bridge_metadata_chain_observables_retained":
            notes.append("NOT strict no_all: bridge metadata masked; chain observables retained")

        v3_frozen = v3.get("rc_uot_q_pair_f1")
        v3b_gb = v3b.get("rc_uot_q_global_best_f1")
        delta = ""
        if v3_frozen is not None and v3b_gb not in (None, ""):
            try:
                delta = f"{float(v3b_gb) - float(v3_frozen):+.4f}"
                notes.append("delta compares v3b test global_best vs v3 full-set frozen (different denominators)")
            except (TypeError, ValueError):
                delta = "N/A"

        fields = v3.get("fields_masked")
        if fields is None and mask_id == "no_bridge_metadata_chain_observables_retained":
            fields = ["bridge_metadata_masked_chain_observables_retained"]

        unified.append(
            {
                "mask_id": mask_id,
                "ablation_type": v3.get("ablation_type") or ("v3b_diagnostic" if mask_id not in v3 else ""),
                "fields_masked": ";".join(fields) if isinstance(fields, list) else (fields or ""),
                "connector_status_v3": conn_status or (v3b.get("connector_status") if not v3 else ""),
                "connector_pair_f1_v3": conn_f1 if v3 else "N/A",
                "connector_operational_score_v3": conn_op if v3 else "N/A",
                "rc_uot_q_frozen_f1_v3": _fmt_f1(v3_frozen) if v3_frozen is not None else "N/A",
                "rc_uot_q_frozen_precision_v3": _fmt_f1(v3.get("rc_uot_q_pair_precision")) if v3 else "N/A",
                "rc_uot_q_frozen_recall_v3": _fmt_f1(v3.get("rc_uot_q_pair_recall")) if v3 else "N/A",
                "rc_uot_q_frozen_coverage_v3": _fmt_f1(v3.get("rc_uot_q_coverage")) if v3 else "N/A",
                "rc_uot_q_frozen_tx_cvr_v3": _fmt_f1(v3.get("rc_uot_q_tx_level_cvr")) if v3 else "N/A",
                "rc_uot_q_v3b_global_best_f1": _fmt_f1(v3b_gb) if v3b_gb not in (None, "") else "N/A",
                "rc_uot_q_v3b_global_best_precision": _fmt_f1(v3b.get("rc_uot_q_global_best_precision")) if v3b else "N/A",
                "rc_uot_q_v3b_global_best_recall": _fmt_f1(v3b.get("rc_uot_q_global_best_recall")) if v3b else "N/A",
                "rc_uot_q_v3b_global_best_coverage": _fmt_f1(v3b.get("rc_uot_q_global_best_coverage")) if v3b else "N/A",
                "rc_uot_q_v3b_global_best_tx_cvr": _fmt_f1(v3b.get("rc_uot_q_global_best_tx_CVR")) if v3b else "N/A",
                "delta_v3b_global_best_vs_v3_frozen": delta or "N/A",
                "notes": " | ".join(notes),
            }
        )
    return unified


def main() -> None:
    missing: list[dict[str, str]] = []

    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    v3_dst = PKG / "v3_frozen_main"
    v3b_dst = PKG / "v3b_high_f1_supplement"
    v3_dst.mkdir(parents=True)
    v3b_dst.mkdir(parents=True)

    for rel in V3_FILES:
        _copy_file(V3_SRC / rel, v3_dst / rel, missing)
    _copy_tree(V3_SRC / "masks", v3_dst / "masks", missing)

    for rel in V3B_FILES:
        src = V3B_SRC / rel
        if rel == "manifest.json" and not src.is_file():
            alt = V3B_SRC / "summary/manifest.json"
            if alt.is_file():
                _copy_file(alt, v3b_dst / "summary/manifest.json", missing)
                missing.append(
                    {
                        "expected": str((V3B_SRC / "manifest.json").relative_to(REPO)),
                        "note": "root manifest.json missing; copied summary/manifest.json instead",
                    }
                )
            else:
                missing.append({"expected": str(src.relative_to(REPO))})
            continue
        _copy_file(src, v3b_dst / rel, missing)

    for dname in V3B_DIRS:
        src = V3B_SRC / dname
        if src.is_dir():
            _copy_tree(src, v3b_dst / dname, missing)
        elif dname == "dev_search":
            missing.append({"expected_dir": str(src.relative_to(REPO)), "note": "dev_search/ not present; dev_tune/ copied if exists"})

    # Also copy v3b acceptance and summary extras at supplement root
    for extra in ["acceptance_report.json"]:
        _copy_file(V3B_SRC / extra, v3b_dst / extra, missing)

    # Unified table
    curve = json.loads((V3_SRC / "degradation_curve_v3.json").read_text(encoding="utf-8"))
    v3_rows = {r["mask_id"]: r for r in curve["rows"]}
    v3b_csv = V3B_SRC / "summary/high_f1_test_table.csv"
    v3b_rows = {r["mask_id"]: r for r in csv.DictReader(v3b_csv.open(encoding="utf-8"))}

    unified = _build_unified_rows(v3_rows, v3b_rows)
    cols = list(unified[0].keys()) if unified else []
    ut_dir = PKG / "unified_tables"
    ut_dir.mkdir(parents=True)
    csv_path = ut_dir / "unified_v3_v3b_mask_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(unified)

    md_lines = [
        "# Unified v3 + v3b Mask Table",
        "",
        f"Generated: {_utc()}",
        "",
        "**Lineage:** v3 = frozen main (full-set); v3b = supplementary global_best (held-out test split).",
        "",
        "**Connector operational_score:** BLOCKED → 0 (deployment inapplicability, **not** pair-F1).",
        "",
        "| " + " | ".join(cols[:8]) + " |",
        "|" + "|".join(["---"] * min(8, len(cols))) + "|",
    ]
    for r in unified[:5]:
        md_lines.append("| " + " | ".join(str(r[c])[:24] for c in cols[:8]) + " |")
    md_lines += ["", f"*Full table: {len(unified)} rows — see CSV for all columns.*", ""]
    _write_text(ut_dir / "unified_v3_v3b_mask_table.md", "\n".join(md_lines) + "\n")

    # Stats for summary
    v3b_frozen_vals = [float(r["rc_uot_q_frozen_f1"]) for r in v3b_rows.values() if r.get("rc_uot_q_frozen_f1")]
    v3b_gb_vals = [float(r["rc_uot_q_global_best_f1"]) for r in v3b_rows.values() if r.get("rc_uot_q_global_best_f1")]
    avg_frozen = sum(v3b_frozen_vals) / len(v3b_frozen_vals)
    avg_gb = sum(v3b_gb_vals) / len(v3b_gb_vals)
    top5 = sorted(v3b_rows.values(), key=lambda r: float(r.get("delta_vs_frozen") or 0), reverse=True)[:5]

    strict_gb = float(v3b_rows["no_all_bridge_semantics"]["rc_uot_q_global_best_f1"])
    chain_gb = float(v3b_rows["no_bridge_metadata_chain_observables_retained"]["rc_uot_q_global_best_f1"])

    paper = f"""# Paper-Ready Summary — Bridge Semantic Ablation v3 + v3b

Generated: {_utc()}

## 1. Experiment lineage

- **v3** = frozen main degradation/applicability experiment (full evaluation set; Route A v2 + Phase 1 canonical + frozen RC-UOT-Q fixed-delay).
- **v3b** = supplementary development-selected tuned operating point (global_best only; does **not** replace v3).

## 2. Key frozen v3 result

| Setting | Connector | RC-UOT-Q (frozen) |
|---------|----------:|------------------:|
| full_native | F1 = **0.9736** | F1 = **0.7085** |

Under missing bridge semantics, Connector exhibits **BLOCKED** or **ZERO_PREDICTIONS** (receiver, amount, asset_s, dstChain, timestamp and combinations); RC-UOT-Q remains **ACCEPTED** and evaluable under the v3 frozen protocol.

## 3. Key v3    b result

- **dev / test split:** 2,221 / 5,075 pairs (30% / 70%, hash-seeded).
- **v3b frozen average test F1:** {avg_frozen:.3f}
- **v3b global_best average test F1:** {avg_gb:.3f}
- **Average improvement (global_best − frozen, test split):** {avg_gb - avg_frozen:+.3f}

**Top-5 test-split gains (global_best vs frozen):**

"""
    for r in top5:
        paper += f"- `{r['mask_id']}`: {float(r['rc_uot_q_frozen_f1']):.4f} → {float(r['rc_uot_q_global_best_f1']):.4f} (Δ {float(r['delta_vs_frozen']):+.4f})\n"

    paper += f"""
## 4. no_all diagnostic

**strict `no_all_bridge_semantics`:**
- Connector: **BLOCKED** (pair-F1 N/A)
- RC-UOT-Q global_best test F1 ≈ **{strict_gb:.3f}** (limited recovery)

**`no_bridge_metadata_chain_observables_retained`:**
- RC-UOT-Q global_best test F1 ≈ **{chain_gb:.3f}**
- **chain-observables-retained is NOT strict all-bridge-semantics missing** (metadata suppressed; on-chain observables retained).

## 5. Safe claims

1. Under full native bridge semantics, Connector is the stronger raw top-1 upper bound (v3 frozen).
2. Under missing bridge semantics, Connector may be BLOCKED or emit zero predictions; RC-UOT-Q remains operational.
3. RC-UOT-Q degradation under increasing semantic loss is graded; Connector failure modes are binary.
4. v3b global_best tuning on dev can improve held-out test F1 on several amount-heavy masks without replacing v3 main results.
5. Strict all-bridge-missing limits recovery; chain-observables-retained preserves much higher F1 (~0.706).

## 6. Forbidden claims

- RC-UOT-Q universally outperforms Connector
- RC-UOT-Q dominates Connector under full native bridge semantics
- BLOCKED means pair-F1 = 0
- strict all-bridge-semantics missing achieves high F1
- v3b replaces v3 frozen main result
"""
    _write_text(PKG / "paper_ready_summary.md", paper)

    # File index
    index_rows: list[dict[str, Any]] = []

    def _classify(rel: str) -> tuple[str, str]:
        if rel.startswith("v3_frozen_main/"):
            exp = "v3_frozen_main"
        elif rel.startswith("v3b_high_f1_supplement/"):
            exp = "v3b_high_f1_supplement"
        else:
            exp = "unified_package"
        ext = Path(rel).suffix.lower()
        if ext == ".json":
            ft = "json"
        elif ext == ".md":
            ft = "markdown"
        elif ext == ".csv":
            ft = "csv"
        elif ext == ".zip":
            ft = "zip"
        else:
            ft = "other"
        return exp, ft

    for path in sorted(PKG.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PKG).as_posix()
        exp, ft = _classify(rel)
        index_rows.append(
            {
                "relative_path": rel,
                "source_experiment": exp,
                "file_type": ft,
                "description": rel.split("/")[-1],
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    idx_csv = PKG / "file_index.csv"
    idx_cols = ["relative_path", "source_experiment", "file_type", "description", "size_bytes", "sha256"]
    with idx_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=idx_cols)
        w.writeheader()
        w.writerows(index_rows)

    idx_md = ["# File Index", "", f"Generated: {_utc()}", "", f"Total files: {len(index_rows)}", ""]
    idx_md.append("| relative_path | source_experiment | file_type | size_bytes |")
    idx_md.append("|---------------|-------------------|-----------|----------:|")
    for r in index_rows[:50]:
        idx_md.append(f"| {r['relative_path']} | {r['source_experiment']} | {r['file_type']} | {r['size_bytes']} |")
    if len(index_rows) > 50:
        idx_md.append(f"| ... | ... | ... | ({len(index_rows) - 50} more in file_index.csv) |")
    _write_text(PKG / "file_index.md", "\n".join(idx_md) + "\n")

    key_results = {
        "v3_connector_full_native_f1": 0.9736,
        "v3_rc_uot_q_full_native_f1": 0.7085,
        "v3b_frozen_avg_test_f1": round(avg_frozen, 3),
        "v3b_global_best_avg_test_f1": round(avg_gb, 3),
        "v3b_avg_delta": round(avg_gb - avg_frozen, 3),
        "strict_no_all_global_best_f1": round(strict_gb, 3),
        "chain_observables_retained_global_best_f1": round(chain_gb, 3),
    }

    manifest = {
        "package_name": "bridge_semantic_ablation_v3_v3b_package",
        "created_at": _utc(),
        "source_dirs": {
            "v3": "out/baseline_compare/bridge_semantic_ablation_v3/",
            "v3b": "out/baseline_compare/bridge_semantic_ablation_v3b_high_f1/",
        },
        "lineage": {
            "v3": "Route A v2 + Phase 1 canonical + frozen RC-UOT-Q fixed-delay",
            "v3b": "supplementary high-F1 development-selected tuned operating point",
        },
        "key_results": key_results,
        "claim_boundary": {
            "v3_is_main": True,
            "v3b_is_supplementary": True,
            "blocked_is_not_pair_f1_zero": True,
            "chain_observables_retained_is_not_strict_no_all": True,
        },
        "files": [
            {"relative_path": r["relative_path"], "sha256": r["sha256"], "size_bytes": r["size_bytes"]}
            for r in index_rows
        ],
    }
    _write_json(PKG / "package_manifest.json", manifest)

    # Integrity checks
    blocked_ok = all(
        r["connector_pair_f1_v3"] == "N/A"
        for r in unified
        if r.get("connector_status_v3") == "BLOCKED" and r.get("connector_pair_f1_v3") not in ("", None)
    )
    separated = (
        "no_all_bridge_semantics" in {r["mask_id"] for r in unified}
        and "no_bridge_metadata_chain_observables_retained" in {r["mask_id"] for r in unified}
    )

    v3_acc = json.loads((V3_SRC / "acceptance_report.json").read_text(encoding="utf-8"))
    v3b_acc = json.loads((V3B_SRC / "acceptance_report.json").read_text(encoding="utf-8"))
    full_native = v3_rows["full_native"]

    checks = [
        ("C1", "v3 degradation_curve_v3.json exists", (V3_SRC / "degradation_curve_v3.json").is_file(), ""),
        ("C2", "v3 acceptance all_pass=true", v3_acc.get("all_pass") is True, str(v3_acc.get("all_pass"))),
        ("C3", "v3b acceptance all_pass=true", v3b_acc.get("all_pass") is True, str(v3b_acc.get("all_pass"))),
        (
            "C4",
            "v3 full_native Connector F1 = 0.9736",
            abs(float(full_native["connector_pair_f1"]) - 0.9736) < 0.0001,
            str(full_native.get("connector_pair_f1")),
        ),
        (
            "C5",
            "v3 full_native RC-UOT-Q F1 = 0.7085",
            abs(float(full_native["rc_uot_q_pair_f1"]) - 0.7085) < 0.0001,
            str(full_native.get("rc_uot_q_pair_f1")),
        ),
        (
            "C6",
            "v3b global_best average F1 = 0.514",
            abs(avg_gb - 0.514) < 0.002,
            f"{avg_gb:.4f}",
        ),
        ("C7", "BLOCKED rows use pair-F1=N/A in unified table", blocked_ok, ""),
        ("C8", "strict no_all and chain-observables separated", separated, ""),
        ("C9", "package_manifest.json exists", (PKG / "package_manifest.json").is_file(), ""),
        ("C10", "zip archive created", False, "pending"),
    ]

    # Zip
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PKG.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(PKG.parent).as_posix())

    zip_sha = _sha256(ZIP_OUT)
    _write_text(PKG / "package_zip_sha256.txt", zip_sha + "\n")
    _write_text(REPO / "out/baseline_compare/bridge_semantic_ablation_v3_v3b_package_zip_sha256.txt", zip_sha + "\n")

    checks[-1] = ("C10", "zip archive created", ZIP_OUT.is_file() and ZIP_OUT.stat().st_size > 0, str(ZIP_OUT))

    integrity_json = {
        "generated_at_utc": _utc(),
        "checks": [{"id": c[0], "name": c[1], "pass": c[2], "detail": c[3]} for c in checks],
        "all_pass": all(c[2] for c in checks),
    }
    _write_json(PKG / "integrity_check.json", integrity_json)

    int_md = ["# Integrity Check", "", f"Generated: {_utc()}", "", "| ID | Check | Pass | Detail |", "|----|-------|:----:|--------|"]
    for c in checks:
        int_md.append(f"| {c[0]} | {c[1]} | {'PASS' if c[2] else 'FAIL'} | {c[3]} |")
    int_md.append(f"\n**Overall:** {'PASS' if integrity_json['all_pass'] else 'FAIL'}")
    _write_text(PKG / "integrity_check.md", "\n".join(int_md) + "\n")

    # Missing files report
    miss_lines = ["# Missing Files Report", "", f"Generated: {_utc()}", ""]
    if missing:
        for m in missing:
            miss_lines.append(f"- {json.dumps(m, ensure_ascii=False)}")
    else:
        miss_lines.append("No missing expected files.")
    _write_text(PKG / "missing_files_report.md", "\n".join(miss_lines) + "\n")

    # Fix typo in paper summary
    text = (PKG / "paper_ready_summary.md").read_text(encoding="utf-8")
    text = text.replace("## 3. Key v    b result", "## 3. Key v3b result")
    (PKG / "paper_ready_summary.md").write_text(text, encoding="utf-8")

    # Rebuild unified md with full markdown table
    md_full = [
        "# Unified v3 + v3b Mask Table",
        "",
        f"Generated: {_utc()}",
        "",
        "**v3** = frozen main (full-set). **v3b** = supplementary global_best (held-out test).",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in unified:
        md_full.append("| " + " | ".join(str(r[c]).replace("|", "\\|")[:40] for c in cols) + " |")
    _write_text(ut_dir / "unified_v3_v3b_mask_table.md", "\n".join(md_full) + "\n")

    print("PACKAGE_DONE")
    print("integrity_all_pass", integrity_json["all_pass"])
    for c in checks:
        print(c[0], "PASS" if c[2] else "FAIL", c[1])


if __name__ == "__main__":
    main()
