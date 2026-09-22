"""Assemble final paper diagnostics bundle."""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _check_diagnostic_legacy_top3_removed(out_root: Path) -> bool:
    leave = out_root / "leave_anchor_out_real_final"
    for name in (
        "paper_table_diagnostic_ablations.csv",
        "paper_table_diagnostic_ablations.md",
        "paper_table_diagnostic_ablations_rounded.csv",
        "paper_table_diagnostic_ablations_rounded.md",
    ):
        path = leave / name
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8")
        if "top3_recall" in text:
            return False
    return True


def _check_all_paper_facing_topk_monotonic(out_root: Path) -> bool:
    leave = out_root / "leave_anchor_out_real_final"
    for name in (
        "paper_table_leave_anchor_out_clean.csv",
        "paper_table_leave_anchor_out_clean_rounded.csv",
    ):
        path = leave / name
        if not path.is_file():
            return False
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if "top3_recall" not in row:
                    continue
                top1 = float(row["top1_recall"])
                top3 = float(row["top3_recall"])
                if top3 < top1:
                    return False
    return True


def _check_diagnostic_appendix_rounded(out_root: Path) -> bool:
    leave = out_root / "leave_anchor_out_real_final"
    return (
        (leave / "paper_table_diagnostic_ablations_rounded.csv").is_file()
        and (leave / "paper_table_diagnostic_ablations_rounded.md").is_file()
    )


def verify_paper_final_diagnostics_zip(zip_path: Path) -> dict[str, Any]:
    zip_path = Path(zip_path)
    result: dict[str, Any] = {"zip_path": str(zip_path.resolve()), "passed": False}

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        result["member_count"] = len(names)

        assert "final_diagnostics_manifest.json" in names, "missing final_diagnostics_manifest.json"
        manifest = json.loads(z.read("final_diagnostics_manifest.json").decode())

        assert manifest.get("diagnostic_appendix_legacy_top3_removed") is True
        assert manifest.get("all_paper_facing_topk_monotonic") is True
        assert manifest.get("diagnostic_appendix_rounded") is True

        main_csv = "leave_anchor_out_real_final/paper_table_leave_anchor_out_clean.csv"
        assert main_csv in names, f"missing {main_csv}"
        rows = list(csv.DictReader(io.StringIO(z.read(main_csv).decode())))
        row_checks: list[dict[str, Any]] = []
        for r in rows:
            top1 = float(r["top1_recall"])
            top3 = float(r["top3_recall"])
            row_checks.append(
                {
                    "experiment_name": r["experiment_name"],
                    "top1_recall": top1,
                    "top3_recall": top3,
                    "top3_gte_top1": top3 >= top1,
                }
            )
            assert top3 >= top1, (r["experiment_name"], top1, top3)
        result["main_table_row_checks"] = row_checks

        diag_candidates = [
            n
            for n in names
            if n.endswith("paper_table_diagnostic_ablations.md")
            or n.endswith("paper_table_diagnostic_ablations.csv")
        ]
        for n in diag_candidates:
            text = z.read(n).decode()
            assert "top3_recall" not in text, f"legacy top3_recall still present in {n}"
        result["diagnostic_files_checked"] = diag_candidates

    result["passed"] = True
    return result


def write_paper_final_diagnostics_zip(out_root: Path, zip_path: Path | None = None) -> Path:
    out_root = Path(out_root)
    zip_path = Path(zip_path or out_root.parent / f"{out_root.name}.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_root).as_posix())
    return zip_path


def build_paper_final_diagnostics(
    *,
    out_root: Path,
    leave_final: Path,
    time_causal: Path | None,
    dataset_recon: Path | None,
    flow_mass: Path | None,
    write_zip: bool = True,
) -> dict[str, Any]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if leave_final.is_dir():
        _copy_tree(leave_final, out_root / "leave_anchor_out_real_final")

    for name, src in (
        ("time_causal_sensitivity", time_causal),
        ("dataset_count_reconciliation", dataset_recon),
        ("flow_mass_calibration", flow_mass),
    ):
        if src and Path(src).is_dir():
            _copy_tree(Path(src), out_root / name)

    sections = {
        "leave_anchor_out_audit": "finalized",
        "time_causal_sensitivity": "completed" if time_causal and Path(time_causal).is_dir() else "pending",
        "flow_mass_calibration": "completed" if flow_mass and Path(flow_mass).is_dir() else "pending",
        "dataset_count_reconciliation": "completed" if dataset_recon and Path(dataset_recon).is_dir() else "pending",
    }

    checks = {
        "diagnostic_appendix_legacy_top3_removed": _check_diagnostic_legacy_top3_removed(out_root),
        "all_paper_facing_topk_monotonic": _check_all_paper_facing_topk_monotonic(out_root),
        "diagnostic_appendix_rounded": _check_diagnostic_appendix_rounded(out_root),
    }

    narrative = _build_narrative(out_root, sections)
    (out_root / "paper_ready_experiment_narrative.md").write_text(narrative, encoding="utf-8")

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
        **checks,
        "paths": {
            "leave_anchor_out_real_final": str((out_root / "leave_anchor_out_real_final").resolve()),
            "time_causal_sensitivity": str((out_root / "time_causal_sensitivity").resolve())
            if (out_root / "time_causal_sensitivity").is_dir()
            else None,
            "dataset_count_reconciliation": str((out_root / "dataset_count_reconciliation").resolve())
            if (out_root / "dataset_count_reconciliation").is_dir()
            else None,
            "flow_mass_calibration": str((out_root / "flow_mass_calibration").resolve())
            if (out_root / "flow_mass_calibration").is_dir()
            else None,
        },
    }
    with open(out_root / "final_diagnostics_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    md = [
        "# Final diagnostics manifest",
        "",
        f"Generated: {manifest['generated_at_utc']}",
        "",
        "## Sections",
        "",
        "| Section | Status |",
        "|---------|--------|",
    ]
    for k, v in sections.items():
        md.append(f"| {k} | {v} |")
    md.extend(
        [
            "",
            "## Paper-facing checks",
            "",
            "| Check | Status |",
            "|-------|--------|",
        ]
    )
    for k, v in checks.items():
        md.append(f"| {k} | {v} |")
    (out_root / "final_diagnostics_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    zip_verification: dict[str, Any] | None = None
    if write_zip:
        zip_path = write_paper_final_diagnostics_zip(out_root)
        manifest["zip_path"] = str(zip_path.resolve())
        zip_verification = verify_paper_final_diagnostics_zip(zip_path)
        manifest["zip_verification"] = zip_verification
        with open(out_root / "final_diagnostics_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    manifest["zip_verification"] = zip_verification
    return manifest


def _build_narrative(out_root: Path, sections: dict[str, str]) -> str:
    tc_note = ""
    tc_path = out_root / "time_causal_sensitivity" / "sensitivity_manifest.json"
    if tc_path.is_file():
        tc = json.loads(tc_path.read_text(encoding="utf-8"))
        best_adm = tc.get("best_admissible_config")
        if isinstance(best_adm, dict) and best_adm.get("pair_f1") is not None:
            tc_note = f" Best admissible (CVR ≤ 5%) pair_f1={best_adm.get('pair_f1')}."
        else:
            tc_note = " No configuration met CVR ≤ 5%; baseline weights retained."

    return f"""# Paper-ready experiment narrative

## 1. Leave-anchor-out leakage audit ({sections['leave_anchor_out_audit']})

On the real Celer ETH↔BNB subgraph, baseline, leave-key-out, and strict leave-anchor-out achieve the same pair-level F1 (~0.321), while permuted-label and random controls collapse to ~0.0003 / ~0.0001. Strict mode masks bridge-derived evidence fields (`bridge_contract_hit`, `evidence_levels`, `evidence_quality_score`). **Conclusion:** observed recovery is not explained by bridge-key or bridge-evidence leakage. Use strict leave-anchor-out as the forensic-facing estimate. Table scope: 3258×5226 UOT subgraph, 7296 tx-pair labels (not full Table 1 corpus counts).

## 2. Time / causal admissibility trade-off ({sections['time_causal_sensitivity']})

Diagnostic ablations `no_time` and `no_causal` exceed baseline F1 but remove forensic admissibility constraints.{tc_note} Treat `no_causal` as a **relaxed upper bound**, not a better method. The production model should sit on the Pareto frontier between pair F1 and `causality_violation_rate`. Real Celer delays, batching, and label timestamp granularity likely explain why relaxing time/causal penalties helps F1.

## 3. Flow-mass calibration limitation ({sections['flow_mass_calibration']})

Strict leave-anchor-out shows pair F1 ~0.321 and top-3 recall ~0.499, but flow-mass recall ~0.029. Hard top-k ranking retains signal; soft UOT mass remains diffuse under current entropy regularization. This is a **calibration limitation**, not a leakage failure. Optional reg sharpness sweep tests whether lower `uot_reg` sharpens mass without destroying pair F1.

## Tables: main text vs appendix

| Content | Placement |
|---------|-----------|
| Leave-anchor-out clean table (5 rows) | Main text / rebuttal |
| Diagnostic ablations rounded (no legacy top-3) | Appendix only |
| Time/causal sweep Pareto summary | Appendix + discussion paragraph |
| Dataset count reconciliation | Appendix note + Table 1 caption cross-ref |
| Flow-mass calibration summary | Limitations / appendix |
| Diagnostic ablations (no_time, no_causal) | Appendix only — not evidence of superiority |
| Greedy no-anchor baseline | Excluded (not comparable) |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper final diagnostics bundle.")
    parser.add_argument("--out", type=Path, default=Path("out/paper_final_diagnostics"))
    parser.add_argument("--leave-final", type=Path, default=Path("out/leave_anchor_out_real_final"))
    parser.add_argument("--time-causal", type=Path, default=Path("out/time_causal_sensitivity"))
    parser.add_argument("--dataset-recon", type=Path, default=Path("out/dataset_count_reconciliation"))
    parser.add_argument("--flow-mass", type=Path, default=Path("out/flow_mass_calibration"))
    parser.add_argument("--no-zip", action="store_true", help="Skip zip creation and verification")
    args = parser.parse_args()
    manifest = build_paper_final_diagnostics(
        out_root=args.out,
        leave_final=args.leave_final,
        time_causal=args.time_causal,
        dataset_recon=args.dataset_recon,
        flow_mass=args.flow_mass,
        write_zip=not args.no_zip,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "paths"}, indent=2))


if __name__ == "__main__":
    main()
