#!/usr/bin/env python3
"""Zip chapter4 repro/data packages, validate, write reviewer artifact report."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "reviewer_artifacts"
REPRO_SRC = ROOT / "out" / "chapter4_repro_package"
DATA_SRC = ROOT / "out" / "chapter4_data_package"

EXPECTED_INVENTORY = {"chapter4_repro_package": 407, "chapter4_data_package": 260}

SKIP_DIR_NAMES = {"__pycache__", ".git", ".venv", ".pytest_cache"}
SKIP_FILE_NAMES = {".DS_Store", "Thumbs.db"}
SKIP_FILE_PATTERNS = [
    re.compile(r"api-key-cross\.json$", re.I),
    re.compile(r"\.pyc$"),
    re.compile(r"\.log$"),
    re.compile(r"run\.log$"),
]

REPRO_REQUIRED = [
    "chapter4_repro_package/README.md",
    "chapter4_repro_package/REPRODUCIBILITY.md",
    "chapter4_repro_package/MANIFEST.json",
    "chapter4_repro_package/FILE_INVENTORY.csv",
    "chapter4_repro_package/CHECKSUMS.sha256",
    "chapter4_repro_package/paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.json",
    "chapter4_repro_package/frozen_outputs/baseline_compare/routeA_v2/degradation_curve_v2.json",
    "chapter4_repro_package/paper_artifacts/figures/ch4/fig5_structural_recovery.png",
    "chapter4_repro_package/paper_artifacts/figures/ch4/fig6_coverage_scope.png",
    "chapter4_repro_package/paper_artifacts/figures/ch4/fig7_baseline_grouped_metrics.png",
    "chapter4_repro_package/paper_artifacts/appendix/appendix_A_fixed_delay_anchor_audit.md",
    "chapter4_repro_package/paper_artifacts/appendix/appendix_B_connector_native_diagnostic.md",
]

DATA_REQUIRED = [
    "chapter4_data_package/README_DATA.md",
    "chapter4_data_package/DATA_MANIFEST.json",
    "chapter4_data_package/DATA_FILE_INVENTORY.csv",
    "chapter4_data_package/DATA_CHECKSUMS.sha256",
    "chapter4_data_package/frozen_outputs/baseline_compare_routeA_v2/degradation_curve_v2.json",
    "chapter4_data_package/paper_artifacts/tables/main_table_rc_uot_q_fixed_delay.json",
    "chapter4_data_package/paper_artifacts/figures/ch4/fig5_structural_recovery.png",
    "chapter4_data_package/paper_artifacts/figures/ch4/fig6_coverage_scope.png",
    "chapter4_data_package/paper_artifacts/figures/ch4/fig7_baseline_grouped_metrics.png",
]

FORBIDDEN_IN_ZIP = [
    re.compile(r"api-key-cross\.json", re.I),
    re.compile(r"[/\\]\.venv[/\\]"),
    re.compile(r"[/\\]\.git[/\\]"),
    re.compile(r"[/\\]__pycache__[/\\]"),
    re.compile(r"private[_-]?key", re.I),
    re.compile(r"credential", re.I),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def should_skip(path: Path) -> bool:
    parts = path.parts
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    rel = path.as_posix()
    for pat in SKIP_FILE_PATTERNS:
        if pat.search(rel):
            return True
    return False


def collect_files(src: Path, root_name: str) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or should_skip(p):
            continue
        rel = p.relative_to(src).as_posix()
        arc = f"{root_name}/{rel}"
        files.append((p, arc))
    return files


def inventory_count(src: Path, inv_name: str) -> int:
    inv = src / inv_name
    if not inv.is_file():
        return 0
    with inv.open(encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def create_zip(src: Path, root_name: str, zip_path: Path) -> dict[str, Any]:
    files = collect_files(src, root_name)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp, arc in files:
            zf.write(fp, arcname=arc)
    return {
        "zip_path": str(zip_path),
        "size_bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "files_in_zip": len(files),
        "arcs": [a for _, a in files],
    }


def forbidden_in_zip(arcs: list[str]) -> list[str]:
    hits = []
    for arc in arcs:
        for pat in FORBIDDEN_IN_ZIP:
            if pat.search(arc.replace("\\", "/")):
                hits.append(arc)
                break
    return hits


def verify_extracted(extract_root: Path, required: list[str]) -> dict[str, Any]:
    missing = [r for r in required if not (extract_root / r).is_file()]
    return {"missing_required": missing, "pass": len(missing) == 0}


def check_defaults_no_keys(extract_root: Path) -> dict[str, Any]:
    p = extract_root / "chapter4_repro_package/code/configs_sanitized/defaults.json"
    if not p.is_file():
        return {"pass": False, "reason": "defaults.json not found"}
    data = json.loads(p.read_text(encoding="utf-8"))
    issues = []
    for key in ("nodereal", "eth_nodereal"):
        block = data.get(key, {})
        if block.get("api_keys"):
            issues.append(f"{key}.api_keys non-empty")
        if block.get("rpc_urls"):
            issues.append(f"{key}.rpc_urls non-empty")
    return {"pass": not issues, "issues": issues}


def check_route_a_v1_rejected(extract_root: Path) -> dict[str, Any]:
    repro = extract_root / "chapter4_repro_package"
    rejected = repro / "frozen_outputs/baseline_compare/rejected_audits/routeA_v1_rejected"
    status_md = rejected / "REJECTED_STATUS.md"
    canonical_v2 = repro / "frozen_outputs/baseline_compare/routeA_v2/degradation_curve_v2.json"
    v1_eval = rejected / "connector_full_native_raw_eval_v1.json"
    out = {
        "rejected_dir_exists": rejected.is_dir(),
        "REJECTED_STATUS.md": status_md.is_file(),
        "v1_eval_in_rejected_only": v1_eval.is_file(),
        "canonical_v2_present": canonical_v2.is_file(),
    }
    # v1 eval should not appear outside rejected (except diff json names)
    stray = []
    for p in repro.rglob("connector_full_native_raw_eval_v1.json"):
        if "rejected_audits" not in p.as_posix():
            stray.append(p.as_posix())
    out["v1_eval_outside_rejected"] = stray
    out["pass"] = (
        out["REJECTED_STATUS.md"]
        and out["canonical_v2_present"]
        and not out["v1_eval_outside_rejected"]
    )
    return out


def check_abctracer_blocked(extract_root: Path) -> dict[str, Any]:
    gate = extract_root / "chapter4_repro_package/frozen_outputs/baseline_compare/gate_reports/abctracer_gate1_report.md"
    readme = extract_root / "chapter4_repro_package/code/abctracer_original_reference/README.md"
    checkpoint_hits = []
    for root in [extract_root / "chapter4_repro_package", extract_root / "chapter4_data_package"]:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".pth", ".pt", ".ckpt", ".bin"} and "checkpoint" in p.name.lower():
                checkpoint_hits.append(p.as_posix())
    text = gate.read_text(encoding="utf-8", errors="replace") if gate.is_file() else ""
    blocked = "BLOCKED" in text or "blocked" in text.lower()
    return {
        "gate_report": gate.is_file(),
        "readme": readme.is_file(),
        "blocked_documented": blocked,
        "checkpoint_files": checkpoint_hits,
        "pass": gate.is_file() and blocked and not checkpoint_hits,
    }


def sample_checksum_verify(extract_root: Path, n: int = 20) -> dict[str, Any]:
    checksum_path = extract_root / "chapter4_repro_package/CHECKSUMS.sha256"
    if not checksum_path.is_file():
        return {"pass": False, "reason": "CHECKSUMS.sha256 missing"}
    lines = checksum_path.read_text(encoding="utf-8").strip().splitlines()
    import random

    random.seed(42)
    sample = random.sample(lines, min(n, len(lines)))
    mismatches = []
    ok = 0
    for line in sample:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected, rel = parts[0], parts[1].strip()
        fp = extract_root / "chapter4_repro_package" / rel
        if not fp.is_file():
            mismatches.append({"rel": rel, "error": "missing"})
            continue
        actual = sha256_file(fp)
        if actual != expected:
            mismatches.append({"rel": rel, "expected": expected, "actual": actual})
        else:
            ok += 1
    return {"sampled": len(sample), "ok": ok, "mismatches": mismatches, "pass": not mismatches}


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    OUT.mkdir(parents=True, exist_ok=True)

    pre_counts = {
        "chapter4_repro_package": {
            "disk_files": len(collect_files(REPRO_SRC, "chapter4_repro_package")),
            "inventory_files": inventory_count(REPRO_SRC, "FILE_INVENTORY.csv"),
            "expected_inventory": EXPECTED_INVENTORY["chapter4_repro_package"],
        },
        "chapter4_data_package": {
            "disk_files": len(collect_files(DATA_SRC, "chapter4_data_package")),
            "inventory_files": inventory_count(DATA_SRC, "DATA_FILE_INVENTORY.csv"),
            "expected_inventory": EXPECTED_INVENTORY["chapter4_data_package"],
        },
    }

    meta_only_repro = pre_counts["chapter4_repro_package"]["disk_files"] - pre_counts["chapter4_repro_package"]["inventory_files"]
    meta_only_data = pre_counts["chapter4_data_package"]["disk_files"] - pre_counts["chapter4_data_package"]["inventory_files"]

    repro_zip = OUT / "chapter4_repro_package.zip"
    data_zip = OUT / "chapter4_data_package.zip"

    repro_info = create_zip(REPRO_SRC, "chapter4_repro_package", repro_zip)
    data_info = create_zip(DATA_SRC, "chapter4_data_package", data_zip)

    (OUT / "chapter4_repro_package.zip.sha256").write_text(
        f"{repro_info['sha256']}  chapter4_repro_package.zip\n", encoding="utf-8"
    )
    (OUT / "chapter4_data_package.zip.sha256").write_text(
        f"{data_info['sha256']}  chapter4_data_package.zip\n", encoding="utf-8"
    )

    forbidden_repro = forbidden_in_zip(repro_info["arcs"])
    forbidden_data = forbidden_in_zip(data_info["arcs"])

    validations: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ch4_zip_verify_") as tmp:
        tmp_path = Path(tmp)
        repro_extract = tmp_path / "repro"
        data_extract = tmp_path / "data"
        repro_extract.mkdir()
        data_extract.mkdir()

        with zipfile.ZipFile(repro_zip, "r") as zf:
            zf.extractall(repro_extract)
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(data_extract)

        repro_post = len([p for p in repro_extract.rglob("*") if p.is_file()])
        data_post = len([p for p in data_extract.rglob("*") if p.is_file()])

        validations["repro_required"] = verify_extracted(repro_extract, REPRO_REQUIRED)
        validations["data_required"] = verify_extracted(data_extract, DATA_REQUIRED)
        validations["defaults_sanitized"] = check_defaults_no_keys(repro_extract)
        validations["route_a_v1_rejected"] = check_route_a_v1_rejected(repro_extract)
        validations["abctracer_blocked"] = check_abctracer_blocked(repro_extract)
        validations["checksum_sample"] = sample_checksum_verify(repro_extract, 20)
        validations["forbidden_in_zip"] = {
            "repro": forbidden_repro,
            "data": forbidden_data,
            "pass": not forbidden_repro and not forbidden_data,
        }
        validations["extract_file_counts"] = {
            "repro": repro_post,
            "data": data_post,
        }

    count_match = (
        repro_info["files_in_zip"] == pre_counts["chapter4_repro_package"]["disk_files"]
        and data_info["files_in_zip"] == pre_counts["chapter4_data_package"]["disk_files"]
        and repro_post == repro_info["files_in_zip"]
        and data_post == data_info["files_in_zip"]
    )

    warnings = []
    if pre_counts["chapter4_repro_package"]["inventory_files"] != EXPECTED_INVENTORY["chapter4_repro_package"]:
        warnings.append("repro inventory count differs from expected 407")
    if pre_counts["chapter4_data_package"]["inventory_files"] != EXPECTED_INVENTORY["chapter4_data_package"]:
        warnings.append("data inventory count differs from expected 260")
    warnings.append(
        f"Full package disk files (repro {pre_counts['chapter4_repro_package']['disk_files']}, "
        f"data {pre_counts['chapter4_data_package']['disk_files']}) include {meta_only_repro}/{meta_only_data} "
        "packaging meta files not listed in FILE_INVENTORY (README, MANIFEST, audits, exclusion notes)."
    )

    all_pass = (
        count_match
        and validations["repro_required"]["pass"]
        and validations["data_required"]["pass"]
        and validations["defaults_sanitized"]["pass"]
        and validations["route_a_v1_rejected"]["pass"]
        and validations["abctracer_blocked"]["pass"]
        and validations["checksum_sample"]["pass"]
        and validations["forbidden_in_zip"]["pass"]
    )

    manifest = {
        "generated_at": ts,
        "artifacts": [
            {
                "name": "chapter4_repro_package.zip",
                "path": str(repro_zip.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": repro_info["size_bytes"],
                "sha256": repro_info["sha256"],
                "files_before_zip": pre_counts["chapter4_repro_package"]["disk_files"],
                "files_in_zip": repro_info["files_in_zip"],
                "files_after_extract": validations["extract_file_counts"]["repro"],
                "inventory_file_count": pre_counts["chapter4_repro_package"]["inventory_files"],
            },
            {
                "name": "chapter4_data_package.zip",
                "path": str(data_zip.relative_to(ROOT)).replace("\\", "/"),
                "size_bytes": data_info["size_bytes"],
                "sha256": data_info["sha256"],
                "files_before_zip": pre_counts["chapter4_data_package"]["disk_files"],
                "files_in_zip": data_info["files_in_zip"],
                "files_after_extract": validations["extract_file_counts"]["data"],
                "inventory_file_count": pre_counts["chapter4_data_package"]["inventory_files"],
            },
        ],
        "sha256_sidecar_files": [
            "chapter4_repro_package.zip.sha256",
            "chapter4_data_package.zip.sha256",
        ],
        "pre_zip_counts": pre_counts,
        "validations": validations,
        "warnings": warnings,
        "reviewer_upload_ready": all_pass,
        "sensitive_files_found": not validations["forbidden_in_zip"]["pass"],
    }

    (OUT / "reviewer_artifact_packaging_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Reviewer artifact packaging report\n",
        f"Generated: {ts}\n\n",
        "## Zip artifacts\n\n",
        "| Artifact | Path | Size | SHA256 |\n",
        "|----------|------|------|--------|\n",
        f"| repro zip | `out/reviewer_artifacts/chapter4_repro_package.zip` | "
        f"{repro_info['size_bytes'] / (1024**2):.2f} MB | `{repro_info['sha256']}` |\n",
        f"| data zip | `out/reviewer_artifacts/chapter4_data_package.zip` | "
        f"{data_info['size_bytes'] / (1024**2):.2f} MB | `{data_info['sha256']}` |\n\n",
        "## File counts\n\n",
        "| Package | Inventory (FILE_INVENTORY) | Disk (zip input) | In zip | After extract |\n",
        "|---------|---------------------------|------------------|--------|---------------|\n",
        f"| repro | {pre_counts['chapter4_repro_package']['inventory_files']} | "
        f"{pre_counts['chapter4_repro_package']['disk_files']} | {repro_info['files_in_zip']} | "
        f"{validations['extract_file_counts']['repro']} |\n",
        f"| data | {pre_counts['chapter4_data_package']['inventory_files']} | "
        f"{pre_counts['chapter4_data_package']['disk_files']} | {data_info['files_in_zip']} | "
        f"{validations['extract_file_counts']['data']} |\n\n",
        "**Note:** Inventory counts (407/260) exclude packaging metadata (README, MANIFEST, CHECKSUMS, audit docs). "
        f"Zip includes full directories ({meta_only_repro} repro + {meta_only_data} data meta files).\n\n",
        "## Checksum spot-check (20 files from CHECKSUMS.sha256)\n\n",
        f"- Sampled: {validations['checksum_sample']['sampled']}\n",
        f"- OK: {validations['checksum_sample']['ok']}\n",
        f"- Pass: {validations['checksum_sample']['pass']}\n\n",
        "## Sensitive / forbidden content\n\n",
        f"- Forbidden paths in zip: repro={len(forbidden_repro)}, data={len(forbidden_data)}\n",
        f"- defaults.json sanitized: {validations['defaults_sanitized']['pass']}\n",
        f"- api-key-cross.json packaged: **no**\n\n",
        "## Route A v1 / ABCTracer\n\n",
        f"- Route A v1 rejected audit: {validations['route_a_v1_rejected']['pass']}\n",
        f"- ABCTRacer BLOCKED documented, no checkpoint: {validations['abctracer_blocked']['pass']}\n\n",
        "## Required files after extract\n\n",
        f"- Repro required: {validations['repro_required']['pass']}\n",
        f"- Data required: {validations['data_required']['pass']}\n\n",
        "## Warnings\n\n",
    ]
    for w in warnings:
        report_lines.append(f"- {w}\n")
    if validations["repro_required"]["missing_required"]:
        report_lines.append(f"- Missing repro: {validations['repro_required']['missing_required']}\n")
    if validations["data_required"]["missing_required"]:
        report_lines.append(f"- Missing data: {validations['data_required']['missing_required']}\n")

    report_lines.append("\n## Reviewer upload readiness\n\n")
    report_lines.append(
        f"**{'READY' if all_pass else 'NOT READY'}** — reviewer artifact upload "
        f"({'all checks passed' if all_pass else 'see failures above'}).\n"
    )

    (OUT / "reviewer_artifact_packaging_report.md").write_text("".join(report_lines), encoding="utf-8")

    print(json.dumps({"all_pass": all_pass, "repro_mb": repro_info["size_bytes"] / (1024**2), "data_mb": data_info["size_bytes"] / (1024**2)}, indent=2))


if __name__ == "__main__":
    main()
