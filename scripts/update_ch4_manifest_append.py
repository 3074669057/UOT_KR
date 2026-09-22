"""Append generated README entries to chapter4 package manifests."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update(pkg: str, manifest_name: str, csv_name: str, checksum_name: str) -> None:
    root = ROOT / "out" / pkg
    rows = list(csv.DictReader((root / csv_name).open(encoding="utf-8")))
    existing = {r["relative_path"] for r in rows}
    ts = datetime.now(timezone.utc).isoformat()
    candidates = [
        ("code/connector_original_reference/README.md", "global", "audit"),
        ("code/abctracer_original_reference/README.md", "global", "audit"),
        ("frozen_outputs/structural_recovery/README.md", "4.3", "audit"),
    ]
    for rel, section, role in candidates:
        if rel in existing:
            continue
        p = root / rel
        if not p.is_file():
            continue
        rows.append(
            {
                "relative_path": rel,
                "source_path": "generated_in_package",
                "package_role": role,
                "experiment_section": section,
                "size_bytes": str(p.stat().st_size),
                "sha256": sha256_file(p),
                "copied_at": ts,
                "notes": "packaging doc",
                "redacted": "False",
            }
        )
    fields = [
        "relative_path",
        "source_path",
        "package_role",
        "experiment_section",
        "size_bytes",
        "sha256",
        "copied_at",
        "notes",
        "redacted",
    ]
    with (root / csv_name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (root / checksum_name).write_text(
        "".join(f"{r['sha256']}  {r['relative_path']}\n" for r in sorted(rows, key=lambda x: x["relative_path"])),
        encoding="utf-8",
    )
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "relative_path": r["relative_path"],
            "source_path": r["source_path"],
            "package_role": r["package_role"],
            "experiment_section": r["experiment_section"],
            "size_bytes": int(r["size_bytes"]),
            "sha256": r["sha256"],
            "copied_at": r["copied_at"],
            "notes": r["notes"],
            "redacted": r["redacted"] == "True",
        }
        for r in rows
    ]
    manifest["file_count"] = len(rows)
    manifest["total_size_bytes"] = sum(int(r["size_bytes"]) for r in rows)
    manifest["generated_at"] = ts
    (root / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{pkg}: {len(rows)} files")


if __name__ == "__main__":
    update("chapter4_repro_package", "MANIFEST.json", "FILE_INVENTORY.csv", "CHECKSUMS.sha256")
    update("chapter4_data_package", "DATA_MANIFEST.json", "DATA_FILE_INVENTORY.csv", "DATA_CHECKSUMS.sha256")
