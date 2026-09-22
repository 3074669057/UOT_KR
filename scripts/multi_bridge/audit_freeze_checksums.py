"""Generate SHA256 checksum manifest for the candidate main run (PHASE 1 freeze)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges"
AUDIT = REPO / "out" / "multi_bridge_expansion" / "faithful_flow_structural_three_bridges_audit"
BRIDGES = ("Celer", "Multi", "Poly")
SEEDS = (42, 43, 44, 45, 46)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[Path]:
    files: list[Path] = []
    files += [
        OUT / "aggregated" / "structural_per_seed.csv",
        OUT / "aggregated" / "structural_aggregated.csv",
        OUT / "aggregated" / "verification_report.json",
        OUT / "aggregated" / "verification_report.md",
        OUT / "feature_stats" / "feature_sanity.json",
        OUT / "feature_stats" / "feature_sanity.csv",
        OUT / "diagnostics" / "diagnostics.json",
        OUT / "diagnostics" / "diagnostics.md",
        OUT / "diagnostics" / "one_to_one_flow_baselines_per_seed.csv",
        OUT / "baselines" / "structural_three_bridges.json",
        OUT / "baselines" / "baseline_structural_per_seed.csv",
        OUT / "audit" / "feature_provenance.md",
        OUT / "audit" / "audit_summary.md",
        OUT / "RUN_MANIFEST.md",
        OUT / "figures" / "fig5b_three_bridge_structural_recovery.png",
        OUT / "figures" / "fig5b_three_bridge_structural_recovery.pdf",
        OUT / "figures" / "fig5b_three_bridge_structural_recovery.svg",
        OUT / "figures" / "fig5b_three_bridge_structural_recovery_data.csv",
    ]
    for br in BRIDGES:
        files += [OUT / "feature_stats" / br / f for f in ("flow_segments_eth.csv", "flow_segments_bnb.csv", "flow_labels.csv", "feature_sanity.json")]
        for seed in SEEDS:
            root = OUT / "per_seed" / br / f"seed_{seed}"
            files += [
                root / "run_config.json",
                root / "flow_segments_eth_synth.csv",
                root / "flow_segments_bnb_synth.csv",
                root / "labels" / "synthetic_flow_labels.csv",
                root / "labels" / "synthetic_uot_eval_metrics.json",
                root / "eval" / "uot_evaluation_metrics.json",
                root / "uot" / "uot_transport_plan.csv",
                root / "uot" / "uot_summary.json",
            ]
    # key inputs (data maps / prices)
    files += [
        REPO / "data" / "Token" / "token_prices_usd.json",
        REPO / "data" / "Token" / "multi_any_token_map.json",
        REPO / "data" / "Token" / "poly_eth_bsc_token_map.json",
        # frozen reference (must remain untouched)
        REPO / "out" / "chapter4_repro_package" / "frozen_outputs" / "structural_recovery" / "synthetic_eval_aggregated.json",
        # pipeline scripts (reproducibility)
        REPO / "scripts" / "multi_bridge" / "faithful_flow_features.py",
        REPO / "scripts" / "multi_bridge" / "run_faithful_flow_structural.py",
        REPO / "scripts" / "multi_bridge" / "run_structural_three_bridges.py",
        REPO / "scripts" / "multi_bridge" / "verify_structural_results.py",
        REPO / "src" / "cross" / "domain" / "evaluation" / "semi_synthetic_flows.py",
        REPO / "src" / "cross" / "domain" / "evaluation" / "synthetic_segment_subgraph.py",
        REPO / "src" / "cross" / "domain" / "labels" / "uot_flow_loader.py",
        REPO / "manuscript_final" / "04_experiments.md",
    ]
    return [p for p in files if p.is_file()]


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated": "2026-09-01",
        "git_commit": "d5cd14d",
        "git_tag": "faithful-three-bridge-structural-candidate-2026-09-01",
        "files": [],
    }
    for p in collect():
        manifest["files"].append({"path": str(p.relative_to(REPO)), "sha256": sha256(p), "bytes": p.stat().st_size})
    (AUDIT / "checksums_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"checksummed {len(manifest['files'])} files -> {AUDIT / 'checksums_manifest.json'}")


if __name__ == "__main__":
    main()
