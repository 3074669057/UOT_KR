"""Run output directory layout: artifacts grouped under stable subfolders.

Writers use :func:`output_file` / :class:`RunLayout`. Readers use :func:`locate_output_file`
to support both the nested layout and legacy flat ``out_dir`` trees.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# Canonical relative path (from run root) for each well-known filename.
OUTPUT_RELATIVE: Final[dict[str, str]] = {
    # evidence/
    "evidence_eth.csv": "evidence/evidence_eth.csv",
    "evidence_bnb.csv": "evidence/evidence_bnb.csv",
    "evidence_candidates.csv": "evidence/evidence_candidates.csv",
    "evidence_candidates.json": "evidence/evidence_candidates.json",
    "evidence_validation.json": "evidence/evidence_validation.json",
    "evidence_export_debug.json": "evidence/evidence_export_debug.json",
    "candidate_pool_raw.csv": "evidence/candidate_pool_raw.csv",
    "receipt_verify_debug.csv": "evidence/receipt_verify_debug.csv",
    # uot/
    "uot_flow_segments_eth.csv": "uot/uot_flow_segments_eth.csv",
    "uot_flow_segments_bnb.csv": "uot/uot_flow_segments_bnb.csv",
    # Label-layer canonical segment exports (NOT UOT solver snapshots).
    "flow_segments_eth.csv": "labels/flow_segments_eth.csv",
    "flow_segments_bnb.csv": "labels/flow_segments_bnb.csv",
    "uot_transport_plan.csv": "uot/uot_transport_plan.csv",
    "uot_cost_matrix.csv": "uot/uot_cost_matrix.csv",
    "uot_cost_components.csv": "uot/uot_cost_components.csv",
    "uot_flow_correspondence.csv": "uot/uot_flow_correspondence.csv",
    "uot_unmatched_mass.csv": "uot/uot_unmatched_mass.csv",
    "uot_summary.json": "uot/uot_summary.json",
    "uot_diagnostics.json": "uot/uot_diagnostics.json",
    "uot_split_merge_summary.json": "uot/uot_split_merge_summary.json",
    "traceability_index.csv": "uot/traceability_index.csv",
    "uot_marginals.csv": "uot/uot_marginals.csv",
    "causal_feasibility_summary.json": "uot/causal_feasibility_summary.json",
    "matching_transport_matrix.npz": "uot/matching_transport_matrix.npz",
    # eval/
    "uot_evaluation_metrics.json": "eval/uot_evaluation_metrics.json",
    "uot_eval_by_pattern.csv": "eval/uot_eval_by_pattern.csv",
    "synthetic_eval_by_scenario.csv": "eval/synthetic_eval_by_scenario.csv",
    "uot_eval_summary.txt": "eval/uot_eval_summary.txt",
    "ablation_metrics.csv": "eval/ablation_metrics.csv",
    "threshold_sensitivity.csv": "experiments/threshold_sensitivity.csv",
    # matching/
    "matching_pairs.csv": "matching/matching_pairs.csv",
    "matching_metrics.json": "matching/matching_metrics.json",
    "matching_evidence.json": "matching/matching_evidence.json",
    "matching_case_report.json": "matching/matching_case_report.json",
    "matching_address_mapping.csv": "matching/matching_address_mapping.csv",
    "matching_flow_correspondence.json": "matching/matching_flow_correspondence.json",
    "matching_unmatched_mass.json": "matching/matching_unmatched_mass.json",
    "matching_flow_metrics.json": "matching/matching_flow_metrics.json",
    "flow_level_metrics.json": "matching/flow_level_metrics.json",
    "path_b_pairs.csv": "matching/path_b_pairs.csv",
    "path_b_pairs_all_candidates.csv": "matching/path_b_pairs_all_candidates.csv",
    "path_b_pairs_accepted.csv": "matching/path_b_pairs_accepted.csv",
    "path_b_low_confidence_candidates.csv": "matching/path_b_low_confidence_candidates.csv",
    "path_b_output_summary.json": "matching/path_b_output_summary.json",
    "path_b_evidence.json": "matching/path_b_evidence.json",
    "path_b_case_report.json": "matching/path_b_case_report.json",
    "path_b_vs_label.json": "matching/path_b_vs_label.json",
    "baseline_hungarian.csv": "matching/baseline_hungarian.csv",
    "baseline_greedy.csv": "matching/baseline_greedy.csv",
    "address_mapping.csv": "matching/address_mapping.csv",
    # labels/ (celer weak-label pipeline)
    "tx_anchor_labels.csv": "labels/tx_anchor_labels.csv",
    "tx_anchor_candidates.csv": "labels/tx_anchor_candidates.csv",
    "tx_anchor_low_confidence.csv": "labels/tx_anchor_low_confidence.csv",
    "tx_anchor_receiver_mismatch_debug.csv": "labels/tx_anchor_receiver_mismatch_debug.csv",
    "tx_anchor_diagnostics.json": "labels/tx_anchor_diagnostics.json",
    "flow_labels.csv": "labels/flow_labels.csv",
    "flow_labels_weak.csv": "labels/flow_labels_weak.csv",
    "flow_labels_from_celer.csv": "labels/flow_labels_from_celer.csv",
    "flow_labels_merged.csv": "labels/flow_labels_merged.csv",
    "tx_anchor_labels_weak.csv": "labels/tx_anchor_labels_weak.csv",
    "tx_anchor_labels_from_celer.csv": "labels/tx_anchor_labels_from_celer.csv",
    "tx_anchor_labels_merged.csv": "labels/tx_anchor_labels_merged.csv",
    "label_diagnostics.json": "labels/label_diagnostics.json",
    "flow_label_stats.json": "labels/flow_label_stats.json",
    "validation_report.json": "labels/validation_report.json",
    "tx_to_flow_map.csv": "labels/tx_to_flow_map.csv",
    "synthetic_flow_labels.csv": "labels/synthetic_flow_labels.csv",
    "synthetic_uot_eval_metrics.json": "labels/synthetic_uot_eval_metrics.json",
    # path_a/
    "pairs.csv": "path_a/pairs.csv",
    "ambiguous.csv": "path_a/ambiguous.csv",
    "edges.csv": "path_a/edges.csv",
    "path_a_vs_label.json": "path_a/path_a_vs_label.json",
    # aml/
    "aml_suspect_txs.csv": "aml/aml_suspect_txs.csv",
    "aml_summary.json": "aml/aml_summary.json",
    "aml_rule_hits.csv": "aml/aml_rule_hits.csv",
    # reports/
    "run_report.json": "reports/run_report.json",
    "paper_artifact_validation.json": "reports/paper_artifact_validation.json",
    "paper_experiment_summary.md": "reports/paper_experiment_summary.md",
    "relation_summary.json": "reports/relation_summary.json",
    "run_audit_report.md": "reports/run_audit_report.md",
    "run_audit_report.json": "reports/run_audit_report.json",
    "module1_ml_txs.json": "reports/module1_ml_txs.json",
    # validation/
    "token_route_validation.json": "validation/token_route_validation.json",
    # experiments/
    "ablation_results.csv": "experiments/ablation_results.csv",
    "uot_experiment_summary.json": "experiments/uot_experiment_summary.json",
    "artifact_inventory.json": "experiments/artifact_inventory.json",
    # logs/
    "run.log": "logs/run.log",
}


def output_file(root: Path, filename: str) -> Path:
    """Return writable path for ``filename`` (mkdir parents)."""
    root = Path(root)
    rel = OUTPUT_RELATIVE.get(filename, filename)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def output_subpath(root: Path, *parts: str) -> Path:
    """Writable path ``root / parts[0] / ...`` with parents created."""
    root = Path(root)
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Files that may legitimately exist in more than one subdirectory (read tries in order).
READ_FALLBACKS: Final[dict[str, tuple[str, ...]]] = {
    # Never prefer uot/ for label-layer segments (uot/ may hold solver subgraph snapshots).
    "flow_segments_eth.csv": (
        "labels/flow_segments_eth.csv",
        "label_layer_v1/flow_segments_eth.csv",
        "flow_segments_eth.csv",
        "uot/flow_segments_eth.csv",
    ),
    "flow_segments_bnb.csv": (
        "labels/flow_segments_bnb.csv",
        "label_layer_v1/flow_segments_bnb.csv",
        "flow_segments_bnb.csv",
        "uot/flow_segments_bnb.csv",
    ),
}


def locate_output_file(root: Path, filename: str) -> Path:
    """Return an existing path if found (preferred nested paths then legacy flat), else canonical write path."""
    root = Path(root)
    if filename in READ_FALLBACKS:
        for rel in READ_FALLBACKS[filename]:
            p = root / rel
            if p.is_file():
                return p
        return output_file(root, filename)
    rel = OUTPUT_RELATIVE.get(filename, filename)
    nested = root / rel
    if nested.is_file():
        return nested
    flat = root / filename
    if flat.is_file():
        return flat
    return nested


def locate_output_dir(root: Path, sub: str) -> Path:
    """Return ``root/sub`` if it exists as a dir, else ``root`` (legacy)."""
    root = Path(root)
    d = root / sub
    if d.is_dir():
        return d
    return root


class RunLayout:
    """Convenience accessors for standard run subdirectories."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def file(self, filename: str) -> Path:
        return output_file(self.root, filename)

    def locate(self, filename: str) -> Path:
        return locate_output_file(self.root, filename)

    def _d(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def evidence(self) -> Path:
        return self._d("evidence")

    @property
    def uot(self) -> Path:
        return self._d("uot")

    @property
    def matching(self) -> Path:
        return self._d("matching")

    @property
    def labels(self) -> Path:
        return self._d("labels")

    @property
    def path_a(self) -> Path:
        return self._d("path_a")

    @property
    def aml(self) -> Path:
        return self._d("aml")

    @property
    def eval(self) -> Path:
        return self._d("eval")

    @property
    def reports(self) -> Path:
        return self._d("reports")

    @property
    def validation(self) -> Path:
        return self._d("validation")

    @property
    def experiments(self) -> Path:
        return self._d("experiments")

    @property
    def logs(self) -> Path:
        return self._d("logs")

    @property
    def paper_tables(self) -> Path:
        return self._d("paper_tables")
