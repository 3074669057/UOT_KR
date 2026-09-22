"""Static feature provenance registry for leakage-resistant ablation audits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# Each entry documents one feature observed in the RC-UOT / Path B pipeline.
_FEATURE_PROVENANCE: dict[str, dict[str, Any]] = {
    "flow_id": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flow.segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": False,
        "used_in_cost": False,
        "used_in_decode": True,
        "used_in_evaluation": True,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Row identifier only; not used for cross-chain pairing in cost matrix.",
    },
    "amount_usd": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Normalized transfer amount from on-chain value + token decimals.",
    },
    "start_time": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "On-chain block timestamp aggregation.",
    },
    "end_time": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "On-chain block timestamp aggregation.",
    },
    "route_type": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flow.segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Token route / asset-group path from registry, not label pairs.",
    },
    "asset_group": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.labels.flow_segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Asset grouping from Celer evidence export; encodes token path not src-dst pair.",
    },
    "aml_score": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.flow_features",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "AML rule/model score on source txs.",
    },
    "graph_embedding": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.flow_features",
        "used_in_matching": True,
        "used_in_candidate_filter": False,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Optional graph checkpoint embedding.",
    },
    "address_set": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.labels.flow_segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": False,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Address novelty overlap; single-chain addresses only.",
    },
    "tx_hashes": {
        "stage": "flow_segment_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.segment_builder",
        "used_in_matching": True,
        "used_in_candidate_filter": False,
        "used_in_cost": False,
        "used_in_decode": True,
        "used_in_evaluation": True,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Row IDs for decode export; not read by cost matrix for pairing.",
    },
    "evidence_level": {
        "stage": "cost_matrix_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.labels.uot_flow_loader",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Mapped from evidence_quality_mean produced by --build-celer-evidence / flow_segment_builder; bridge RPC receipt tier proxy.",
    },
    "evidence_levels": {
        "stage": "cost_matrix_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.flows.flow_features",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Comma-joined bridge evidence level strings from Celer evidence pipeline.",
    },
    "evidence_quality_score": {
        "stage": "cost_matrix_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.labels.uot_flow_loader",
        "used_in_matching": True,
        "used_in_candidate_filter": False,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Derived from evidence_quality_mean in Celer evidence CSV; used in cost evidence penalty and UOT target marginals.",
    },
    "bridge_contract_hit": {
        "stage": "cost_matrix_construction",
        "source_object": "flow_segment",
        "source_file_or_module": "cross.domain.uot.cost_matrix",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": True,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Bridge-contract interaction flag; applies bridge_prior_bonus in cost matrix.",
    },
    "message_key": {
        "stage": "candidate_generation",
        "source_object": "bridge_event",
        "source_file_or_module": "scripts.run_phase15_celer_abi_decode",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": False,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": True,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Direct Celer bridge message key; encodes cross-chain pairing.",
    },
    "srcTxhash": {
        "stage": "evaluation",
        "source_object": "label_row",
        "source_file_or_module": "label/celer_label.csv",
        "used_in_matching": False,
        "used_in_candidate_filter": False,
        "used_in_cost": False,
        "used_in_decode": False,
        "used_in_evaluation": True,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": True,
        "derived_from_flow_label": True,
        "derived_from_ground_truth": True,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Ground-truth evaluation only.",
    },
    "dstTxhash": {
        "stage": "evaluation",
        "source_object": "label_row",
        "source_file_or_module": "label/celer_label.csv",
        "used_in_matching": False,
        "used_in_candidate_filter": False,
        "used_in_cost": False,
        "used_in_decode": False,
        "used_in_evaluation": True,
        "derived_from_bridge_event": True,
        "derived_from_tx_anchor": True,
        "derived_from_flow_label": True,
        "derived_from_ground_truth": True,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Ground-truth evaluation only.",
    },
    "fake_oracle_pair_id": {
        "stage": "cost_matrix_construction",
        "source_object": "negative_control_probe",
        "source_file_or_module": "cross.domain.labels.anchor_masking",
        "used_in_matching": True,
        "used_in_candidate_filter": True,
        "used_in_cost": False,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": True,
        "allowed_in_leave_anchor_out_strict": False,
        "reason": "Synthetic oracle probe for negative control; must always be masked.",
    },
}


def feature_provenance_lookup(feature_name: str) -> dict[str, Any]:
    key = feature_name.strip().lower().replace("-", "_")
    if key in _FEATURE_PROVENANCE:
        return dict(_FEATURE_PROVENANCE[key])
    return {
        "feature_name": feature_name,
        "stage": "unknown",
        "source_object": "unknown",
        "source_file_or_module": "unknown",
        "used_in_matching": False,
        "used_in_candidate_filter": False,
        "used_in_cost": False,
        "used_in_decode": False,
        "used_in_evaluation": False,
        "derived_from_bridge_event": False,
        "derived_from_tx_anchor": False,
        "derived_from_flow_label": False,
        "derived_from_ground_truth": False,
        "allowed_in_leave_anchor_out_strict": True,
        "reason": "Unregistered field; strict audit treats as allowed unless name matches forbidden patterns.",
    }


def forbidden_in_strict_mode(feature_name: str) -> bool:
    prov = feature_provenance_lookup(feature_name)
    if prov.get("allowed_in_leave_anchor_out_strict") is False:
        return True
    return bool(
        prov.get("derived_from_bridge_event")
        and (prov.get("used_in_cost") or prov.get("used_in_candidate_filter") or prov.get("used_in_matching"))
        and feature_name.lower()
        in {
            "evidence_level",
            "evidence_levels",
            "evidence_quality_score",
            "evidence_quality_mean",
            "bridge_contract_hit",
        }
    )


def is_bridge_derived_feature(feature_name: str) -> bool:
    prov = feature_provenance_lookup(feature_name)
    return bool(prov.get("derived_from_bridge_event"))


def build_feature_provenance_report(
    observed_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    names = sorted(set(observed_features or list(_FEATURE_PROVENANCE.keys())))
    rows: list[dict[str, Any]] = []
    for name in names:
        prov = feature_provenance_lookup(name)
        row = {"feature_name": name, **prov}
        rows.append(row)
    return rows


def write_feature_provenance_reports(out_dir: Path, observed_features: list[str] | None = None) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_feature_provenance_report(observed_features)
    csv_path = out_dir / "feature_provenance_report.csv"
    json_path = out_dir / "feature_provenance_report.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"rows": rows, "forbidden_strict_count": sum(1 for r in rows if not r.get("allowed_in_leave_anchor_out_strict"))},
            f,
            indent=2,
        )
    return csv_path, json_path
