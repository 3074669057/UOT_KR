"""V3b-specific mask variants (does not modify v3 registry)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from cross.baseline_compare.bridge_semantic_masking import (
    I_FIELDS,
    TIMESTAMP_POLICY_DEFAULT,
    BridgeSemanticField,
    MaskSpec,
    get_mask_spec,
    iter_b_subset_mask_specs,
)
from cross.baseline_compare.v3b_high_f1.constants import CHAIN_OBSERVABLES_MASK, STRICT_NO_ALL_MASK

CHAIN_OBSERVABLES_FIELDS = [
    "receiver",
    "asset_s",
    "dstChain",
    "event",
    "bridge",
    "sender",
]


def b_subset_mask_ids() -> list[str]:
    return [s.mask_id for s in iter_b_subset_mask_specs()]


def eval_mask_ids() -> list[str]:
    """31 B-subsets + strict no_all + chain-observables variant."""
    ids = b_subset_mask_ids()
    if STRICT_NO_ALL_MASK not in ids:
        ids.append(STRICT_NO_ALL_MASK)
    ids.append(CHAIN_OBSERVABLES_MASK)
    return ids


def get_v3b_mask_spec(mask_id: str) -> MaskSpec:
    if mask_id == CHAIN_OBSERVABLES_MASK:
        return _chain_observables_spec()
    return get_mask_spec(mask_id)


def _chain_observables_spec() -> MaskSpec:
    """Bridge metadata missing; chain-observable amount/time/graph retained for RC-UOT-Q."""
    base = get_mask_spec(STRICT_NO_ALL_MASK)
    fields_set = frozenset(CHAIN_OBSERVABLES_FIELDS)
    connector_actions: list[dict[str, Any]] = []
    for f in CHAIN_OBSERVABLES_FIELDS:
        if f in I_FIELDS:
            connector_actions.append(
                {"field": f, "action": "omit_from_wl_adapter", "note": "metadata cleared in audit"}
            )
        elif f == "receiver":
            connector_actions.append({"field": "receiver", "action": "clear_string", "column": "args.receiver", "value": ""})
        elif f == "asset_s":
            connector_actions.append({"field": "asset_s", "action": "clear_string", "column": "args.asset_s", "value": ""})
        elif f == "dstChain":
            connector_actions.append({"field": "dstChain", "action": "clear_string", "column": "args.dstChain", "value": ""})
    connector_actions.extend(
        [
            {"field": "amount", "action": "zero_float", "column": "args.amount", "value": 0.0},
            {"field": "timestamp", "action": "zero_float", "column": "timestamp", "value": 0.0},
        ]
    )

    rc_actions: list[dict[str, Any]] = [
        {"action": "zero_weight", "key": "route", "value": 0.0},
        {"action": "zero_weight", "key": "graph", "value": 0.0},
        {"action": "zero_weight", "key": "novelty", "value": 0.0},
        {"action": "mask_flow_fields", "fields": ["addresses", "address_features", "address_count"]},
        {
            "action": "mask_flow_fields",
            "fields": ["token_symbol", "asset_group", "route_type"],
            "neutralize_route_type_to": "unknown",
            "no_independent_knob": True,
        },
        {
            "action": "mask_flow_fields",
            "fields": ["route_id", "route_type"],
            "neutralize_route_type_to": "unknown",
            "no_independent_knob": True,
        },
        {
            "action": "retain_decode_chain_timestamp_lookup",
            "policy": TIMESTAMP_POLICY_DEFAULT,
            "note": "chain-observable block/tx timestamps retained for decode",
        },
        {
            "action": "retain_chain_observable_amount_in_transport",
            "note": "amount weight NOT zeroed; flow-segment transfer amounts retained",
        },
    ]
    for id_field in I_FIELDS:
        rc_actions.append({"action": "leave_key_out_flow_mask", "field": id_field})

    return MaskSpec(
        mask_id=CHAIN_OBSERVABLES_MASK,
        ablation_type="chain_observables_retained",
        fields_masked=CHAIN_OBSERVABLES_FIELDS + ["amount", "timestamp"],
        n_fields_masked=len(CHAIN_OBSERVABLES_FIELDS) + 2,
        aliases=["v3b_supplementary_not_strict_no_all"],
        bitmask=None,
        timestamp_policy=TIMESTAMP_POLICY_DEFAULT,
        connector_actions=connector_actions,
        rc_uot_q_actions=rc_actions,
        expected_connector_status="ACCEPTED_OR_BLOCKED",
        notes=(
            "NOT strict all-evidence missing. Bridge metadata masked on Connector input; "
            "RC-UOT-Q retains chain-observable transfer amount, block/tx timestamps, and graph/flow structure."
        ),
    )


def prepare_chain_observables_flows(
    eth: list[dict[str, Any]],
    bnb: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    """Like v3 receiver+asset+dstChain+I mask but WITHOUT zeroing amount/time weights."""
    from copy import deepcopy

    from cross.baseline_compare.rc_uot_q_pilot import _apply_mask_flow_fields, _renormalize_weights
    from cross.domain.labels.anchor_masking import AnchorMaskMode, mask_matching_flows
    from cross.domain.uot.cost_matrix import default_cost_weights

    eth_f, bnb_f = deepcopy(eth), deepcopy(bnb)
    weights = dict(default_cost_weights())

    for key in ("route", "graph", "novelty"):
        weights[key] = 0.0
    _apply_mask_flow_fields(eth_f, ["addresses", "address_features", "address_count"])
    _apply_mask_flow_fields(bnb_f, ["addresses", "address_features", "address_count"])
    _apply_mask_flow_fields(eth_f, ["token_symbol", "asset_group", "route_type"], neutralize_route_type_to="unknown")
    _apply_mask_flow_fields(bnb_f, ["token_symbol", "asset_group", "route_type"], neutralize_route_type_to="unknown")
    _apply_mask_flow_fields(eth_f, ["route_id", "route_type"], neutralize_route_type_to="unknown")
    _apply_mask_flow_fields(bnb_f, ["route_id", "route_type"], neutralize_route_type_to="unknown")

    eth_f, m1 = mask_matching_flows(eth_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
    bnb_f, m2 = mask_matching_flows(bnb_f, mode=AnchorMaskMode.LEAVE_KEY_OUT)
    weights = _renormalize_weights(weights)
    meta = {
        "mask_id": CHAIN_OBSERVABLES_MASK,
        "variant": "chain_observables_retained",
        "amount_weight_zeroed": False,
        "time_weight_zeroed": False,
        "masked_flow_fields_leave_key_out": sorted(set(m1 + m2)),
    }
    return eth_f, bnb_f, weights, meta
