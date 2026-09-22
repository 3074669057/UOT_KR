"""Bridge semantic ablation v3 mask registry (spec: docs/bridge_semantic_ablation_v3_spec.md)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

LINEAGE = "Route A v2 + Phase 1 canonical + frozen RC-UOT-Q fixed-delay"

CANONICAL_CONNECTOR_F1 = 0.9736140350877193
REJECTED_ROUTEA_V1_F1 = 0.9953379953379954

TIMESTAMP_POLICY_DEFAULT = "bridge_timestamp_missing_chain_time_retained"
TIMESTAMP_POLICY_STRICT = "strict_timestamp_missing"

# Bitmask order: receiver=1, amount=2, asset_s=4, dstChain=8, timestamp=16
B_FIELD_ORDER: tuple[str, ...] = (
    "receiver",
    "amount",
    "asset_s",
    "dstChain",
    "timestamp",
)
B_FIELD_BITS: dict[str, int] = {name: 1 << i for i, name in enumerate(B_FIELD_ORDER)}

I_FIELDS: tuple[str, ...] = ("event", "bridge", "sender")

ALIAS_TO_CANONICAL: dict[str, str] = {
    "no_receiver": "combo_receiver",
    "no_amount": "combo_amount",
    "no_receiver_no_amount": "combo_receiver_amount",
    "group_temporal_only": "combo_timestamp",
    "group_no_receiver": "combo_receiver",
    "group_no_amount": "combo_amount",
    "group_receiver_amount": "combo_receiver_amount",
    "group_amount_asset": "combo_amount_asset_s",
    "group_asset_route": "combo_asset_s_dstChain",
    "group_route_receiver": "combo_receiver_dstChain",
    "group_matching_core": "combo_receiver_amount_asset_s_dstChain",
}

V2_MASK_EQUIVALENTS: dict[str, str] = {
    "no_receiver": "combo_receiver",
    "no_amount": "combo_amount",
    "no_receiver_no_amount": "combo_receiver_amount",
}

CODE_WEIGHT_KEYS: tuple[str, ...] = (
    "amount",
    "time",
    "route",
    "risk",
    "graph",
    "evidence",
    "novelty",
)

NPZ_COMPONENT_KEYS: tuple[str, ...] = (
    "amount_cost",
    "time_cost",
    "route_cost",
    "risk_cost",
    "graph_cost",
    "evidence_cost",
    "address_novelty_cost",
)

FIELDS_WITHOUT_INDEPENDENT_KNOB: frozenset[str] = frozenset({"asset_s", "dstChain"})


class BridgeSemanticField(StrEnum):
    RECEIVER = "receiver"
    AMOUNT = "amount"
    ASSET_S = "asset_s"
    DST_CHAIN = "dstChain"
    TIMESTAMP = "timestamp"
    EVENT = "event"
    BRIDGE = "bridge"
    SENDER = "sender"


class AblationType(StrEnum):
    FULL = "full"
    ID_ANCHOR = "id_anchor"
    SINGLE = "single"
    COMBINATION = "combination"
    ALL_BRIDGE = "all_bridge"


class ConnectorExpectedStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    ZERO_PREDICTIONS = "ZERO_PREDICTIONS"
    ERROR = "ERROR"
    ACCEPTED_OR_BLOCKED = "ACCEPTED_OR_BLOCKED"
    UNKNOWN_BEFORE_DRY_RUN = "UNKNOWN_BEFORE_DRY_RUN"


class ConnectorObservedStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    ZERO_PREDICTIONS = "ZERO_PREDICTIONS"
    ERROR = "ERROR"
    NOT_PROBED = "NOT_PROBED"


@dataclass
class MaskSpec:
    mask_id: str
    ablation_type: str
    fields_masked: list[str]
    n_fields_masked: int
    aliases: list[str]
    bitmask: int | None
    timestamp_policy: str
    connector_actions: list[dict[str, Any]]
    rc_uot_q_actions: list[dict[str, Any]]
    expected_connector_status: str
    observed_connector_status: str = ConnectorObservedStatus.NOT_PROBED.value
    notes: str = ""
    lineage: str = LINEAGE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fields_masked_set(self) -> frozenset[str]:
        return frozenset(self.fields_masked)


def _combo_mask_id(fields: tuple[str, ...]) -> str:
    return "combo_" + "_".join(fields)


def _bitmask_for_fields(fields: tuple[str, ...]) -> int:
    return sum(B_FIELD_BITS[f] for f in fields)


def _ablation_type_for_b_subset(fields: tuple[str, ...]) -> str:
    return AblationType.SINGLE.value if len(fields) == 1 else AblationType.COMBINATION.value


def _connector_actions_for_fields(fields: frozenset[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if BridgeSemanticField.RECEIVER.value in fields:
        actions.append({"field": "receiver", "action": "clear_string", "column": "args.receiver", "value": ""})
    if BridgeSemanticField.AMOUNT.value in fields:
        actions.append({"field": "amount", "action": "zero_float", "column": "args.amount", "value": 0.0})
    if BridgeSemanticField.ASSET_S.value in fields:
        actions.append({"field": "asset_s", "action": "clear_string", "column": "args.asset_s", "value": ""})
    if BridgeSemanticField.DST_CHAIN.value in fields:
        actions.append({"field": "dstChain", "action": "clear_string", "column": "args.dstChain", "value": ""})
    if BridgeSemanticField.TIMESTAMP.value in fields:
        actions.append({"field": "timestamp", "action": "zero_float", "column": "timestamp", "value": 0.0})
    for id_field in I_FIELDS:
        if id_field in fields:
            actions.append(
                {
                    "field": id_field,
                    "action": "omit_from_wl_adapter",
                    "note": "metadata cleared in audit; not passed to WithdrawLocator columns",
                }
            )
    return actions


def _rc_uot_q_actions_for_fields(
    fields: frozenset[str],
    *,
    timestamp_policy: str = TIMESTAMP_POLICY_DEFAULT,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if BridgeSemanticField.RECEIVER.value in fields:
        actions.extend(
            [
                {"action": "zero_weight", "key": "route", "value": 0.0},
                {"action": "zero_weight", "key": "graph", "value": 0.0},
                {"action": "zero_weight", "key": "novelty", "value": 0.0},
                {"action": "mask_flow_fields", "fields": ["addresses", "address_features", "address_count"]},
            ]
        )
    if BridgeSemanticField.AMOUNT.value in fields:
        actions.append({"action": "zero_weight", "key": "amount", "value": 0.0})
    if BridgeSemanticField.ASSET_S.value in fields:
        actions.append(
            {
                "action": "mask_flow_fields",
                "fields": ["token_symbol", "asset_group", "route_type"],
                "neutralize_route_type_to": "unknown",
                "no_independent_knob": True,
            }
        )
    if BridgeSemanticField.DST_CHAIN.value in fields:
        actions.extend(
            [
                {"action": "zero_weight", "key": "route", "value": 0.0},
                {
                    "action": "mask_flow_fields",
                    "fields": ["route_id", "route_type"],
                    "neutralize_route_type_to": "unknown",
                    "no_independent_knob": True,
                },
            ]
        )
    if BridgeSemanticField.TIMESTAMP.value in fields:
        actions.append({"action": "zero_weight", "key": "time", "value": 0.0})
        if timestamp_policy == TIMESTAMP_POLICY_DEFAULT:
            actions.append(
                {
                    "action": "retain_decode_chain_timestamp_lookup",
                    "policy": TIMESTAMP_POLICY_DEFAULT,
                }
            )
        elif timestamp_policy == TIMESTAMP_POLICY_STRICT:
            actions.append(
                {
                    "action": "disable_decode_chain_timestamp_lookup",
                    "policy": TIMESTAMP_POLICY_STRICT,
                }
            )
    for id_field in I_FIELDS:
        if id_field in fields:
            actions.append({"action": "leave_key_out_flow_mask", "field": id_field})
    return actions


def _expected_connector_status(fields: frozenset[str], mask_id: str) -> str:
    if mask_id == "full_native":
        return ConnectorExpectedStatus.ACCEPTED.value
    if mask_id == "id_anchor_masked":
        return ConnectorExpectedStatus.ACCEPTED.value
    if BridgeSemanticField.RECEIVER.value in fields:
        return ConnectorExpectedStatus.BLOCKED.value
    if mask_id == "combo_amount" and fields == frozenset({"amount"}):
        return ConnectorExpectedStatus.ZERO_PREDICTIONS.value
    if mask_id == "no_all_bridge_semantics":
        return ConnectorExpectedStatus.ACCEPTED_OR_BLOCKED.value
    probe_determined = frozenset(
        {
            BridgeSemanticField.ASSET_S.value,
            BridgeSemanticField.DST_CHAIN.value,
            BridgeSemanticField.TIMESTAMP.value,
        }
    )
    if fields & probe_determined:
        return ConnectorExpectedStatus.ACCEPTED_OR_BLOCKED.value
    if fields - frozenset(I_FIELDS):
        return ConnectorExpectedStatus.ACCEPTED_OR_BLOCKED.value
    return ConnectorExpectedStatus.UNKNOWN_BEFORE_DRY_RUN.value


def _build_b_subset_spec(fields: tuple[str, ...]) -> MaskSpec:
    mask_id = _combo_mask_id(fields)
    fields_set = frozenset(fields)
    aliases = sorted(alias for alias, canonical in ALIAS_TO_CANONICAL.items() if canonical == mask_id)
    return MaskSpec(
        mask_id=mask_id,
        ablation_type=_ablation_type_for_b_subset(fields),
        fields_masked=list(fields),
        n_fields_masked=len(fields),
        aliases=aliases,
        bitmask=_bitmask_for_fields(fields),
        timestamp_policy=TIMESTAMP_POLICY_DEFAULT if "timestamp" in fields else "none",
        connector_actions=_connector_actions_for_fields(fields_set),
        rc_uot_q_actions=_rc_uot_q_actions_for_fields(fields_set),
        expected_connector_status=_expected_connector_status(fields_set, mask_id),
        notes=f"B-subset ablation over {list(fields)}.",
    )


def _build_full_native_spec() -> MaskSpec:
    return MaskSpec(
        mask_id="full_native",
        ablation_type=AblationType.FULL.value,
        fields_masked=[],
        n_fields_masked=0,
        aliases=[],
        bitmask=None,
        timestamp_policy="none",
        connector_actions=[],
        rc_uot_q_actions=[{"action": "frozen_rc_uot_q_reference", "source": "rc_uot_q_frozen"}],
        expected_connector_status=ConnectorExpectedStatus.ACCEPTED.value,
        notes="Phase 1 canonical native bridge semantics; Connector F1 anchor 0.9736.",
    )


def _build_id_anchor_spec() -> MaskSpec:
    fields = list(I_FIELDS)
    fields_set = frozenset(fields)
    return MaskSpec(
        mask_id="id_anchor_masked",
        ablation_type=AblationType.ID_ANCHOR.value,
        fields_masked=fields,
        n_fields_masked=len(fields),
        aliases=["RouteA_v2_id_anchor_masked"],
        bitmask=None,
        timestamp_policy="none",
        connector_actions=_connector_actions_for_fields(fields_set),
        rc_uot_q_actions=_rc_uot_q_actions_for_fields(fields_set),
        expected_connector_status=ConnectorExpectedStatus.ACCEPTED.value,
        notes="ID-anchor fields masked; WithdrawLocator input columns unchanged vs full_native.",
    )


def _build_no_all_bridge_spec() -> MaskSpec:
    fields = list(B_FIELD_ORDER) + list(I_FIELDS)
    fields_set = frozenset(fields)
    return MaskSpec(
        mask_id="no_all_bridge_semantics",
        ablation_type=AblationType.ALL_BRIDGE.value,
        fields_masked=fields,
        n_fields_masked=len(fields),
        aliases=["phase2_extended_not_equivalent"],
        bitmask=sum(B_FIELD_BITS.values()),
        timestamp_policy=TIMESTAMP_POLICY_DEFAULT,
        connector_actions=_connector_actions_for_fields(fields_set),
        rc_uot_q_actions=_rc_uot_q_actions_for_fields(fields_set, timestamp_policy=TIMESTAMP_POLICY_DEFAULT),
        expected_connector_status=ConnectorExpectedStatus.ACCEPTED_OR_BLOCKED.value,
        notes=(
            "Extends Phase 2 fair anchor-masked condition with B∪I and explicit timestamp policy B; "
            "not equivalent to Phase 2 (Phase 2 preserved timestamp)."
        ),
    )


def _build_registry() -> dict[str, MaskSpec]:
    specs: dict[str, MaskSpec] = {}
    specs["full_native"] = _build_full_native_spec()
    specs["id_anchor_masked"] = _build_id_anchor_spec()
    for r in range(1, len(B_FIELD_ORDER) + 1):
        for combo in combinations(B_FIELD_ORDER, r):
            spec = _build_b_subset_spec(combo)
            specs[spec.mask_id] = spec
    specs["no_all_bridge_semantics"] = _build_no_all_bridge_spec()
    if len(specs) != 34:
        raise RuntimeError(f"Expected 34 canonical masks, got {len(specs)}")
    return specs


_REGISTRY: dict[str, MaskSpec] | None = None


def _registry() -> dict[str, MaskSpec]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def iter_b_subset_mask_specs() -> Iterator[MaskSpec]:
    for mask_id, spec in sorted(_registry().items(), key=lambda x: (x[1].bitmask or 0, x[0])):
        if spec.ablation_type in (AblationType.SINGLE.value, AblationType.COMBINATION.value):
            yield spec


def iter_main_mask_specs() -> Iterator[MaskSpec]:
    order = ["full_native", "id_anchor_masked"]
    order.extend(_combo_mask_id(c) for r in range(1, len(B_FIELD_ORDER) + 1) for c in combinations(B_FIELD_ORDER, r))
    order.append("no_all_bridge_semantics")
    reg = _registry()
    for mask_id in order:
        yield reg[mask_id]


def get_mask_spec(mask_id: str) -> MaskSpec:
    reg = _registry()
    canonical = ALIAS_TO_CANONICAL.get(mask_id, mask_id)
    if canonical not in reg:
        raise KeyError(f"Unknown mask_id or alias: {mask_id!r}")
    return reg[canonical]


def resolve_canonical_mask_id(mask_id: str) -> str:
    return ALIAS_TO_CANONICAL.get(mask_id, mask_id)


def write_mask_specs(output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    written: list[Path] = []
    for spec in iter_main_mask_specs():
        path = output_dir / "masks" / spec.mask_id / "spec.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def apply_connector_mask_to_row(
    native_row: dict[str, Any],
    fields_masked: frozenset[str],
) -> dict[str, Any]:
    """Apply B-field masks to a WithdrawLocator adapter row (does not mutate input)."""
    row = dict(native_row)
    if BridgeSemanticField.RECEIVER.value in fields_masked:
        row["args.receiver"] = ""
    if BridgeSemanticField.AMOUNT.value in fields_masked:
        row["args.amount"] = 0.0
    if BridgeSemanticField.ASSET_S.value in fields_masked:
        row["args.asset_s"] = ""
    if BridgeSemanticField.DST_CHAIN.value in fields_masked:
        row["args.dstChain"] = ""
    if BridgeSemanticField.TIMESTAMP.value in fields_masked:
        row["timestamp"] = 0.0
    return row


def connector_row_unchanged_for_id_anchor(native_row: dict[str, Any], masked_row: dict[str, Any]) -> bool:
    """ID-anchor mask must not change WL-visible columns."""
    keys = (
        "txhash",
        "timestamp",
        "args.receiver",
        "args.amount",
        "args.asset_s",
        "args.srcChain",
        "args.dstChain",
    )
    return all(native_row.get(k) == masked_row.get(k) for k in keys)


def status_expectation_matches(
    expected: str,
    observed: str,
) -> tuple[bool, bool]:
    """Return (matches, anomaly)."""
    if observed == ConnectorObservedStatus.ERROR.value:
        return expected == ConnectorExpectedStatus.ERROR.value, expected != ConnectorExpectedStatus.ERROR.value
    if expected in (
        ConnectorExpectedStatus.ACCEPTED_OR_BLOCKED.value,
        ConnectorExpectedStatus.UNKNOWN_BEFORE_DRY_RUN.value,
    ):
        return observed in (
            ConnectorObservedStatus.ACCEPTED.value,
            ConnectorObservedStatus.BLOCKED.value,
            ConnectorObservedStatus.ZERO_PREDICTIONS.value,
        ), False
    return expected == observed, expected != observed


def intended_weight_changes(rc_uot_q_actions: list[dict[str, Any]]) -> dict[str, float]:
    changes: dict[str, float] = {}
    for act in rc_uot_q_actions:
        if act.get("action") == "zero_weight":
            changes[str(act["key"])] = float(act.get("value", 0.0))
    return changes


def fields_with_no_independent_knob(fields_masked: frozenset[str]) -> list[str]:
    return sorted(f for f in fields_masked if f in FIELDS_WITHOUT_INDEPENDENT_KNOB)


def build_rc_uot_q_masking_audit(
    spec: MaskSpec,
    *,
    production_audit: dict[str, Any],
) -> dict[str, Any]:
    requested = list(spec.fields_masked)
    intended = intended_weight_changes(spec.rc_uot_q_actions)
    code_keys = set(production_audit.get("code_weight_keys") or CODE_WEIGHT_KEYS)
    config_keys = set(production_audit.get("config_weight_keys") or [])
    npz_keys = set(production_audit.get("npz_component_keys") or [])

    available_weight_knobs = sorted(code_keys & set(CODE_WEIGHT_KEYS))
    unavailable_weight_knobs: list[str] = []
    for key in intended:
        if key not in code_keys:
            unavailable_weight_knobs.append(key)

    weight_to_component = {
        "amount": "amount_cost",
        "time": "time_cost",
        "route": "route_cost",
        "risk": "risk_cost",
        "graph": "graph_cost",
        "evidence": "evidence_cost",
        "novelty": "address_novelty_cost",
    }
    missing_npz: list[str] = []
    for key in intended:
        comp = weight_to_component.get(key)
        if comp and npz_keys and comp not in npz_keys:
            missing_npz.append(comp)

    no_knob_fields = fields_with_no_independent_knob(spec.fields_masked_set())
    implementable = not unavailable_weight_knobs and not missing_npz

    return {
        "mask_id": spec.mask_id,
        "requested_fields_masked": requested,
        "intended_weight_changes": intended,
        "actual_available_weight_knobs": available_weight_knobs,
        "config_weight_keys": sorted(config_keys),
        "code_weight_keys": sorted(code_keys),
        "unavailable_weight_knobs": unavailable_weight_knobs,
        "npz_component_keys_present": sorted(npz_keys),
        "missing_npz_components_for_intended_weights": missing_npz,
        "no_independent_knob_fields": no_knob_fields,
        "no_independent_knob": bool(no_knob_fields),
        "rc_uot_q_actions": spec.rc_uot_q_actions,
        "implementable": implementable if npz_keys else None,
        "implementable_note": (
            "NPZ not found; weight keys validated against code defaults only."
            if not npz_keys
            else None
        ),
        "would_touch_production_uot_base": False,
        "production_read_only": True,
    }
