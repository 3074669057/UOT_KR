"""Anchor-field masking for leave-the-anchor-out ablation.

Ground-truth labels may retain bridge anchors; matching / cost-matrix inputs must not.
Two masking tiers:
  - leave_key_out: direct bridge keys / pair ids only
  - leave_anchor_out_strict: keys + bridge-/anchor-/label-derived evidence features
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from cross.domain.labels.feature_provenance import (
    feature_provenance_lookup,
    forbidden_in_strict_mode,
    is_bridge_derived_feature,
)


class AnchorMaskMode(StrEnum):
    NONE = "none"
    LEAVE_KEY_OUT = "leave_key_out"
    LEAVE_ANCHOR_OUT_STRICT = "leave_anchor_out_strict"


# Direct bridge / pair keys (leave_key_out).
LEAVE_KEY_OUT_PATTERNS: tuple[str, ...] = (
    r"message[_]?key",
    r"bridge[_]?message[_]?key",
    r"celer[_]?message[_]?key",
    r"transfer[_]?id",
    r"bridge[_]?transfer[_]?id",
    r"anchor[_]?id",
    r"tx[_]?anchor(?:[_]?id)?",
    r"tx[_]?anchor[_]?id",
    r"evidence[_]?anchor",
    r"bridge[_]?event[_]?id",
    r"ground[_]?truth[_]?pair",
    r"label[_]?pair",
    r"paired[_]?tx",
    r"matched[_]?tx",
    r"fake[_]?oracle[_]?pair[_]?id",
    r"fake[_]?message[_]?key",
    r"fake[_]?bridge[_]?pair",
)

# Additional derived / bridge-evidence fields (strict only).
DERIVED_STRICT_PATTERNS: tuple[str, ...] = (
    r"evidence[_]?level",
    r"evidence[_]?levels",
    r"evidence[_]?quality(?:[_]?score)?",
    r"bridge[_]?evidence",
    r"bridge[_]?label(?:[_]?match)?",
    r"bridge[_]?consistency",
    r"bridge[_]?contract[_]?hit",
    r"bridge[_]?prior",
    r"celer[_]?evidence",
    r"celer[_]?label",
    r"anchor[_]?score",
    r"anchor[_]?confidence",
    r"tx[_]?anchor[_]?score",
    r"flow[_]?label[_]?score",
    r"covered[_]?by[_]?bridge",
    r"covered[_]?subset[_]?flag",
    r"is[_]?bridge[_]?covered",
    r"oracle[_]?match",
    r"oracle[_]?candidate",
    r"label[_]?candidate",
    r"tracking[_]?src[_]?txhash",
    r"tracking[_]?dst[_]?txhash",
    r"label[_]?dst[_]?txhash",
    r"boost[_]?label[_]?dst",
    r"true[_]?dst",
    r"expected[_]?dst",
    r"src[_]?dst[_]?pair",
    r"source[_]?target[_]?pair",
    r"dst[_]?hash[_]?from[_]?anchor",
    r"target[_]?hash[_]?from[_]?anchor",
    r"exact[_]?bridge[_]?pair",
    r"event[_]?nonce",
    r"nonce[_]?as[_]?pair[_]?key",
    r"celer[_]?pair[_]?key",
    r"bridge[_]?pair[_]?key",
)

# Non-anchor features allowed in leave_anchor_out_strict matching.
MATCHING_FEATURES_ALLOWED_STRICT: tuple[str, ...] = (
    "flow_id",
    "chain",
    "amount_usd",
    "start_time",
    "end_time",
    "aml_score",
    "aml_risk_score_raw",
    "aml_rule_hits",
    "aml_risk_level",
    "route_type",
    "route_id",
    "asset_group",
    "token_symbol",
    "price_snapshot_ok",
    "graph_embedding",
    "address_set",
    "tx_count",
    "raw_amount_sum",
    "human_amount_sum",
    "bridge",
    "tx_hashes",
)

MATCHING_FEATURES_FORBIDDEN_STRICT: tuple[str, ...] = LEAVE_KEY_OUT_PATTERNS + DERIVED_STRICT_PATTERNS

_COVERED_SUBSET_DEFINITION = (
    "Labeled src flows with accepted bridge-evidence flow labels (celer_label / flow_labels); "
    "evaluation reads ground truth only; matching never reads label pairs or bridge-derived evidence."
)

COVERED_SUBSET_DEFINITION = _COVERED_SUBSET_DEFINITION

_SRC_TOKENS = frozenset({"src", "source", "from"})
_DST_TOKENS = frozenset({"dst", "target", "to"})
_HASH_TOKENS = frozenset({"hash", "txhash", "tx_hash", "txhashes"})


def _normalize_field_name(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name or ""))
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _compiled(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def _matches_patterns(norm: str, patterns: tuple[str, ...]) -> bool:
    for pat in _compiled(patterns):
        if pat.search(norm):
            return True
    return False


def _is_src_dst_hash_pair_field(norm: str) -> bool:
    parts = [p for p in norm.split("_") if p]
    if len(parts) < 3:
        return False
    has_src = any(t in parts for t in _SRC_TOKENS)
    has_dst = any(t in parts for t in _DST_TOKENS)
    has_hash = any(t in parts for t in _HASH_TOKENS) or any(p.endswith("hash") for p in parts)
    pairish = "pair" in parts or "match" in parts or "anchor" in parts
    return has_src and has_dst and has_hash and pairish


def is_forbidden_field(name: str, mode: AnchorMaskMode | str) -> bool:
    """Return True if field must be removed for the given masking mode."""
    mode_s = AnchorMaskMode(str(mode or AnchorMaskMode.NONE))
    if mode_s == AnchorMaskMode.NONE:
        return False
    norm = _normalize_field_name(name)
    if not norm:
        return False
    if _matches_patterns(norm, LEAVE_KEY_OUT_PATTERNS):
        return True
    if _is_src_dst_hash_pair_field(norm):
        return True
    if mode_s == AnchorMaskMode.LEAVE_KEY_OUT:
        if norm.startswith("fake_"):
            return True
        return False
    # strict
    if _matches_patterns(norm, DERIVED_STRICT_PATTERNS):
        return True
    if forbidden_in_strict_mode(norm):
        return True
    if "anchor" in norm and norm not in {"amount_usd"}:
        return True
    if norm in {"paired_flow_id", "matched_flow_id", "gt_dst_flow_id", "oracle_dst_flow_id"}:
        return True
    return False


# Backward-compatible alias
def is_anchor_field(name: str, *, strict: bool = True) -> bool:
    mode = AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT if strict else AnchorMaskMode.LEAVE_KEY_OUT
    return is_forbidden_field(name, mode)


def mask_anchor_fields(
    obj: dict[str, Any],
    *,
    mode: AnchorMaskMode | str = AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT,
) -> tuple[dict[str, Any], list[str]]:
    out = deepcopy(obj)
    masked: list[str] = []
    for key in list(out.keys()):
        if is_forbidden_field(str(key), mode):
            out.pop(key, None)
            masked.append(str(key))
    return out, masked


def mask_anchor_dataframe(
    df: pd.DataFrame,
    *,
    mode: AnchorMaskMode | str = AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT,
) -> tuple[pd.DataFrame, list[str]]:
    masked_cols = [c for c in df.columns if is_forbidden_field(str(c), mode)]
    out = df.drop(columns=masked_cols, errors="ignore").copy()
    return out, [str(c) for c in masked_cols]


def mask_matching_flows(
    flows: list[dict[str, Any]],
    *,
    mode: AnchorMaskMode | str = AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT,
) -> tuple[list[dict[str, Any]], list[str]]:
    masked_all: list[str] = []
    out: list[dict[str, Any]] = []
    for f in flows:
        cleaned, masked = mask_anchor_fields(f, mode=mode)
        masked_all.extend(masked)
        out.append(cleaned)
    return out, sorted(set(masked_all))


def collect_schema_keys(flows: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for f in flows:
        keys.update(str(k) for k in f.keys())
    return sorted(keys)


def scan_matching_features_for_leakage(
    features: list[dict[str, Any]],
    *,
    mode: AnchorMaskMode | str = AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT,
    audit_strict: bool = True,
) -> dict[str, Any]:
    """Pre-matching leakage scan on candidate / cost-matrix feature dicts."""
    violations: list[dict[str, str]] = []
    mode_s = AnchorMaskMode(str(mode or AnchorMaskMode.NONE))
    for idx, feat in enumerate(features):
        for key, val in feat.items():
            if val is None or val == "" or val == []:
                continue
            if is_forbidden_field(str(key), mode_s):
                violations.append({"index": str(idx), "field": str(key), "reason": "forbidden_field_present"})
                continue
            prov = feature_provenance_lookup(str(key))
            if mode_s == AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT and audit_strict:
                if prov.get("allowed_in_leave_anchor_out_strict", True):
                    continue
                if prov.get("derived_from_bridge_event") or prov.get("derived_from_tx_anchor"):
                    violations.append({"index": str(idx), "field": str(key), "reason": "bridge_or_anchor_derived"})
                elif prov.get("derived_from_flow_label") or prov.get("derived_from_ground_truth"):
                    violations.append({"index": str(idx), "field": str(key), "reason": "label_or_gt_derived"})
    passed = len(violations) == 0
    return {
        "leakage_scan_passed": passed,
        "violations": violations,
        "mode": str(mode_s),
        "audit_strict": bool(audit_strict),
        "n_features_scanned": len(features),
    }


def count_allowed_forbidden_fields(
    schema: list[str],
    *,
    mode: AnchorMaskMode | str,
) -> tuple[int, int]:
    allowed = forbidden = 0
    for k in schema:
        if is_forbidden_field(k, mode):
            forbidden += 1
        else:
            allowed += 1
    return allowed, forbidden


def resolve_anchor_mask_mode(
    *,
    leave_anchor_out: bool = False,
    anchor_mask_mode: str | None = None,
    anchor_mask_strict: bool = True,
) -> AnchorMaskMode:
    if anchor_mask_mode and str(anchor_mask_mode).strip().lower() not in ("", "none"):
        return AnchorMaskMode(str(anchor_mask_mode).strip().lower())
    if leave_anchor_out:
        if anchor_mask_strict:
            return AnchorMaskMode.LEAVE_ANCHOR_OUT_STRICT
        return AnchorMaskMode.LEAVE_KEY_OUT
    return AnchorMaskMode.NONE


def write_anchor_mask_report(
    before_schema: list[str],
    after_schema: list[str],
    out_path: Path,
    *,
    masked_fields: list[str] | None = None,
    leakage_scan: dict[str, Any] | None = None,
    mode: str | None = None,
) -> None:
    removed = sorted(set(before_schema) - set(after_schema))
    report = {
        "mode": mode,
        "before_schema": before_schema,
        "after_schema": after_schema,
        "removed_fields": removed,
        "masked_field_count": len(set(masked_fields or removed)),
        "masked_fields": sorted(set(masked_fields or removed)),
        "leakage_scan": leakage_scan or {},
        "matching_features_allowed_strict": list(MATCHING_FEATURES_ALLOWED_STRICT),
        "matching_features_forbidden_strict": list(MATCHING_FEATURES_FORBIDDEN_STRICT),
        "covered_subset_definition": _COVERED_SUBSET_DEFINITION,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_masked_fields_csv(masked_fields: list[str], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"field_name": sorted(set(masked_fields))}).to_csv(out_path, index=False)


def covered_subset_metrics_from_cmp(cmp: dict[str, Any]) -> dict[str, Any]:
    fm = (cmp.get("uot") or {}).get("flow_metrics") or {}
    uot = cmp.get("uot") or {}
    um_s = uot.get("unmatched_source_mass") or []
    n_src = int(uot.get("n_eth_flows") or 0)
    n_dst = int(uot.get("n_bnb_flows") or 0)
    abstention = float(sum(float(x) for x in um_s) / max(len(um_s), 1)) if um_s else 0.0
    abstention = max(0.0, min(1.0, abstention))
    top3 = fm.get("top3_flow_correspondence_accuracy")
    pair_rec = fm.get("pair_recall")
    meta = cmp.get("anchor_mask_meta") or {}
    schema = list(meta.get("after_schema") or meta.get("before_schema") or [])
    mode = meta.get("mode") or AnchorMaskMode.NONE
    if schema:
        used_n, forb_n = count_allowed_forbidden_fields(schema, mode=mode)
    else:
        used_n, forb_n = None, None
    return {
        "pair_precision": fm.get("pair_precision"),
        "pair_recall": pair_rec,
        "pair_f1": fm.get("pair_f1"),
        "top1_recall": pair_rec,
        "top3_recall": top3,
        "flow_mass_recall": fm.get("flow_mass_recall"),
        "top3_flow_correspondence_accuracy": top3,
        "coverage": pair_rec,
        "abstention_rate": abstention,
        "unmatched_mass_ratio": fm.get("unmatched_mass_ratio"),
        "n_source_flows": n_src,
        "n_target_flows": n_dst,
        "used_feature_count": used_n,
        "forbidden_feature_count": forb_n,
        "leakage_scan_passed": meta.get("leakage_scan_passed"),
    }


def metrics_delta(baseline: dict[str, Any], masked: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in baseline:
        b = baseline.get(key)
        m = masked.get(key)
        if isinstance(b, (int, float)) and isinstance(m, (int, float)):
            delta[key] = float(m) - float(b)
        elif b is not None and m is not None:
            delta[key] = m
    return delta


def inject_fake_perfect_anchor_fields(
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    label_df: pd.DataFrame,
) -> list[str]:
    """Negative-control probe: inject oracle fields that encode true pairings."""
    from cross.shared.normalize import norm_addr

    dst_to_flow: dict[str, str] = {}
    for f in bnb_flows:
        fid = str(f.get("flow_id") or "")
        for h in f.get("tx_hashes") or []:
            dst_to_flow[norm_addr(str(h))] = fid

    tx_to_flow: dict[str, str] = {}
    for f in eth_flows:
        fid = str(f.get("flow_id") or "")
        for h in f.get("tx_hashes") or []:
            tx_to_flow[norm_addr(str(h))] = fid

    truth: dict[str, str] = {}
    for _, r in label_df.iterrows():
        s = norm_addr(r.get("srcTxhash", r.get("srcTxHash", "")))
        d = norm_addr(r.get("dstTxhash", r.get("dstTxHash", "")))
        if s and d:
            truth[s] = d

    injected: list[str] = []
    for f in eth_flows:
        src_txs = [norm_addr(str(h)) for h in f.get("tx_hashes") or []]
        dst_flow = ""
        for st in src_txs:
            d_tx = truth.get(st, "")
            if d_tx:
                dst_flow = dst_to_flow.get(d_tx, "")
                if dst_flow:
                    break
        if not dst_flow:
            continue
        f["fake_oracle_pair_id"] = dst_flow
        f["fake_message_key"] = dst_flow
        f["fake_bridge_pair"] = dst_flow
        injected.extend(["fake_oracle_pair_id", "fake_message_key", "fake_bridge_pair"])
    return sorted(set(injected))
