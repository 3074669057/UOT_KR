"""Tunable RC-UOT-Q decoder operating points (v3b supplementary; no core edits)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from cross.application.experiments.run_admissible_decoding import _tx_to_flow_index
from cross.domain.evaluation.flow_metrics import pair_precision_recall_f1
from cross.domain.uot.delay_policy import flow_pair_delay_sec
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float


@dataclass(frozen=True)
class OperatingPointParams:
    top_k: int = 1
    transport_mass_threshold: float = 0.0
    confidence_threshold: float = 0.0
    margin_threshold: float = 0.0
    coverage_threshold: float = 0.0
    time_admissible_window_sec: float = 43200.0
    abstention_policy: str = "joint_time"
    flow_pair_aggregation_rule: str = "argmax_mass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def simplicity_score(self) -> int:
        s = 0
        if self.abstention_policy == "joint_time":
            s += 0
        elif self.abstention_policy == "none":
            s += 1
        elif self.abstention_policy == "topk_rescue":
            s += 2
        else:
            s += 3
        s += sum(
            1
            for v in (
                self.transport_mass_threshold,
                self.confidence_threshold,
                self.margin_threshold,
                self.coverage_threshold,
            )
            if v > 0
        )
        if self.top_k > 1:
            s += 1
        return s


def frozen_v3_params() -> OperatingPointParams:
    from cross.baseline_compare.v3b_high_f1.constants import FROZEN_PARAMS

    return OperatingPointParams(**FROZEN_PARAMS)


@dataclass
class SrcCandidateCache:
    src_tx: str
    flow_i: int
    candidates: list[dict[str, Any]]


def _flow_admissible(eth_flows, bnb_flows, i, j, window_sec: float) -> bool:
    if i < 0 or j < 0:
        return False
    d = flow_pair_delay_sec(eth_flows[i], bnb_flows[j], policy="tx_if_available_else_flow_representative")
    if d < 0:
        return False
    if window_sec > 0 and d > window_sec:
        return False
    return True


def _tx_admissible(src_tx, dst_tx, eth_ts, bnb_ts, window_sec: float) -> bool:
    ts_s = eth_ts.get(norm_addr(src_tx))
    ts_d = bnb_ts.get(norm_addr(dst_tx)) if dst_tx else None
    if ts_s is None or ts_d is None:
        return False
    td = float(ts_d - ts_s)
    if td < 0:
        return False
    if window_sec > 0 and td > window_sec:
        return False
    return True


def build_candidate_cache(
    *,
    truth: dict[str, str],
    p: np.ndarray,
    eth_flows: list[dict[str, Any]],
    bnb_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    max_k: int = 10,
) -> list[SrcCandidateCache]:
    tx_to_i = _tx_to_flow_index(eth_flows)
    caches: list[SrcCandidateCache] = []

    for src_tx in truth:
        i = tx_to_i.get(src_tx, -1)
        if i < 0:
            caches.append(SrcCandidateCache(src_tx=src_tx, flow_i=-1, candidates=[]))
            continue

        row = src_all[src_all["txhash"].astype(str).map(norm_addr) == src_tx]
        if row.empty:
            s_ts, s_amt = eth_ts.get(src_tx, 0.0), 0.0
        else:
            r = row.iloc[0]
            s_ts = safe_float(r.get("timestamp"), eth_ts.get(src_tx, 0.0))
            s_amt = safe_float(r.get("args.amount"), 0.0)

        mass_row = p[i]
        row_sum = float(mass_row.sum()) or 1.0
        order = np.argsort(-mass_row).astype(int)[:max_k]
        cands: list[dict[str, Any]] = []
        for j in order:
            j = int(j)
            raw_mass = float(mass_row[j])
            conf = raw_mass / row_sum
            dst, _, _ = pick_dst_tx_in_flow(
                src_tx, float(s_ts), float(s_amt), bnb_flows[j], dst_norm, policy="legacy"
            )
            cands.append(
                {
                    "j": j,
                    "raw_mass": raw_mass,
                    "confidence": conf,
                    "dst": dst or "",
                    "has_dst": bool(dst),
                }
            )
        caches.append(SrcCandidateCache(src_tx=src_tx, flow_i=i, candidates=cands))

    return caches


def decode_with_params(
    cache: SrcCandidateCache,
    params: OperatingPointParams,
    *,
    eth_flows,
    bnb_flows,
    eth_ts,
    bnb_ts,
) -> tuple[str | None, dict[str, Any]]:
    meta: dict[str, Any] = {"abstained": True, "flow_i": cache.flow_i, "flow_j": -1}
    if cache.flow_i < 0 or not cache.candidates:
        return None, meta

    cands = cache.candidates[: params.top_k]
    if len(cands) >= 2:
        margin = cands[0]["confidence"] - cands[1]["confidence"]
    else:
        margin = cands[0]["confidence"] if cands else 0.0

    def passes_thresholds(c: dict[str, Any]) -> bool:
        if c["raw_mass"] < params.transport_mass_threshold:
            return False
        if c["confidence"] < params.confidence_threshold:
            return False
        if margin < params.margin_threshold:
            return False
        return True

    def joint_ok(c: dict[str, Any]) -> bool:
        if not c["has_dst"]:
            return False
        j = int(c["j"])
        if not _flow_admissible(eth_flows, bnb_flows, cache.flow_i, j, params.time_admissible_window_sec):
            return False
        if not _tx_admissible(cache.src_tx, c["dst"], eth_ts, bnb_ts, params.time_admissible_window_sec):
            return False
        return True

    policy = params.abstention_policy
    chosen: dict[str, Any] | None = None

    if policy == "none":
        for c in cands:
            if passes_thresholds(c) and c["has_dst"]:
                chosen = c
                break
    elif policy == "joint_time":
        for c in cands:
            if passes_thresholds(c) and joint_ok(c):
                chosen = c
                break
    elif policy == "confidence":
        for c in cands:
            if passes_thresholds(c) and c["has_dst"]:
                chosen = c
                break
    elif policy == "joint_time_confidence":
        for c in cands:
            if passes_thresholds(c) and joint_ok(c):
                chosen = c
                break
    elif policy == "topk_rescue":
        for c in cands:
            if joint_ok(c):
                chosen = c
                break
    else:
        raise ValueError(f"Unknown abstention_policy: {policy}")

    if chosen is None:
        return None, meta

    meta.update({"abstained": False, "flow_j": int(chosen["j"]), "confidence": chosen["confidence"]})
    return chosen["dst"], meta


def evaluate_on_truth(
    *,
    caches: list[SrcCandidateCache],
    truth: dict[str, str],
    params: OperatingPointParams,
    eth_flows,
    bnb_flows,
    eth_ts,
    bnb_ts,
    label_df: pd.DataFrame,
    split: str,
    mask_id: str,
    operating_point_type: str,
) -> dict[str, Any]:
    mapping: dict[str, str | None] = {}
    meta_by_src: dict[str, dict[str, Any]] = {}
    for c in caches:
        if c.src_tx not in truth:
            continue
        dst, meta = decode_with_params(
            c, params, eth_flows=eth_flows, bnb_flows=bnb_flows, eth_ts=eth_ts, bnb_ts=bnb_ts
        )
        mapping[c.src_tx] = dst
        meta_by_src[c.src_tx] = meta

    pairs = [{"srcTxHash": s, "dstTxHash": d} for s, d in mapping.items() if d]
    pairs_df = pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["srcTxHash", "dstTxHash"])
    eval_labels = label_df.rename(columns={"src_tx_hash": "srcTxHash", "dst_tx_hash": "dstTxHash"})
    pr = pair_precision_recall_f1(pairs_df, eval_labels)

    n_labeled = len(truth)
    n_abstained = sum(1 for s in truth if not mapping.get(s))
    n_predicted = n_labeled - n_abstained
    tp = int(pr.get("tp") or 0)
    fp = int(pr.get("fp") or 0)
    fn = int(pr.get("fn") or 0)

    tx_viol = tx_eval = 0
    covered_flows: set[int] = set()
    for s in truth:
        m = meta_by_src.get(s) or {}
        if m.get("abstained") or not mapping.get(s):
            continue
        i = int(m.get("flow_i", -1))
        if i >= 0:
            covered_flows.add(i)
        pred = mapping.get(s)
        ts_s = eth_ts.get(s)
        ts_d = bnb_ts.get(pred) if pred else None
        if ts_s is not None and ts_d is not None:
            tx_eval += 1
            td = float(ts_d - ts_s)
            if td < 0 or (params.time_admissible_window_sec > 0 and td > params.time_admissible_window_sec):
                tx_viol += 1

    tx_cvr = float(tx_viol / max(tx_eval, 1)) if tx_eval else 0.0
    coverage = float(len(covered_flows) / max(len(eth_flows), 1))

    return {
        "mask_id": mask_id,
        "split": split,
        "operating_point_type": operating_point_type,
        "pair_precision": pr.get("pair_precision"),
        "pair_recall": pr.get("pair_recall"),
        "pair_f1": pr.get("pair_f1"),
        "top3_recall": None,
        "coverage": coverage,
        "abstention_rate": float(n_abstained / max(n_labeled, 1)),
        "tx_CVR": tx_cvr,
        "n_predicted": n_predicted,
        "n_correct": tp,
        "n_false_positive": fp,
        "n_false_negative": fn,
        "n_abstained": n_abstained,
        "selected_params": params.to_dict(),
        "tx_cvr_constraint_satisfied": tx_cvr <= 0.01,
    }


def predictions_df(mapping: dict[str, str | None], truth: dict[str, str]) -> pd.DataFrame:
    rows = [{"src_tx": s, "dst_tx": mapping.get(s) or ""} for s in sorted(truth.keys())]
    return pd.DataFrame(rows)
