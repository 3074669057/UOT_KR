"""Fixed RC-UOT-Q decoder shared by all transport solvers.

Freezes all non-solver factors:
  - Quotient grouping
  - Coverage qualification
  - Joint temporal admissibility filter
  - Abstention logic
  - Post-processing

All solvers go through exactly the same decode pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cross.domain.uot.delay_policy import flow_pair_delay_sec
from cross.domain.uot.tx_decode_policy import pick_dst_tx_in_flow
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float

DEFAULT_DELAY_POLICY = "tx_if_available_else_flow_representative"


@dataclass
class DecodeConfig:
    """Configuration for the fixed RC-UOT-Q decoder."""
    strategy: str = "joint_time_admissible_filter"
    delay_policy: str = DEFAULT_DELAY_POLICY
    tx_decode_policy: str = "legacy"
    rescore_with_cost: bool = False
    score_column: str = "transport_mass"
    mass_threshold: float | None = None


@dataclass
class DecodeResult:
    """Output of the fixed decoder."""
    mapping: dict[str, str | None]  # src_tx -> pred_dst_tx or None (abstain)
    meta: dict[str, dict[str, Any]]  # per-src-tx metadata
    n_abstained: int = 0
    n_predicted: int = 0
    flow_coverages: set[int] = field(default_factory=set)


def decode_with_fixed_rc_uot_q(
    T: np.ndarray,
    C: np.ndarray,
    *,
    source_flows: list[dict[str, Any]],
    target_flows: list[dict[str, Any]],
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    truth: dict[str, str],
    eth_ts: dict[str, float],
    bnb_ts: dict[str, float],
    config: DecodeConfig | None = None,
    score_threshold: float | None = None,
) -> DecodeResult:
    """Run the fixed RC-UOT-Q decoder on a transport matrix T.

    This decoder does NOT know which solver produced T.
    It applies:
      1. Row-argmax over target flows (quotient grouping)
      2. Joint temporal admissibility filter (flow + tx level)
      3. Score-threshold abstention (if score_threshold is not None)
      4. Coverage qualification
      5. Per-tx dst tx picking within the selected target flow

    Args:
        T: Transport/assignment matrix [n_source, n_target].
        C: Cost matrix (used for rescoring if rescore_with_cost is True).
        source_flows: Source flow segments.
        target_flows: Target flow segments.
        src_all: Source transactions DataFrame.
        dst_norm: Destination transactions DataFrame (normalized).
        truth: Ground truth mapping {src_tx: dst_tx}.
        eth_ts: ETH transaction timestamps {txhash: timestamp}.
        bnb_ts: BNB transaction timestamps {txhash: timestamp}.
        config: Decoder configuration.
        score_threshold: Optional score threshold for abstention.

    Returns:
        DecodeResult with mapping, metadata, and statistics.
    """
    cfg = config or DecodeConfig()
    p = np.asarray(T, dtype=float)
    n_src, n_dst = p.shape
    tx_to_i = _tx_to_flow_index(source_flows)

    # Precompute tx-to-amount mapping (avoids repeated DataFrame scans)
    _tx_to_amount: dict[str, float] = {}
    if "txhash" in src_all.columns:
        for _, r in src_all.iterrows():
            h = norm_addr(str(r.get("txhash", "")))
            if h:
                try:
                    _tx_to_amount[h] = float(r.get("args.amount", 0.0))
                except (ValueError, TypeError):
                    _tx_to_amount[h] = 0.0

    mapping: dict[str, str | None] = {}
    meta: dict[str, dict[str, Any]] = {}
    n_abstained = 0
    n_predicted = 0
    covered_flows: set[int] = set()

    # Precompute row orders per flow
    _flow_to_order: dict[int, list[int]] = {}
    for _fi in range(n_src):
        _rp = p[_fi]
        if _rp.any():
            _flow_to_order[_fi] = [int(x) for x in np.argsort(-_rp)]

    for src_tx, gt_dst in truth.items():
        base_meta: dict[str, Any] = {
            "abstained": True,
            "flow_i": -1,
            "flow_j": -1,
            "transport_mass": 0.0,
            "reason": "",
        }

        i = tx_to_i.get(src_tx, -1)
        if i < 0 or i >= n_src:
            mapping[src_tx] = None
            base_meta["reason"] = "no_source_flow"
            meta[src_tx] = base_meta
            n_abstained += 1
            continue

        s_ts = eth_ts.get(src_tx, 0.0)
        s_amt = _tx_to_amount.get(src_tx, 0.0)

        # Step 1: Row-argmax (quotient grouping)
        order = _flow_to_order.get(i, [])
        if not order:
            mapping[src_tx] = None
            base_meta["reason"] = "empty_transport_row"
            meta[src_tx] = base_meta
            n_abstained += 1
            continue

        j = order[0]
        mass = float(p[i, j])
        base_meta["flow_i"] = i
        base_meta["flow_j"] = j
        base_meta["transport_mass"] = mass

        # Step 2: Score-threshold abstention
        if score_threshold is not None and mass < score_threshold:
            mapping[src_tx] = None
            base_meta["reason"] = f"below_threshold_{mass:.6f}_<_thr_{score_threshold:.6f}"
            meta[src_tx] = base_meta
            n_abstained += 1
            continue

        # Step 3: Joint temporal admissibility filter
        if cfg.strategy in ("flow_time_admissible_filter", "joint_time_admissible_filter"):
            if i >= 0 and j >= 0 and i < len(source_flows) and j < len(target_flows):
                flow_delay = flow_pair_delay_sec(source_flows[i], target_flows[j], policy=cfg.delay_policy)
                if float(flow_delay) < 0:
                    mapping[src_tx] = None
                    base_meta["reason"] = "flow_time_inadmissible"
                    meta[src_tx] = base_meta
                    n_abstained += 1
                    continue

        # Step 4: Pick dst tx within the selected target flow
        dst_f = target_flows[j] if 0 <= j < len(target_flows) else {}
        dst_pick, eff_policy, used_fallback = pick_dst_tx_in_flow(
            src_tx, float(s_ts), float(s_amt), dst_f, dst_norm,
            policy=cfg.tx_decode_policy,
        )

        if not dst_pick:
            mapping[src_tx] = None
            base_meta["reason"] = "no_dst_tx_in_flow"
            meta[src_tx] = base_meta
            n_abstained += 1
            continue

        # Step 5: Tx-level temporal admissibility
        if cfg.strategy in ("tx_time_admissible_filter", "joint_time_admissible_filter"):
            ts_s = eth_ts.get(src_tx)
            ts_d = bnb_ts.get(norm_addr(dst_pick))
            if ts_s is None or ts_d is None or float(ts_d - ts_s) < 0:
                mapping[src_tx] = None
                base_meta["reason"] = "tx_time_inadmissible"
                meta[src_tx] = base_meta
                n_abstained += 1
                continue

        # Success
        mapping[src_tx] = dst_pick
        base_meta["abstained"] = False
        base_meta["dst_tx"] = dst_pick
        base_meta["tx_decode_policy"] = eff_policy
        base_meta["tx_decode_fallback"] = used_fallback
        meta[src_tx] = base_meta
        n_predicted += 1
        covered_flows.add(i)

    return DecodeResult(
        mapping=mapping,
        meta=meta,
        n_abstained=n_abstained,
        n_predicted=n_predicted,
        flow_coverages=covered_flows,
    )


def _tx_to_flow_index(source_flows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, sf in enumerate(source_flows):
        for txh in sf.get("tx_hashes") or []:
            h = norm_addr(str(txh))
            if h:
                out[h] = i
    return out


def compute_coverage(decode_result: DecodeResult, n_source_flows: int) -> float:
    """Compute flow-level coverage from decode result."""
    if n_source_flows <= 0:
        return 0.0
    return float(len(decode_result.flow_coverages) / max(n_source_flows, 1))


def compute_abstention_rate(decode_result: DecodeResult, n_ground_truth: int) -> float:
    """Compute abstention rate from decode result."""
    if n_ground_truth <= 0:
        return 0.0
    return float(decode_result.n_abstained / max(n_ground_truth, 1))
