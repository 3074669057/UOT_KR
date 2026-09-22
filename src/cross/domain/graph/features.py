"""Temporal-graph style edge features built from Path B candidates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.utils.safe_cast import safe_float
from cross.domain.ranker.features import FEATURE_DIM, compute_base_error_and_features

GRAPH_EDGE_DIM = FEATURE_DIM + 6


@dataclass
class GraphSrcBatch:
    src_txhash: str
    candidate_hashes: list[str]
    edge_features: np.ndarray
    labels: np.ndarray
    base_errors: np.ndarray


def candidate_hashes_for_src(
    pos: int,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    dst_ts: pd.Series,
    dst_window_sec: float,
    receiver_mode: str,
    eth_from: str,
) -> list[str]:
    row = src_all.iloc[pos]
    ts = safe_float(row.get("timestamp"), 0.0)
    mask = (dst_ts > ts) & (dst_ts <= ts + dst_window_sec)
    sl = dst_norm.loc[mask].copy()
    if sl.empty:
        sl = dst_norm.copy()
    recv = norm_addr(eth_from)
    if (receiver_mode or "eth_from").strip().lower() != "bnb_pick_per_candidate":
        sl = sl[sl["to"].map(norm_addr) == recv]
    return sorted({norm_addr(h) for h in sl["hash"].astype(str)})


def _augment_edge_feature(feat: np.ndarray, *, src_pos_norm: float, cand_rank_norm: float, cand_count_norm: float) -> np.ndarray:
    tail = np.array([src_pos_norm, cand_rank_norm, cand_count_norm, feat[2], feat[4], feat[8]], dtype=np.float32)
    return np.concatenate([feat.astype(np.float32), tail], axis=0)


def build_graph_training_batches(
    *,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    labels: pd.DataFrame,
    ratio_by_eth_token: dict[str, float] | None,
    median_delay_sec: float | None,
    delay_weight: float,
    dst_window_sec: float,
    receiver_mode: str,
    bnb_bridge_address: str,
    top_k: int = 64,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
) -> list[GraphSrcBatch]:
    want_eth = {str(x).strip().lower() for x in src_all["args.asset_s"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "")}
    want_bnb = {str(x).strip().lower() for x in dst_norm["contractAddress"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "", "0x0000000000000000000000000000000000000000")}
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)
    dst_ts = pd.to_numeric(dst_norm["timeStamp"], errors="coerce").fillna(0)
    truth_by_src = {norm_addr(r.get("srcTxhash", "")): norm_addr(r.get("dstTxhash", "")) for _, r in labels.iterrows()}
    out: list[GraphSrcBatch] = []
    n_src = max(len(src_all), 1)
    for pos in range(len(src_all)):
        src_h = norm_addr(src_all.iloc[pos].get("txhash", ""))
        true_h = truth_by_src.get(src_h, "")
        if not src_h or not true_h:
            continue
        eth_from = norm_addr(src_all.iloc[pos].get("args.receiver", ""))
        cands = candidate_hashes_for_src(pos, src_all, dst_norm, dst_ts, dst_window_sec, receiver_mode, eth_from)
        if true_h not in cands:
            cands = [true_h] + cands
        cands = cands[:top_k]
        if not cands:
            continue
        feats: list[np.ndarray] = []
        labels_y: list[float] = []
        base_errs: list[float] = []
        k = len(cands)
        for j, h in enumerate(cands):
            base_err, feat = compute_base_error_and_features(
                pos, h, src_all=src_all, dst_all=dst_norm, dst_ts=dst_ts, dst_window_sec=dst_window_sec,
                dec_e=dec_e, dec_b=dec_b, ratio_by_eth_token=ratio_by_eth_token, median_delay_sec=median_delay_sec,
                delay_weight=delay_weight, receiver_mode=receiver_mode, bnb_bridge_address=bnb_bridge_address,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            )
            f = _augment_edge_feature(
                feat,
                src_pos_norm=float(pos / n_src),
                cand_rank_norm=float((j + 1) / max(k, 1)),
                cand_count_norm=float(k / max(top_k, 1)),
            )
            feats.append(f); labels_y.append(1.0 if h == true_h else 0.0); base_errs.append(float(base_err))
        out.append(GraphSrcBatch(src_txhash=src_h, candidate_hashes=cands, edge_features=np.stack(feats, axis=0).astype(np.float32), labels=np.array(labels_y, dtype=np.float32), base_errors=np.array(base_errs, dtype=np.float32)))
    return out


__all__ = ["GRAPH_EDGE_DIM", "GraphSrcBatch", "candidate_hashes_for_src", "build_graph_training_batches"]
