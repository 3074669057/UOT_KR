"""Graph ranker inference adapter -> Path B edge_score_fn."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.domain.path_b.greedy import EdgeScoreFn

from .features import GRAPH_EDGE_DIM, candidate_hashes_for_src
from .model import TemporalGraphRanker
from cross.domain.ranker.features import compute_base_error_and_features


def edge_score_fn_from_graph_checkpoint(
    checkpoint_path: Path | str,
    *,
    src_all: pd.DataFrame,
    dst_norm: pd.DataFrame,
    dst_window_sec: float,
    ratio_by_eth_token: dict[str, float] | None,
    median_delay_sec: float | None,
    delay_weight: float,
    receiver_mode: str,
    bnb_bridge_address: str,
    ratio_by_eth_bnb_pair: dict[tuple[str, str], float] | None = None,
) -> EdgeScoreFn:
    path = Path(checkpoint_path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta") or {}
    hid = int(meta.get("hid", 96)); heads = int(meta.get("heads", 4)); top_k = int(meta.get("top_k", 64))
    model = TemporalGraphRanker(GRAPH_EDGE_DIM, hid=hid, heads=heads)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    want_eth = {str(x).strip().lower() for x in src_all["args.asset_s"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "")}
    want_bnb = {str(x).strip().lower() for x in dst_norm["contractAddress"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "", "0x0000000000000000000000000000000000000000")}
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)
    dst_ts = pd.to_numeric(dst_norm["timeStamp"], errors="coerce").fillna(0)
    per_src_probs: dict[int, dict[str, float]] = {}

    def _compute_for_src(pos: int) -> dict[str, float]:
        if pos in per_src_probs:
            return per_src_probs[pos]
        eth_from = norm_addr(src_all.iloc[pos].get("args.receiver", ""))
        cands = candidate_hashes_for_src(pos, src_all, dst_norm, dst_ts, dst_window_sec, receiver_mode, eth_from)[:top_k]
        if not cands:
            per_src_probs[pos] = {}
            return {}
        feats = []
        for j, h in enumerate(cands):
            _, f = compute_base_error_and_features(
                pos, h, src_all=src_all, dst_all=dst_norm, dst_ts=dst_ts, dst_window_sec=dst_window_sec,
                dec_e=dec_e, dec_b=dec_b, ratio_by_eth_token=ratio_by_eth_token, median_delay_sec=median_delay_sec,
                delay_weight=delay_weight, receiver_mode=receiver_mode, bnb_bridge_address=bnb_bridge_address,
                ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
            )
            tail = np.array([float(pos / max(len(src_all), 1)), float((j + 1) / max(len(cands), 1)), float(len(cands) / max(top_k, 1)), float(f[2]), float(f[4]), float(f[8])], dtype=np.float32)
            feats.append(np.concatenate([f.astype(np.float32), tail], axis=0))
        x = torch.tensor(np.stack(feats, axis=0)[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            prob = torch.softmax(model(x)[0], dim=0).detach().cpu().numpy()
        out = {norm_addr(h): float(p) for h, p in zip(cands, prob)}
        per_src_probs[pos] = out
        return out

    def fn(pos: int, dst_hash: str, base_err: float) -> float:
        p = float(_compute_for_src(pos).get(norm_addr(dst_hash), 0.0))
        return float(base_err * (2.0 - p))

    return fn


__all__ = ["edge_score_fn_from_graph_checkpoint"]
