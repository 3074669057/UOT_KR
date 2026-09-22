"""Load a trained RankerMLP checkpoint and expose edge_score_fn."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from cross.shared.connector_decimals import decimals_for_eth_bnb
from cross.shared.normalize import norm_addr
from cross.domain.path_b.greedy import EdgeScoreFn

from .features import FEATURE_DIM, compute_base_error_and_features
from .model import RankerMLP


def edge_score_fn_from_checkpoint(
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
    mean = ckpt["feat_mean"]; std = ckpt["feat_std"]; hid = int(ckpt.get("hid", 64))
    model = RankerMLP(FEATURE_DIM, hid=hid)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    want_eth = {str(x).strip().lower() for x in src_all["args.asset_s"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "")}
    want_bnb = {str(x).strip().lower() for x in dst_norm["contractAddress"].tolist() if str(x).strip() and str(x).lower() not in ("nan", "", "0x0000000000000000000000000000000000000000")}
    dec_e, dec_b = decimals_for_eth_bnb(want_eth, want_bnb)
    dst_ts = pd.to_numeric(dst_norm["timeStamp"], errors="coerce").fillna(0)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    def fn(pos: int, dst_hash: str, base_err: float) -> float:
        _, feat = compute_base_error_and_features(
            pos, dst_hash, src_all=src_all, dst_all=dst_norm, dst_ts=dst_ts, dst_window_sec=dst_window_sec,
            dec_e=dec_e, dec_b=dec_b, ratio_by_eth_token=ratio_by_eth_token, median_delay_sec=median_delay_sec,
            delay_weight=delay_weight, receiver_mode=receiver_mode, bnb_bridge_address=bnb_bridge_address,
            ratio_by_eth_bnb_pair=ratio_by_eth_bnb_pair,
        )
        x = torch.tensor(feat, dtype=torch.float32, device=device)
        x = (x - mean_t) / std_t
        with torch.no_grad():
            p = torch.sigmoid(model(x.unsqueeze(0))[0]).item()
        return float(base_err * (2.0 - p))

    return fn


__all__ = ["edge_score_fn_from_checkpoint"]
